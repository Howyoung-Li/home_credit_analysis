from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import lightgbm as lgb
import numpy as np
import pandas as pd


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

PDO = 50
BASE_SCORE = 600
BASE_ODDS = 20
DECLINE_PCT = 0.10
MANUAL_REVIEW_PCT = 0.10


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        rank_sum = np.bincount(inverse, weights=ranks)
        avg_rank = rank_sum / counts
        ranks = avg_rank[inverse]
    n_pos = float(y.sum())
    n_neg = float(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ks_score(y: np.ndarray, risk_score: np.ndarray) -> float:
    order = np.argsort(-risk_score)
    y_sorted = y[order]
    bad_total = y_sorted.sum()
    good_total = len(y_sorted) - bad_total
    if bad_total == 0 or good_total == 0:
        return np.nan
    bad_cum = np.cumsum(y_sorted) / bad_total
    good_cum = np.cumsum(1 - y_sorted) / good_total
    return float(np.max(np.abs(bad_cum - good_cum)))


def psi_score(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    reference = reference[np.isfinite(reference)]
    target = target[np.isfinite(target)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1))).astype(float)
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    tgt_counts, _ = np.histogram(target, bins=edges)
    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    tgt_dist = tgt_counts / max(tgt_counts.sum(), 1)
    eps = 1e-6
    return float(np.sum((tgt_dist - ref_dist) * np.log((tgt_dist + eps) / (ref_dist + eps))))


def pd_to_score(pd_values: np.ndarray) -> np.ndarray:
    pd_values = np.clip(pd_values, 1e-6, 1 - 1e-6)
    odds = (1 - pd_values) / pd_values
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return offset + factor * np.log(odds)


def add_score_grades(scored: pd.DataFrame, reference_split: str = "validation") -> pd.DataFrame:
    reference = scored[scored["screening_split"] == reference_split]["a_card_score"]
    quantiles = np.quantile(reference, np.linspace(0, 1, 11))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        raise ValueError("Not enough unique score cutoffs to create grades.")
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    labels = [f"G{idx:02d}" for idx in range(1, len(quantiles))]
    scored = scored.copy()
    scored["score_grade"] = pd.cut(scored["a_card_score"], bins=quantiles, labels=labels, include_lowest=True)
    scored["score_grade"] = scored["score_grade"].astype("string")
    return scored


def strategy_cutoffs(scored: pd.DataFrame) -> tuple[float, float]:
    validation = scored[scored["screening_split"] == "validation"]
    decline_cutoff = float(np.quantile(validation["pd_lgbm"], 1 - DECLINE_PCT))
    manual_cutoff = float(np.quantile(validation["pd_lgbm"], 1 - DECLINE_PCT - MANUAL_REVIEW_PCT))
    return manual_cutoff, decline_cutoff


def assign_strategy(scored: pd.DataFrame, manual_cutoff: float, decline_cutoff: float) -> pd.DataFrame:
    scored = scored.copy()
    scored["strategy_action"] = np.select(
        [
            scored["pd_lgbm"] >= decline_cutoff,
            scored["pd_lgbm"] >= manual_cutoff,
        ],
        [
            "decline",
            "manual_review",
        ],
        default="approve",
    )
    return scored


def labeled_metrics(split_name: str, part: pd.DataFrame) -> dict[str, float | int | str]:
    y = part["TARGET"].to_numpy(dtype=float)
    risk = part["pd_lgbm"].to_numpy(dtype=float)
    row: dict[str, float | int | str] = {
        "split": split_name,
        "rows": len(part),
        "bad_count": int(y.sum()),
        "bad_rate": float(y.mean()),
        "auc": auc_score(y, risk),
        "ks": ks_score(y, risk),
        "avg_pd": float(risk.mean()),
        "avg_score": float(part["a_card_score"].mean()),
    }
    for pct in [0.01, 0.05, 0.10, 0.20]:
        cutoff = max(int(np.ceil(len(part) * pct)), 1)
        top_idx = np.argsort(-risk)[:cutoff]
        top_bad = y[top_idx].sum()
        row[f"top_{int(pct * 100)}pct_bad_rate"] = float(top_bad / cutoff)
        row[f"top_{int(pct * 100)}pct_bad_capture"] = float(top_bad / max(y.sum(), 1))
        row[f"top_{int(pct * 100)}pct_lift"] = float((top_bad / cutoff) / max(y.mean(), 1e-9))
    return row


def score_band_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, part in scored[scored["TARGET"].notna()].groupby("screening_split"):
        total_bad = max(part["TARGET"].sum(), 1)
        for grade, grade_part in part.groupby("score_grade", observed=False):
            rows.append(
                {
                    "split": split_name,
                    "score_grade": grade,
                    "rows": len(grade_part),
                    "bad_count": int(grade_part["TARGET"].sum()),
                    "bad_rate": float(grade_part["TARGET"].mean()),
                    "bad_capture": float(grade_part["TARGET"].sum() / total_bad),
                    "avg_pd": float(grade_part["pd_lgbm"].mean()),
                    "min_score": float(grade_part["a_card_score"].min()),
                    "max_score": float(grade_part["a_card_score"].max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "score_grade"])


def strategy_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labeled = scored[scored["TARGET"].notna()]
    for split_name, part in labeled.groupby("screening_split"):
        total_bad = max(part["TARGET"].sum(), 1)
        total_good = max(len(part) - part["TARGET"].sum(), 1)
        for action, action_part in part.groupby("strategy_action"):
            bad_count = int(action_part["TARGET"].sum())
            good_count = int(len(action_part) - bad_count)
            rows.append(
                {
                    "split": split_name,
                    "strategy_action": action,
                    "rows": len(action_part),
                    "population_pct": len(action_part) / len(part),
                    "bad_count": bad_count,
                    "good_count": good_count,
                    "bad_rate": float(action_part["TARGET"].mean()),
                    "bad_capture": bad_count / total_bad,
                    "good_capture": good_count / total_good,
                    "avg_pd": float(action_part["pd_lgbm"].mean()),
                    "avg_score": float(action_part["a_card_score"].mean()),
                    "min_score": float(action_part["a_card_score"].min()),
                    "max_score": float(action_part["a_card_score"].max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "strategy_action"])


def strategy_scenarios(scored: pd.DataFrame) -> pd.DataFrame:
    validation = scored[scored["screening_split"] == "validation"]
    labeled = scored[scored["TARGET"].notna()]
    rows = []
    for decline_pct in [0.03, 0.05, 0.10, 0.15, 0.20]:
        for manual_pct in [0.05, 0.10, 0.15, 0.20]:
            if decline_pct + manual_pct >= 0.70:
                continue
            decline_cutoff = float(np.quantile(validation["pd_lgbm"], 1 - decline_pct))
            manual_cutoff = float(np.quantile(validation["pd_lgbm"], 1 - decline_pct - manual_pct))
            for split_name, part in labeled.groupby("screening_split"):
                action = np.select(
                    [part["pd_lgbm"] >= decline_cutoff, part["pd_lgbm"] >= manual_cutoff],
                    ["decline", "manual_review"],
                    default="approve",
                )
                work = part.assign(action=action)
                total_bad = max(work["TARGET"].sum(), 1)
                decline_part = work[work["action"] == "decline"]
                manual_part = work[work["action"] == "manual_review"]
                approve_part = work[work["action"] == "approve"]
                rows.append(
                    {
                        "split": split_name,
                        "decline_pct_target": decline_pct,
                        "manual_review_pct_target": manual_pct,
                        "decline_rows": len(decline_part),
                        "manual_review_rows": len(manual_part),
                        "approve_rows": len(approve_part),
                        "decline_bad_rate": float(decline_part["TARGET"].mean()) if len(decline_part) else np.nan,
                        "manual_review_bad_rate": float(manual_part["TARGET"].mean()) if len(manual_part) else np.nan,
                        "approve_bad_rate": float(approve_part["TARGET"].mean()) if len(approve_part) else np.nan,
                        "decline_bad_capture": float(decline_part["TARGET"].sum() / total_bad),
                        "manual_review_bad_capture": float(manual_part["TARGET"].sum() / total_bad),
                        "decline_plus_review_bad_capture": float(
                            (decline_part["TARGET"].sum() + manual_part["TARGET"].sum()) / total_bad
                        ),
                    }
                )
    return pd.DataFrame(rows)


def external_score_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, part in scored.groupby("screening_split"):
        rows.append(
            {
                "split": split_name,
                "rows": len(part),
                "target_available": bool(part["TARGET"].notna().any()),
                "avg_pd": float(part["pd_lgbm"].mean()),
                "pd_p05": float(part["pd_lgbm"].quantile(0.05)),
                "pd_p50": float(part["pd_lgbm"].quantile(0.50)),
                "pd_p95": float(part["pd_lgbm"].quantile(0.95)),
                "avg_score": float(part["a_card_score"].mean()),
                "score_p05": float(part["a_card_score"].quantile(0.05)),
                "score_p50": float(part["a_card_score"].quantile(0.50)),
                "score_p95": float(part["a_card_score"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_parquet(PROCESSED_DIR / "candidate_feature_matrix.parquet")
    shortlist = pd.read_csv(OUTPUT_TABLE_DIR / "feature_screening_shortlist.csv")
    feature_names = shortlist["candidate_feature"].tolist()
    model_card = json.loads((MODEL_DIR / "lgbm_model_card.json").read_text(encoding="utf-8"))
    calibration_shift = float(model_card["calibration_intercept_shift"])
    best_iteration = int(model_card["best_iteration"])
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))

    raw_score = model.predict(matrix[feature_names].astype("float32"), num_iteration=best_iteration, raw_score=True)
    pd_lgbm = sigmoid(raw_score + calibration_shift)
    scored = matrix[["SK_ID_CURR", "dataset", "screening_split", "TARGET"]].copy()
    scored["pd_lgbm"] = pd_lgbm
    scored["a_card_score"] = pd_to_score(pd_lgbm)
    scored = add_score_grades(scored)
    manual_cutoff, decline_cutoff = strategy_cutoffs(scored)
    scored = assign_strategy(scored, manual_cutoff, decline_cutoff)

    labeled = scored[scored["TARGET"].notna()]
    metrics = pd.DataFrame(
        [
            labeled_metrics(split_name, part)
            for split_name, part in labeled.groupby("screening_split")
        ]
    )
    dev_scores = scored[scored["screening_split"] == "development"]["a_card_score"].to_numpy()
    metrics["score_psi_vs_development"] = [
        psi_score(dev_scores, scored[scored["screening_split"] == split]["a_card_score"].to_numpy())
        for split in metrics["split"]
    ]

    band_metrics = score_band_metrics(scored)
    action_metrics = strategy_metrics(scored)
    scenarios = strategy_scenarios(scored)
    summary = external_score_summary(scored)

    official_test = scored[scored["screening_split"] == "external_unlabeled"].copy()
    external_predictions = official_test[
        ["SK_ID_CURR", "pd_lgbm", "a_card_score", "score_grade", "strategy_action"]
    ].rename(columns={"pd_lgbm": "pred_default_probability"})
    kaggle_submission = official_test[["SK_ID_CURR", "pd_lgbm"]].rename(columns={"pd_lgbm": "TARGET"})

    scale = pd.DataFrame(
        [
            {
                "pdo": PDO,
                "base_score": BASE_SCORE,
                "base_odds_good_bad": BASE_ODDS,
                "factor": PDO / np.log(2),
                "offset": BASE_SCORE - (PDO / np.log(2)) * np.log(BASE_ODDS),
                "score_direction": "higher_score_lower_default_risk",
                "decline_pct_on_validation": DECLINE_PCT,
                "manual_review_pct_on_validation": MANUAL_REVIEW_PCT,
                "manual_pd_cutoff": manual_cutoff,
                "decline_pd_cutoff": decline_cutoff,
                "manual_score_cutoff": pd_to_score(np.array([manual_cutoff]))[0],
                "decline_score_cutoff": pd_to_score(np.array([decline_cutoff]))[0],
            }
        ]
    )

    scored.to_csv(OUTPUT_TABLE_DIR / "a_card_scored_population.csv", index=False)
    metrics.to_csv(OUTPUT_TABLE_DIR / "a_card_internal_test_metrics.csv", index=False)
    band_metrics.to_csv(OUTPUT_TABLE_DIR / "a_card_score_band_metrics.csv", index=False)
    action_metrics.to_csv(OUTPUT_TABLE_DIR / "a_card_strategy_action_metrics.csv", index=False)
    scenarios.to_csv(OUTPUT_TABLE_DIR / "a_card_strategy_scenarios.csv", index=False)
    summary.to_csv(OUTPUT_TABLE_DIR / "a_card_score_distribution_summary.csv", index=False)
    external_predictions.to_csv(OUTPUT_TABLE_DIR / "a_card_external_test_predictions.csv", index=False)
    kaggle_submission.to_csv(OUTPUT_TABLE_DIR / "a_card_kaggle_submission.csv", index=False)
    scale.to_csv(OUTPUT_TABLE_DIR / "a_card_score_scale.csv", index=False)

    final_holdout_row = metrics[metrics["split"] == "final_holdout"].iloc[0]
    validation_action = action_metrics[action_metrics["split"] == "validation"]
    holdout_action = action_metrics[action_metrics["split"] == "final_holdout"]
    external_summary = summary[summary["split"] == "external_unlabeled"].iloc[0]

    md = [
        "# A 卡评分与准入策略初版",
        "",
        "本文件由 `src/07_evaluate_strategy.py` 自动生成。",
        "",
        "## Test 口径说明",
        "",
        "- `final_holdout`：从 `application_train` 内部留出的有标签 test，可计算 AUC、KS、Lift、坏样本捕获等模型指标。",
        "- `external_unlabeled`：Kaggle 官方 `application_test`，没有 `TARGET`，只能输出预测 PD、A 卡分数、风险等级和策略动作，不能计算真实 AUC/KS。",
        "- `final_holdout` 和官方 test 都没有参与特征筛选、模型训练和 early stopping。",
        "",
        "## A 卡刻度",
        "",
        f"- Base score：{BASE_SCORE} 分，对应 good:bad odds = {BASE_ODDS}:1。",
        f"- PDO：{PDO}，坏账 odds 翻倍时分数下降 {PDO} 分。",
        "- 分数方向：分数越高，预测违约风险越低。",
        f"- 当前策略：validation 风险最高 {DECLINE_PCT:.0%} 拒绝，之后 {MANUAL_REVIEW_PCT:.0%} 人工复核，其余自动通过。",
        "",
        "## 内部有标签 Test 指标",
        "",
        "| split | rows | bad_rate | AUC | KS | top_5pct_bad_capture | top_10pct_bad_capture | score_PSI_vs_dev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        md.append(
            f"| {row['split']} | {int(row['rows'])} | {row['bad_rate']:.2%} | {row['auc']:.4f} | "
            f"{row['ks']:.4f} | {row['top_5pct_bad_capture']:.2%} | "
            f"{row['top_10pct_bad_capture']:.2%} | {row['score_psi_vs_development']:.4f} |"
        )

    md.extend(
        [
            "",
            "## 策略动作表现",
            "",
            "| split | action | population_pct | bad_rate | bad_capture | avg_score |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in pd.concat([validation_action, holdout_action]).iterrows():
        md.append(
            f"| {row['split']} | {row['strategy_action']} | {row['population_pct']:.2%} | "
            f"{row['bad_rate']:.2%} | {row['bad_capture']:.2%} | {row['avg_score']:.1f} |"
        )

    md.extend(
        [
            "",
            "## 官方 Test 预测概览",
            "",
            f"- 官方 test 行数：{int(external_summary['rows']):,}",
            f"- 平均预测 PD：{external_summary['avg_pd']:.2%}",
            f"- PD P50/P95：{external_summary['pd_p50']:.2%} / {external_summary['pd_p95']:.2%}",
            f"- Score P50/P05：{external_summary['score_p50']:.1f} / {external_summary['score_p05']:.1f}",
            "",
            "## 输出文件",
            "",
            "- `outputs/tables/a_card_internal_test_metrics.csv`：内部有标签 test 指标。",
            "- `outputs/tables/a_card_score_band_metrics.csv`：分数等级表现。",
            "- `outputs/tables/a_card_strategy_action_metrics.csv`：approve / manual_review / decline 表现。",
            "- `outputs/tables/a_card_strategy_scenarios.csv`：不同拒绝率/复核率情景对比。",
            "- `outputs/tables/a_card_external_test_predictions.csv`：官方 test 预测 PD、分数、等级和策略动作。",
            "- `outputs/tables/a_card_kaggle_submission.csv`：Kaggle submission 格式。",
            "- `outputs/tables/a_card_scored_population.csv`：全量样本评分明细。",
            "",
            "## 当前业务解读",
            "",
            f"- final_holdout AUC 为 {final_holdout_row['auc']:.4f}，KS 为 {final_holdout_row['ks']:.4f}。",
            f"- final_holdout Top 10% 高风险申请捕获 {final_holdout_row['top_10pct_bad_capture']:.2%} 坏样本。",
            "- 当前策略阈值只是 baseline 策略，后续可以根据审批通过率、人工复核产能和误拒成本调整。",
            "",
        ]
    )
    (DOCS_DIR / "a_card_strategy_initial.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_internal_test_metrics.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_score_band_metrics.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_strategy_action_metrics.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_strategy_scenarios.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_external_test_predictions.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_kaggle_submission.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_score_scale.csv'}")
    print(f"Wrote {DOCS_DIR / 'a_card_strategy_initial.md'}")


if __name__ == "__main__":
    main()
