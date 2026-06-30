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
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 2026
NUM_BOOST_ROUND = 2500
EARLY_STOPPING_ROUNDS = 100
LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 2.0,
    "min_gain_to_split": 0.0,
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
    "num_threads": 4,
}


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


def ks_score(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y_sorted = y[order]
    bad_total = y_sorted.sum()
    good_total = len(y_sorted) - bad_total
    if bad_total == 0 or good_total == 0:
        return np.nan
    bad_cum = np.cumsum(y_sorted) / bad_total
    good_cum = np.cumsum(1 - y_sorted) / good_total
    return float(np.max(np.abs(bad_cum - good_cum)))


def logloss(y: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(pred) + (1 - y) * np.log(1 - pred)).mean())


def brier_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((pred - y) ** 2))


def calibrate_intercept_shift(raw_score: np.ndarray, target_rate: float) -> float:
    lower, upper = -30.0, 30.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        avg_pred = sigmoid(raw_score + midpoint).mean()
        if avg_pred < target_rate:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def psi_score(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    reference = reference[np.isfinite(reference)]
    target = target[np.isfinite(target)]
    if len(reference) == 0 or len(target) == 0:
        return np.nan
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


def evaluate_split(name: str, frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float | int | str]:
    y = frame["TARGET"].to_numpy(dtype=np.float64)
    row: dict[str, float | int | str] = {
        "split": name,
        "rows": len(frame),
        "bad_count": int(y.sum()),
        "bad_rate": float(y.mean()),
        "auc": auc_score(y, pred),
        "ks": ks_score(y, pred),
        "logloss": logloss(y, pred),
        "brier": brier_score(y, pred),
        "avg_pred": float(pred.mean()),
    }
    for pct in [0.01, 0.05, 0.10, 0.20]:
        cutoff = max(int(np.ceil(len(frame) * pct)), 1)
        top_idx = np.argsort(-pred)[:cutoff]
        top_bad = y[top_idx].sum()
        row[f"top_{int(pct * 100)}pct_bad_rate"] = float(top_bad / cutoff)
        row[f"top_{int(pct * 100)}pct_bad_capture"] = float(top_bad / max(y.sum(), 1))
        row[f"top_{int(pct * 100)}pct_lift"] = float((top_bad / cutoff) / max(y.mean(), 1e-9))
    return row


def make_lift_table(split_name: str, frame: pd.DataFrame, pred: np.ndarray, bins: int = 10) -> pd.DataFrame:
    work = pd.DataFrame({"TARGET": frame["TARGET"].to_numpy(dtype=int), "pred_risk": pred})
    work = work.sort_values("pred_risk", ascending=False).reset_index(drop=True)
    work["score_band"] = np.floor(np.arange(len(work)) * bins / len(work)).astype(int) + 1
    overall_bad_rate = work["TARGET"].mean()
    total_bad = max(work["TARGET"].sum(), 1)
    rows = []
    cumulative_bad = 0
    for band, part in work.groupby("score_band"):
        bad_count = int(part["TARGET"].sum())
        cumulative_bad += bad_count
        rows.append(
            {
                "split": split_name,
                "score_band": int(band),
                "risk_rank": f"top_{int((band - 1) * 100 / bins)}_{int(band * 100 / bins)}pct",
                "rows": len(part),
                "bad_count": bad_count,
                "bad_rate": float(part["TARGET"].mean()),
                "lift": float(part["TARGET"].mean() / max(overall_bad_rate, 1e-9)),
                "bad_capture": float(bad_count / total_bad),
                "cumulative_bad_capture": float(cumulative_bad / total_bad),
                "min_score": float(part["pred_risk"].min()),
                "max_score": float(part["pred_risk"].max()),
            }
        )
    return pd.DataFrame(rows)


def evaluation_log_to_frame(evals_result: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows = []
    max_len = max(len(values) for metrics in evals_result.values() for values in metrics.values())
    for idx in range(max_len):
        row = {"iteration": idx + 1}
        for dataset_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                row[f"{dataset_name}_{metric_name}"] = values[idx] if idx < len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_parquet(PROCESSED_DIR / "candidate_feature_matrix.parquet")
    shortlist = pd.read_csv(OUTPUT_TABLE_DIR / "feature_screening_shortlist.csv")
    evidence = pd.read_csv(OUTPUT_TABLE_DIR / "feature_selection_evidence_table.csv")
    feature_names = shortlist["candidate_feature"].tolist()

    development = matrix[matrix["screening_split"] == "development"].copy()
    validation = matrix[matrix["screening_split"] == "validation"].copy()
    final_holdout = matrix[matrix["screening_split"] == "final_holdout"].copy()

    x_dev = development[feature_names].astype("float32")
    x_val = validation[feature_names].astype("float32")
    x_holdout = final_holdout[feature_names].astype("float32")
    y_dev = development["TARGET"].astype(int)
    y_val = validation["TARGET"].astype(int)
    y_holdout = final_holdout["TARGET"].astype(int)

    train_set = lgb.Dataset(x_dev, label=y_dev, feature_name=feature_names, free_raw_data=False)
    valid_set = lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=feature_names, free_raw_data=False)
    evals_result: dict[str, dict[str, list[float]]] = {}
    model = lgb.train(
        params=LGBM_PARAMS,
        train_set=train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, valid_set],
        valid_names=["development", "validation"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=False, verbose=False),
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=100),
        ],
    )

    best_iteration = model.best_iteration or NUM_BOOST_ROUND
    dev_raw = model.predict(x_dev, num_iteration=best_iteration, raw_score=True)
    val_raw = model.predict(x_val, num_iteration=best_iteration, raw_score=True)
    holdout_raw = model.predict(x_holdout, num_iteration=best_iteration, raw_score=True)
    calibration_shift = calibrate_intercept_shift(dev_raw, y_dev.mean())
    dev_pred = sigmoid(dev_raw + calibration_shift)
    val_pred = sigmoid(val_raw + calibration_shift)
    holdout_pred = sigmoid(holdout_raw + calibration_shift)

    metrics = pd.DataFrame(
        [
            evaluate_split("development", development, dev_pred),
            evaluate_split("validation", validation, val_pred),
            evaluate_split("final_holdout", final_holdout, holdout_pred),
        ]
    )
    metrics["score_psi_vs_development"] = [
        0.0,
        psi_score(dev_pred, val_pred),
        psi_score(dev_pred, holdout_pred),
    ]

    lift = pd.concat(
        [
            make_lift_table("development", development, dev_pred),
            make_lift_table("validation", validation, val_pred),
            make_lift_table("final_holdout", final_holdout, holdout_pred),
        ],
        ignore_index=True,
    )

    predictions = pd.concat(
        [
            development[["SK_ID_CURR", "screening_split", "TARGET"]].assign(pred_risk=dev_pred),
            validation[["SK_ID_CURR", "screening_split", "TARGET"]].assign(pred_risk=val_pred),
            final_holdout[["SK_ID_CURR", "screening_split", "TARGET"]].assign(pred_risk=holdout_pred),
        ],
        ignore_index=True,
    )

    importance = pd.DataFrame(
        {
            "candidate_feature": feature_names,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }
    )
    importance = importance.merge(
        evidence[
            [
                "candidate_feature",
                "original_feature",
                "encoding",
                "feature_group",
                "auc_power",
                "ks",
                "iv",
                "development_missing_rate",
                "psi_development_validation",
            ]
        ],
        on="candidate_feature",
        how="left",
    )
    importance["importance_gain_pct"] = importance["importance_gain"] / max(importance["importance_gain"].sum(), 1e-9)
    importance = importance.sort_values("importance_gain", ascending=False)

    eval_history = evaluation_log_to_frame(evals_result)

    metrics_path = OUTPUT_TABLE_DIR / "lgbm_metrics.csv"
    lift_path = OUTPUT_TABLE_DIR / "lgbm_lift_table.csv"
    predictions_path = OUTPUT_TABLE_DIR / "lgbm_predictions.csv"
    importance_path = OUTPUT_TABLE_DIR / "lgbm_feature_importance.csv"
    history_path = OUTPUT_TABLE_DIR / "lgbm_training_history.csv"
    metrics.to_csv(metrics_path, index=False)
    lift.to_csv(lift_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    importance.to_csv(importance_path, index=False)
    eval_history.to_csv(history_path, index=False)

    model_path = MODEL_DIR / "lgbm_model.txt"
    model.save_model(str(model_path), num_iteration=best_iteration)
    model_card = {
        "model_type": "LightGBM GBDT binary classifier",
        "lightgbm_version": lgb.__version__,
        "training_split": "development",
        "validation_split": "validation",
        "final_holdout_policy": "final_holdout is scored only after training and early stopping; not used for feature/model selection",
        "feature_count": len(feature_names),
        "params": LGBM_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "calibration_method": "development intercept shift on raw LightGBM scores to match observed bad rate",
        "calibration_intercept_shift": calibration_shift,
    }
    (MODEL_DIR / "lgbm_model_card.json").write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_rows = []
    baseline_path = OUTPUT_TABLE_DIR / "baseline_logit_metrics.csv"
    if baseline_path.exists():
        baseline_metrics = pd.read_csv(baseline_path)
        baseline_metrics.insert(0, "model", "baseline_logit")
        comparison_rows.append(baseline_metrics)
    lgbm_metrics = metrics.copy()
    lgbm_metrics.insert(0, "model", "lightgbm")
    comparison_rows.append(lgbm_metrics)
    comparison = pd.concat(comparison_rows, ignore_index=True)
    comparison_path = OUTPUT_TABLE_DIR / "model_comparison_baseline_lgbm.csv"
    comparison.to_csv(comparison_path, index=False)

    validation_row = metrics[metrics["split"] == "validation"].iloc[0]
    holdout_row = metrics[metrics["split"] == "final_holdout"].iloc[0]
    top_importance = importance.head(15)
    md = [
        "# LightGBM 模型初版",
        "",
        "本文件由 `src/06_train_lgbm.py` 自动生成。",
        "",
        "## 建模口径",
        "",
        "- 使用 `feature_screening_shortlist.csv` 中的统计短名单变量。",
        "- 只在 `development` 上训练 LightGBM。",
        "- 使用 `validation` 做 early stopping。",
        "- `final_holdout` 不参与训练、筛选或调参，只做最终只读评估。",
        "- 训练后只用 development 做 raw score 截距校准，使平均预测 PD 对齐 development 坏样本率。",
        "",
        "## 核心指标",
        "",
        "| split | rows | bad_rate | AUC | KS | top_5pct_bad_capture | top_10pct_bad_capture | score_PSI_vs_dev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        md.append(
            f"| {row['split']} | {int(row['rows'])} | {row['bad_rate']:.2%} | "
            f"{row['auc']:.4f} | {row['ks']:.4f} | {row['top_5pct_bad_capture']:.2%} | "
            f"{row['top_10pct_bad_capture']:.2%} | {row['score_psi_vs_development']:.4f} |"
        )

    md.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `outputs/tables/lgbm_metrics.csv`：development / validation / final_holdout 指标。",
            "- `outputs/tables/lgbm_lift_table.csv`：十分位 Lift 和累计坏样本捕获。",
            "- `outputs/tables/lgbm_feature_importance.csv`：gain/split 特征重要性。",
            "- `outputs/tables/lgbm_predictions.csv`：三个有标签 split 的预测分数。",
            "- `outputs/tables/model_comparison_baseline_lgbm.csv`：Logit 与 LightGBM 同口径对比。",
            "- `outputs/models/lgbm_model.txt`：LightGBM 模型文件。",
            "",
            "## Top Gain 特征",
            "",
            "| feature | gain_pct | split | group | univariate_auc_power |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for _, row in top_importance.iterrows():
        md.append(
            f"| `{row['candidate_feature']}` | {row['importance_gain_pct']:.2%} | "
            f"{int(row['importance_split'])} | {row['feature_group']} | {row['auc_power']:.4f} |"
        )

    md.extend(
        [
            "",
            "## 当前解读",
            "",
            f"- validation AUC 为 {validation_row['auc']:.4f}，KS 为 {validation_row['ks']:.4f}。",
            f"- final_holdout AUC 为 {holdout_row['auc']:.4f}，KS 为 {holdout_row['ks']:.4f}。",
            f"- best_iteration 为 {best_iteration}。",
            "- 下一步应把 LightGBM 分数转成 approve / manual review / decline，并在 final_holdout 上做策略评估。",
            "",
        ]
    )
    (DOCS_DIR / "lgbm_model_initial.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {lift_path}")
    print(f"Wrote {importance_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {model_path}")
    print(f"Wrote {MODEL_DIR / 'lgbm_model_card.json'}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {DOCS_DIR / 'lgbm_model_initial.md'}")


if __name__ == "__main__":
    main()
