from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 2026
L2_PENALTY = 0.01
LEARNING_RATE = 0.05
MAX_ITER = 700
PATIENCE = 60
MIN_DELTA = 1e-5
MISSING_INDICATOR_MIN_RATE = 0.001


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


def weighted_logloss(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return float(np.average(loss, weights=weights))


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


def make_design_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    preprocessing: dict[str, np.ndarray] | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    raw = frame[feature_names].to_numpy(dtype=np.float64, copy=True)
    raw[~np.isfinite(raw)] = np.nan

    if fit:
        lower = np.nanquantile(raw, 0.01, axis=0)
        upper = np.nanquantile(raw, 0.99, axis=0)
        impute = np.nanmedian(raw, axis=0)
        impute = np.where(np.isfinite(impute), impute, 0.0)
        missing_rates = np.isnan(raw).mean(axis=0)
        missing_indicator_mask = missing_rates >= MISSING_INDICATOR_MIN_RATE
    else:
        if preprocessing is None:
            raise ValueError("preprocessing must be supplied when fit=False")
        lower = preprocessing["lower"]
        upper = preprocessing["upper"]
        impute = preprocessing["impute"]
        missing_indicator_mask = preprocessing["missing_indicator_mask"].astype(bool)

    missing_raw = np.isnan(raw)
    clipped = np.clip(raw, lower, upper)
    clipped = np.where(np.isnan(clipped), impute, clipped)
    indicators = missing_raw[:, missing_indicator_mask].astype(np.float64)
    design = np.concatenate([clipped, indicators], axis=1)

    if fit:
        mean = design.mean(axis=0)
        std = design.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        preprocessing = {
            "lower": lower,
            "upper": upper,
            "impute": impute,
            "missing_indicator_mask": missing_indicator_mask.astype(bool),
            "mean": mean,
            "std": std,
        }
    else:
        mean = preprocessing["mean"]
        std = preprocessing["std"]

    design = (design - mean) / std
    indicator_names = [f"missing__{name}" for name, flag in zip(feature_names, missing_indicator_mask) if flag]
    model_feature_names = feature_names + indicator_names
    return design.astype(np.float32), preprocessing, model_feature_names


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    pos_rate = y.mean()
    neg_rate = 1 - pos_rate
    weights = np.where(y == 1, 0.5 / max(pos_rate, 1e-6), 0.5 / max(neg_rate, 1e-6))
    return weights / weights.mean()


def fit_logit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    coef = rng.normal(0, 0.001, size=x_train.shape[1]).astype(np.float64)
    intercept = float(np.log(y_train.mean() / (1 - y_train.mean())))
    weights = class_balanced_weights(y_train).astype(np.float64)

    best_coef = coef.copy()
    best_intercept = intercept
    best_valid_auc = -np.inf
    stale_rounds = 0
    history_rows = []

    for iteration in range(1, MAX_ITER + 1):
        train_pred = sigmoid(x_train @ coef + intercept)
        error = (train_pred - y_train) * weights
        grad_coef = (x_train.T @ error) / len(y_train) + L2_PENALTY * coef
        grad_intercept = float(error.mean())

        coef -= LEARNING_RATE * grad_coef
        intercept -= LEARNING_RATE * grad_intercept

        if iteration == 1 or iteration % 10 == 0:
            valid_pred = sigmoid(x_valid @ coef + intercept)
            valid_auc = auc_score(y_valid, valid_pred)
            train_loss = weighted_logloss(y_train, train_pred, weights)
            valid_loss = logloss(y_valid, valid_pred)
            history_rows.append(
                {
                    "iteration": iteration,
                    "train_weighted_logloss": train_loss,
                    "validation_logloss": valid_loss,
                    "validation_auc": valid_auc,
                    "validation_ks": ks_score(y_valid, valid_pred),
                }
            )

            if valid_auc > best_valid_auc + MIN_DELTA:
                best_valid_auc = valid_auc
                best_coef = coef.copy()
                best_intercept = intercept
                stale_rounds = 0
            else:
                stale_rounds += 10
                if stale_rounds >= PATIENCE:
                    break

    return best_coef, best_intercept, pd.DataFrame(history_rows)


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

    x_dev, preprocessing, model_feature_names = make_design_matrix(development, feature_names, fit=True)
    x_val, _, _ = make_design_matrix(validation, feature_names, preprocessing=preprocessing, fit=False)
    x_holdout, _, _ = make_design_matrix(final_holdout, feature_names, preprocessing=preprocessing, fit=False)
    y_dev = development["TARGET"].to_numpy(dtype=np.float64)
    y_val = validation["TARGET"].to_numpy(dtype=np.float64)
    y_holdout = final_holdout["TARGET"].to_numpy(dtype=np.float64)

    coef, raw_intercept, history = fit_logit(x_dev, y_dev, x_val, y_val)
    raw_dev_score = x_dev @ coef + raw_intercept
    calibration_shift = calibrate_intercept_shift(raw_dev_score, y_dev.mean())
    intercept = raw_intercept + calibration_shift
    dev_pred = sigmoid(x_dev @ coef + intercept)
    val_pred = sigmoid(x_val @ coef + intercept)
    holdout_pred = sigmoid(x_holdout @ coef + intercept)

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

    coefficient_rows = []
    base_feature_count = len(feature_names)
    feature_meta = evidence.set_index("candidate_feature")
    for idx, name in enumerate(model_feature_names):
        base_name = name.replace("missing__", "", 1) if name.startswith("missing__") else name
        meta = feature_meta.loc[base_name] if base_name in feature_meta.index else None
        coefficient_rows.append(
            {
                "model_feature": name,
                "base_candidate_feature": base_name,
                "is_missing_indicator": name.startswith("missing__"),
                "standardized_coefficient": coef[idx],
                "abs_standardized_coefficient": abs(coef[idx]),
                "risk_direction": "higher_risk" if coef[idx] > 0 else "lower_risk",
                "feature_group": "" if meta is None else meta["feature_group"],
                "encoding": "" if meta is None else meta["encoding"],
                "auc_power_univariate": np.nan if meta is None else meta["auc_power"],
                "ks_univariate": np.nan if meta is None else meta["ks"],
                "iv_univariate": np.nan if meta is None else meta["iv"],
            }
        )
    coefficients = pd.DataFrame(coefficient_rows).sort_values("abs_standardized_coefficient", ascending=False)

    metrics_path = OUTPUT_TABLE_DIR / "baseline_logit_metrics.csv"
    lift_path = OUTPUT_TABLE_DIR / "baseline_logit_lift_table.csv"
    history_path = OUTPUT_TABLE_DIR / "baseline_logit_training_history.csv"
    coefficients_path = OUTPUT_TABLE_DIR / "baseline_logit_coefficients.csv"
    predictions_path = OUTPUT_TABLE_DIR / "baseline_logit_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    lift.to_csv(lift_path, index=False)
    history.to_csv(history_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    np.savez_compressed(
        MODEL_DIR / "baseline_logit_model.npz",
        feature_names=np.array(feature_names, dtype=object),
        model_feature_names=np.array(model_feature_names, dtype=object),
        coef=coef,
        intercept=np.array([intercept]),
        lower=preprocessing["lower"],
        upper=preprocessing["upper"],
        impute=preprocessing["impute"],
        missing_indicator_mask=preprocessing["missing_indicator_mask"],
        mean=preprocessing["mean"],
        std=preprocessing["std"],
    )
    model_card = {
        "model_type": "L2 regularized logistic regression implemented with numpy",
        "training_split": "development",
        "validation_split": "validation",
        "final_holdout_policy": "final_holdout is scored only after model fitting and is not used for feature/model selection",
        "base_feature_count": base_feature_count,
        "missing_indicator_count": len(model_feature_names) - base_feature_count,
        "l2_penalty": L2_PENALTY,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "best_iteration": int(history.iloc[history["validation_auc"].idxmax()]["iteration"]) if not history.empty else None,
        "raw_intercept": raw_intercept,
        "calibration_method": "development intercept shift to match observed bad rate",
        "calibration_intercept_shift": calibration_shift,
        "intercept": intercept,
    }
    (MODEL_DIR / "baseline_logit_model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation_row = metrics[metrics["split"] == "validation"].iloc[0]
    holdout_row = metrics[metrics["split"] == "final_holdout"].iloc[0]
    top_coeff = coefficients.head(12)
    md = [
        "# Baseline Logit 模型初版",
        "",
        "本文件由 `src/05_train_baseline.py` 自动生成。",
        "",
        "## 建模口径",
        "",
        "- 使用 `feature_screening_shortlist.csv` 中的统计短名单变量。",
        "- 只在 `development` 上拟合预处理参数和模型参数。",
        "- 使用 `validation` 观察泛化表现。",
        "- `final_holdout` 不参与训练、筛选或调参，只做最终只读评估。",
        "- 类别变量已在 `04_screen_candidate_features.py` 中转成频率编码或 one-hot。",
        "- 数值变量做 1%/99% 截尾、development 中位数填补、标准化；缺失率 >= 0.1% 的变量额外生成缺失指示器。",
        "- Logit 使用类别平衡权重训练以提升排序；训练后只用 development 做截距校准，使平均预测 PD 对齐 development 坏样本率。",
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
            "- `outputs/tables/baseline_logit_metrics.csv`：development / validation / final_holdout 指标。",
            "- `outputs/tables/baseline_logit_lift_table.csv`：十分位 Lift 和累计坏样本捕获。",
            "- `outputs/tables/baseline_logit_coefficients.csv`：标准化系数和风险方向。",
            "- `outputs/tables/baseline_logit_predictions.csv`：三个有标签 split 的预测分数。",
            "- `outputs/models/baseline_logit_model.npz`：模型参数和预处理参数。",
            "",
            "## Top 标准化系数",
            "",
            "| model_feature | direction | coef | group | univariate_auc_power |",
            "|---|---|---:|---|---:|",
        ]
    )
    for _, row in top_coeff.iterrows():
        md.append(
            f"| `{row['model_feature']}` | {row['risk_direction']} | "
            f"{row['standardized_coefficient']:.4f} | {row['feature_group']} | "
            f"{row['auc_power_univariate']:.4f} |"
        )
    md.extend(
        [
            "",
            "## 当前解读",
            "",
            f"- validation AUC 为 {validation_row['auc']:.4f}，KS 为 {validation_row['ks']:.4f}。",
            f"- final_holdout AUC 为 {holdout_row['auc']:.4f}，KS 为 {holdout_row['ks']:.4f}。",
            "- 这个 baseline 的价值不是最终性能，而是建立可复现的训练、验证、最终只读评估和风控指标口径。",
            "- 下一步可以训练 LightGBM，并在模型冻结后做 final_holdout PSI 和高漂移变量 ablation。",
            "",
        ]
    )
    (DOCS_DIR / "baseline_model_initial.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {lift_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {coefficients_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {MODEL_DIR / 'baseline_logit_model.npz'}")
    print(f"Wrote {MODEL_DIR / 'baseline_logit_model_card.json'}")
    print(f"Wrote {DOCS_DIR / 'baseline_model_initial.md'}")


if __name__ == "__main__":
    main()
