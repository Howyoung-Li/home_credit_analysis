from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

ID_COLUMNS = {"dataset", "screening_split", "SK_ID_CURR", "TARGET"}
MAX_ONE_HOT_LEVELS = 60
MIN_CATEGORY_RATE = 0.0005
MISSING_LABEL = "__MISSING__"
VALIDATION_RATE = 0.20
FINAL_HOLDOUT_RATE = 0.20
RANDOM_SEED = 2026
CORR_PAIR_EXPORT_THRESHOLD = 0.80
CORR_PRUNE_THRESHOLD = 0.95
CORR_PRUNE_MAX_FEATURES = 650
AUC_POWER_THRESHOLD = 0.515
KS_THRESHOLD = 0.020
IV_THRESHOLD = 0.005


def sanitize_token(value: object, max_len: int = 48) -> str:
    token = str(value)
    token = re.sub(r"[^0-9A-Za-z]+", "_", token).strip("_")
    token = token or "blank"
    return token[:max_len]


def auc_score(y: pd.Series, x: pd.Series) -> float:
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    if data["y"].nunique() < 2 or data["x"].nunique() < 2:
        return np.nan
    ranks = data["x"].rank(method="average")
    y_values = data["y"].astype(int)
    n_pos = int(y_values.sum())
    n_neg = int(len(y_values) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    rank_sum_pos = float(ranks[y_values == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ks_score(y: pd.Series, x: pd.Series) -> float:
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    if data["y"].nunique() < 2 or data["x"].nunique() < 2:
        return np.nan
    data = data.sort_values("x")
    bad_total = data["y"].sum()
    good_total = len(data) - bad_total
    if bad_total == 0 or good_total == 0:
        return np.nan
    bad_cum = data["y"].cumsum() / bad_total
    good_cum = (1 - data["y"]).cumsum() / good_total
    return float((bad_cum - good_cum).abs().max())


def make_numeric_bins(train_values: pd.Series, bins: int = 10) -> np.ndarray | None:
    values = pd.to_numeric(train_values, errors="coerce").dropna()
    if values.nunique() < 2:
        return None
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(values.quantile(quantiles).to_numpy()).astype(float)
    if len(edges) < 3:
        min_value = values.min()
        max_value = values.max()
        if min_value == max_value:
            return None
        edges = np.array([min_value, max_value], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def information_value(y: pd.Series, x: pd.Series, bins: int = 10) -> float:
    data = pd.DataFrame({"y": y, "x": pd.to_numeric(x, errors="coerce")})
    edges = make_numeric_bins(data["x"], bins=bins)
    if edges is None or data["y"].nunique() < 2:
        return np.nan
    data["bucket"] = pd.cut(data["x"], bins=edges, include_lowest=True).astype("object")
    data.loc[data["x"].isna(), "bucket"] = "missing"
    grouped = data.groupby("bucket", observed=False)["y"].agg(["sum", "count"])
    grouped["bad"] = grouped["sum"]
    grouped["good"] = grouped["count"] - grouped["sum"]
    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    if total_bad == 0 or total_good == 0:
        return np.nan
    eps = 1e-6
    bad_dist = grouped["bad"] / total_bad
    good_dist = grouped["good"] / total_good
    return float(((bad_dist - good_dist) * np.log((bad_dist + eps) / (good_dist + eps))).sum())


def psi_score(train_values: pd.Series, test_values: pd.Series, bins: int = 10) -> float:
    train_numeric = pd.to_numeric(train_values, errors="coerce")
    test_numeric = pd.to_numeric(test_values, errors="coerce")
    edges = make_numeric_bins(train_numeric, bins=bins)
    if edges is None:
        return np.nan
    train_bucket = pd.cut(train_numeric, bins=edges, include_lowest=True).astype("object")
    test_bucket = pd.cut(test_numeric, bins=edges, include_lowest=True).astype("object")
    train_bucket.loc[train_numeric.isna()] = "missing"
    test_bucket.loc[test_numeric.isna()] = "missing"

    labels = sorted(set(train_bucket.dropna().astype(str)) | set(test_bucket.dropna().astype(str)))
    train_dist = train_bucket.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0)
    test_dist = test_bucket.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0)
    eps = 1e-6
    return float(((test_dist - train_dist) * np.log((test_dist + eps) / (train_dist + eps))).sum())


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in ID_COLUMNS and not pd.api.types.is_numeric_dtype(df[col])
    ]


def assign_screening_split(feature_store: pd.DataFrame) -> pd.Series:
    split = pd.Series("external_unlabeled", index=feature_store.index, dtype="string")
    train_idx = feature_store.index[feature_store["dataset"] == "train"]
    rng = np.random.default_rng(RANDOM_SEED)

    for target_value in [0, 1]:
        class_idx = feature_store.index[(feature_store["dataset"] == "train") & (feature_store["TARGET"] == target_value)]
        class_idx = np.array(class_idx)
        rng.shuffle(class_idx)
        final_holdout_size = int(round(len(class_idx) * FINAL_HOLDOUT_RATE))
        validation_size = int(round(len(class_idx) * VALIDATION_RATE))
        final_holdout_idx = class_idx[:final_holdout_size]
        validation_idx = class_idx[final_holdout_size : final_holdout_size + validation_size]
        development_idx = class_idx[final_holdout_size + validation_size :]
        split.loc[development_idx] = "development"
        split.loc[validation_idx] = "validation"
        split.loc[final_holdout_idx] = "final_holdout"

    if (split.loc[train_idx] == "external_unlabeled").any():
        raise ValueError("Some training rows were not assigned to development/validation/final_holdout split.")
    return split


def build_candidate_matrix(feature_store: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_ids = feature_store[["dataset", "SK_ID_CURR", "TARGET"]].copy()
    base_ids["screening_split"] = assign_screening_split(feature_store)
    candidate_parts = [base_ids]
    candidate_rows: list[dict[str, object]] = []

    categorical_cols = detect_categorical_columns(feature_store)
    numeric_cols = [
        col
        for col in feature_store.columns
        if col not in ID_COLUMNS and col not in categorical_cols and pd.api.types.is_numeric_dtype(feature_store[col])
    ]

    numeric_part = feature_store[numeric_cols].replace([np.inf, -np.inf], np.nan).astype("float32")
    candidate_parts.append(numeric_part)
    for col in numeric_cols:
        source_row = manifest[manifest["feature"] == col]
        candidate_rows.append(
            {
                "candidate_feature": col,
                "original_feature": col,
                "encoding": "numeric_original",
                "feature_group": source_row["feature_group"].iloc[0] if not source_row.empty else "unknown",
            }
        )

    development_mask = base_ids["screening_split"] == "development"
    for col in categorical_cols:
        train_values = feature_store.loc[development_mask, col].astype("string").fillna(MISSING_LABEL)
        all_values = feature_store[col].astype("string").fillna(MISSING_LABEL)
        freq_map = train_values.value_counts(normalize=True)

        freq_feature = f"catfreq__{col}"
        candidate_parts.append(pd.DataFrame({freq_feature: all_values.map(freq_map).fillna(0).astype("float32")}))
        source_row = manifest[manifest["feature"] == col]
        feature_group = source_row["feature_group"].iloc[0] if not source_row.empty else "current_application"
        candidate_rows.append(
            {
                "candidate_feature": freq_feature,
                "original_feature": col,
                "encoding": "category_frequency_train",
                "feature_group": feature_group,
            }
        )

        eligible_levels = freq_map[freq_map >= MIN_CATEGORY_RATE].head(MAX_ONE_HOT_LEVELS).index.tolist()
        used_names: set[str] = set()
        one_hot_data: dict[str, pd.Series] = {}
        for idx, level in enumerate(eligible_levels, start=1):
            suffix = sanitize_token(level)
            feature_name = f"catoh__{col}__{idx:02d}_{suffix}"
            while feature_name in used_names:
                feature_name = f"{feature_name}_{idx}"
            used_names.add(feature_name)
            one_hot_data[feature_name] = (all_values == level).astype("int8")
            candidate_rows.append(
                {
                    "candidate_feature": feature_name,
                    "original_feature": col,
                    "encoding": "category_one_hot_train_top_level",
                    "feature_group": feature_group,
                }
            )
        if one_hot_data:
            candidate_parts.append(pd.DataFrame(one_hot_data))

    candidate_matrix = pd.concat(candidate_parts, axis=1)
    candidate_manifest = pd.DataFrame(candidate_rows)
    return candidate_matrix, candidate_manifest


def screen_candidates(candidate_matrix: pd.DataFrame, candidate_manifest: pd.DataFrame) -> pd.DataFrame:
    development = candidate_matrix[candidate_matrix["screening_split"] == "development"]
    validation = candidate_matrix[candidate_matrix["screening_split"] == "validation"]
    y = development["TARGET"].astype(int)

    rows: list[dict[str, object]] = []
    candidate_features = [col for col in candidate_matrix.columns if col not in ID_COLUMNS]
    for idx, col in enumerate(candidate_features, start=1):
        development_col = development[col]
        validation_col = validation[col]
        auc = auc_score(y, development_col)
        auc_power = np.nan if pd.isna(auc) else max(auc, 1 - auc)
        rows.append(
            {
                "candidate_feature": col,
                "development_missing_rate": float(development_col.isna().mean()),
                "validation_missing_rate": float(validation_col.isna().mean()),
                "development_nunique": int(development_col.nunique(dropna=True)),
                "validation_nunique": int(validation_col.nunique(dropna=True)),
                "psi_development_validation": psi_score(development_col, validation_col),
                "auc": auc,
                "auc_power": auc_power,
                "ks": ks_score(y, development_col),
                "iv": information_value(y, development_col),
            }
        )
        if idx % 100 == 0:
            print(f"Screened {idx}/{len(candidate_features)} features")

    report = pd.DataFrame(rows).merge(candidate_manifest, on="candidate_feature", how="left")
    report["quality_pass"] = (
        (report["development_missing_rate"] <= 0.95)
        & (report["development_nunique"] > 1)
        & ((report["psi_development_validation"].isna()) | (report["psi_development_validation"] <= 0.35))
    )
    report["univariate_pass"] = (
        report["quality_pass"]
        & (
            (report["auc_power"] >= AUC_POWER_THRESHOLD)
            | (report["ks"] >= KS_THRESHOLD)
            | (report["iv"] >= IV_THRESHOLD)
        )
    )
    report["screening_score"] = (
        report["auc_power"].fillna(0.5) - 0.5
        + report["ks"].fillna(0) * 0.5
        + np.minimum(report["iv"].fillna(0), 1.0) * 0.1
        - report["development_missing_rate"].fillna(1) * 0.02
        - report["psi_development_validation"].fillna(0).clip(lower=0, upper=1) * 0.02
    )
    return report.sort_values(["univariate_pass", "screening_score"], ascending=[False, False])


def correlation_prune(
    candidate_matrix: pd.DataFrame,
    report: pd.DataFrame,
    max_features: int = CORR_PRUNE_MAX_FEATURES,
    corr_threshold: float = CORR_PRUNE_THRESHOLD,
) -> pd.DataFrame:
    selected = report[report["univariate_pass"]].head(max_features).copy()
    if selected.empty:
        report["corr_keep"] = False
        report["corr_drop_reason"] = ""
        return report

    train = candidate_matrix[candidate_matrix["screening_split"] == "development"]
    selected_features = selected["candidate_feature"].tolist()
    matrix = train[selected_features].copy()
    matrix = matrix.fillna(matrix.median(numeric_only=True)).astype("float32")
    corr = matrix.corr(method="pearson").abs()

    keep: list[str] = []
    drop_reason: dict[str, str] = {}
    drop_with: dict[str, str] = {}
    drop_corr: dict[str, float] = {}
    score_map = selected.set_index("candidate_feature")["screening_score"].to_dict()
    for feature in selected_features:
        correlated_kept = [kept for kept in keep if corr.loc[feature, kept] >= corr_threshold]
        if not correlated_kept:
            keep.append(feature)
            continue
        best_kept = max(correlated_kept, key=lambda item: score_map.get(item, -math.inf))
        if score_map.get(feature, -math.inf) > score_map.get(best_kept, -math.inf):
            keep.remove(best_kept)
            drop_reason[best_kept] = f"corr>={corr_threshold} with stronger feature {feature}"
            drop_with[best_kept] = feature
            drop_corr[best_kept] = float(corr.loc[feature, best_kept])
            keep.append(feature)
        else:
            drop_reason[feature] = f"corr>={corr_threshold} with stronger feature {best_kept}"
            drop_with[feature] = best_kept
            drop_corr[feature] = float(corr.loc[feature, best_kept])

    report = report.copy()
    keep_set = set(keep)
    report["corr_keep"] = report["univariate_pass"] & report["candidate_feature"].isin(keep_set)
    report["corr_prune_threshold"] = corr_threshold
    report["corr_drop_with"] = report["candidate_feature"].map(drop_with).fillna("")
    report["corr_drop_abs_corr"] = report["candidate_feature"].map(drop_corr)
    report["corr_drop_reason"] = report["candidate_feature"].map(drop_reason).fillna("")
    report["final_stat_shortlist"] = report["univariate_pass"] & report["corr_keep"]
    return report


def add_selection_stage(report: pd.DataFrame) -> pd.DataFrame:
    report = report.copy()
    conditions = [
        report["final_stat_shortlist"],
        report["univariate_pass"] & ~report["corr_keep"],
        report["quality_pass"] & ~report["univariate_pass"],
        ~report["quality_pass"],
    ]
    choices = [
        "final_shortlist",
        "correlation_dropped",
        "failed_univariate_screen",
        "failed_quality_screen",
    ]
    report["selection_stage"] = np.select(conditions, choices, default="not_reviewed")
    return report


def write_selection_evidence_tables(candidate_matrix: pd.DataFrame, report: pd.DataFrame) -> None:
    evidence_columns = [
        "candidate_feature",
        "original_feature",
        "encoding",
        "feature_group",
        "selection_stage",
        "quality_pass",
        "univariate_pass",
        "corr_keep",
        "final_stat_shortlist",
        "development_missing_rate",
        "validation_missing_rate",
        "development_nunique",
        "validation_nunique",
        "psi_development_validation",
        "auc",
        "auc_power",
        "ks",
        "iv",
        "screening_score",
        "corr_prune_threshold",
        "corr_drop_with",
        "corr_drop_abs_corr",
        "corr_drop_reason",
    ]
    report[evidence_columns].to_csv(OUTPUT_TABLE_DIR / "feature_selection_evidence_table.csv", index=False)

    selected = report[report["univariate_pass"]].copy()
    if selected.empty:
        pd.DataFrame().to_csv(OUTPUT_TABLE_DIR / "feature_correlation_pairs_ge_0p80.csv", index=False)
        pd.DataFrame().to_csv(OUTPUT_TABLE_DIR / "feature_correlation_prune_decisions.csv", index=False)
        return

    development = candidate_matrix[candidate_matrix["screening_split"] == "development"]
    selected_features = selected["candidate_feature"].tolist()
    matrix = development[selected_features].copy()
    matrix = matrix.fillna(matrix.median(numeric_only=True)).astype("float32")
    corr = matrix.corr(method="pearson").abs()
    meta = report.set_index("candidate_feature")

    pair_rows: list[dict[str, object]] = []
    for i, feature_a in enumerate(selected_features):
        for feature_b in selected_features[i + 1 :]:
            abs_corr = float(corr.loc[feature_a, feature_b])
            if abs_corr < CORR_PAIR_EXPORT_THRESHOLD:
                continue
            score_a = float(meta.loc[feature_a, "screening_score"])
            score_b = float(meta.loc[feature_b, "screening_score"])
            kept_a = bool(meta.loc[feature_a, "final_stat_shortlist"])
            kept_b = bool(meta.loc[feature_b, "final_stat_shortlist"])
            pair_rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "abs_corr_development": abs_corr,
                    "above_prune_threshold": abs_corr >= CORR_PRUNE_THRESHOLD,
                    "feature_a_group": meta.loc[feature_a, "feature_group"],
                    "feature_b_group": meta.loc[feature_b, "feature_group"],
                    "feature_a_score": score_a,
                    "feature_b_score": score_b,
                    "feature_a_auc_power": meta.loc[feature_a, "auc_power"],
                    "feature_b_auc_power": meta.loc[feature_b, "auc_power"],
                    "feature_a_iv": meta.loc[feature_a, "iv"],
                    "feature_b_iv": meta.loc[feature_b, "iv"],
                    "feature_a_missing_rate": meta.loc[feature_a, "development_missing_rate"],
                    "feature_b_missing_rate": meta.loc[feature_b, "development_missing_rate"],
                    "feature_a_final_shortlist": kept_a,
                    "feature_b_final_shortlist": kept_b,
                    "kept_feature": feature_a if kept_a and not kept_b else feature_b if kept_b and not kept_a else "",
                    "dropped_feature": feature_b if kept_a and not kept_b else feature_a if kept_b and not kept_a else "",
                }
            )

    pair_table = pd.DataFrame(pair_rows).sort_values("abs_corr_development", ascending=False)
    pair_table.to_csv(OUTPUT_TABLE_DIR / "feature_correlation_pairs_ge_0p80.csv", index=False)

    prune_columns = [
        "candidate_feature",
        "corr_drop_with",
        "corr_drop_abs_corr",
        "screening_score",
        "auc_power",
        "ks",
        "iv",
        "development_missing_rate",
        "psi_development_validation",
        "feature_group",
        "encoding",
        "corr_drop_reason",
    ]
    prune_decisions = report[report["selection_stage"] == "correlation_dropped"][prune_columns].sort_values(
        "corr_drop_abs_corr", ascending=False
    )
    prune_decisions.to_csv(OUTPUT_TABLE_DIR / "feature_correlation_prune_decisions.csv", index=False)


def write_docs(report: pd.DataFrame, candidate_matrix: pd.DataFrame) -> None:
    split_summary = (
        candidate_matrix.groupby(["dataset", "screening_split"], dropna=False)
        .agg(rows=("SK_ID_CURR", "count"), bad_rate=("TARGET", "mean"))
        .reset_index()
    )
    split_summary.to_csv(OUTPUT_TABLE_DIR / "feature_screening_split_summary.csv", index=False)

    group_summary = (
        report.groupby("feature_group", dropna=False)
        .agg(
            candidate_count=("candidate_feature", "count"),
            quality_pass_count=("quality_pass", "sum"),
            univariate_pass_count=("univariate_pass", "sum"),
            final_shortlist_count=("final_stat_shortlist", "sum"),
            median_auc_power=("auc_power", "median"),
            median_iv=("iv", "median"),
            median_psi_dev_val=("psi_development_validation", "median"),
        )
        .reset_index()
        .sort_values("final_shortlist_count", ascending=False)
    )
    group_summary.to_csv(OUTPUT_TABLE_DIR / "feature_screening_group_summary.csv", index=False)

    shortlist = report[report["final_stat_shortlist"]].sort_values("screening_score", ascending=False)
    top_rows = shortlist.head(30)

    md = [
        "# 候选特征筛选初版",
        "",
        "本文件由 `src/04_screen_candidate_features.py` 自动生成。",
        "",
        "## 方法",
        "",
        "- 从申请级特征集市读取候选变量。",
        "- 数值变量保留原始数值形态。",
        "- 在 `application_train` 内部按标签分层切出 development / validation / final_holdout。",
        "- 类别变量只用 development 拟合频率编码和 top 类别 one-hot。",
        "- 先做 target-blind 质量筛选：缺失率、唯一值、development/validation PSI。",
        f"- 再只用 development 标签做 target-aware 单变量筛选：AUC power >= {AUC_POWER_THRESHOLD:.3f}、KS >= {KS_THRESHOLD:.3f}、IV >= {IV_THRESHOLD:.3f}。",
        "- 单变量门槛设置为宽松口径，目的是保留近期窗口和分组聚合变量，让 LightGBM 继续判断非线性与交互价值。",
        "- 最后对单变量候选做高相关去重，避免重复变量堆叠。",
        "- `final_holdout` 和官方 Kaggle test 不参与本阶段任何筛选。",
        "",
        "## 样本切分",
        "",
        "| dataset | screening_split | rows | bad_rate |",
        "|---|---|---:|---:|",
    ]
    for _, row in split_summary.iterrows():
        bad_rate = "" if pd.isna(row["bad_rate"]) else f"{row['bad_rate']:.2%}"
        md.append(f"| {row['dataset']} | {row['screening_split']} | {int(row['rows'])} | {bad_rate} |")

    md.extend(
        [
            "",
            "## 产出概览",
            "",
            f"- 候选矩阵行数：{candidate_matrix.shape[0]:,}",
            f"- 候选特征数：{candidate_matrix.shape[1] - len(ID_COLUMNS):,}",
            f"- 质量筛选通过：{int(report['quality_pass'].sum()):,}",
        f"- 单变量筛选通过：{int(report['univariate_pass'].sum()):,}",
        f"- 相关性去重后短名单：{int(report['final_stat_shortlist'].sum()):,}",
        "",
        "## 证据表清单",
        "",
        "- `outputs/tables/feature_screening_report.csv`：全量候选变量筛选报告。",
        "- `outputs/tables/feature_selection_evidence_table.csv`：面试证据总表，含筛选阶段、缺失率、PSI、AUC、KS、IV、相关性去重原因。",
        "- `outputs/tables/feature_correlation_pairs_ge_0p80.csv`：development 样本中绝对相关系数 >= 0.80 的变量对。",
        "- `outputs/tables/feature_correlation_prune_decisions.csv`：因相关性 >= 0.95 被去重剔除的变量及其对应保留变量。",
        "- `outputs/tables/feature_screening_shortlist.csv`：最终统计短名单。",
        "",
        "## 分组结果",
        "",
            "| feature_group | candidate_count | quality_pass | univariate_pass | final_shortlist | median_auc_power | median_iv | median_psi_dev_val |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in group_summary.iterrows():
        md.append(
            f"| {row['feature_group']} | {int(row['candidate_count'])} | {int(row['quality_pass_count'])} | "
            f"{int(row['univariate_pass_count'])} | {int(row['final_shortlist_count'])} | "
            f"{row['median_auc_power']:.4f} | {row['median_iv']:.4f} | {row['median_psi_dev_val']:.4f} |"
        )

    md.extend(
        [
            "",
            "## Top 30 统计短名单",
            "",
            "| feature | original_feature | encoding | group | auc_power | ks | iv | psi_dev_val | missing |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_rows.iterrows():
        md.append(
            f"| `{row['candidate_feature']}` | `{row['original_feature']}` | {row['encoding']} | "
            f"{row['feature_group']} | {row['auc_power']:.4f} | {row['ks']:.4f} | "
            f"{row['iv']:.4f} | {row['psi_development_validation']:.4f} | {row['development_missing_rate']:.2%} |"
        )

    md.extend(
        [
            "",
            "## 如何避免先射箭再画靶",
            "",
            "- 业务解释不参与初始候选是否生成；候选由数据源、聚合口径和编码规则系统产生。",
            "- 第一轮剔除规则提前固定，先看 development/validation 质量和稳定性，再看 development 单变量区分度。",
            "- `final_holdout` 不参与筛选，留给后续模型最终评估、PSI 诊断和漂移变量 ablation。",
            "- 官方 Kaggle test 是无标签外部样本，不参与任何特征选择或模型选择。",
            "- 只有通过统计筛选的变量才进入后续业务 reason code 解释。",
            "- 下一轮模型筛选会继续用交叉验证、LightGBM 特征重要性和 permutation/null importance 验证。",
            "",
        ]
    )
    (DOCS_DIR / "feature_screening_initial.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    feature_store = pd.read_parquet(PROCESSED_DIR / "application_feature_store.parquet")
    manifest = pd.read_csv(PROCESSED_DIR / "feature_manifest_initial.csv")

    candidate_matrix, candidate_manifest = build_candidate_matrix(feature_store, manifest)
    candidate_matrix_path = PROCESSED_DIR / "candidate_feature_matrix.parquet"
    candidate_manifest_path = PROCESSED_DIR / "candidate_feature_manifest.csv"
    candidate_matrix.to_parquet(candidate_matrix_path, index=False)
    candidate_manifest.to_csv(candidate_manifest_path, index=False)

    report = screen_candidates(candidate_matrix, candidate_manifest)
    report = correlation_prune(candidate_matrix, report)
    report = add_selection_stage(report)

    report_path = OUTPUT_TABLE_DIR / "feature_screening_report.csv"
    shortlist_path = OUTPUT_TABLE_DIR / "feature_screening_shortlist.csv"
    report.to_csv(report_path, index=False)
    report[report["final_stat_shortlist"]].to_csv(shortlist_path, index=False)
    stale_watchlist_path = OUTPUT_TABLE_DIR / "feature_external_shift_watchlist.csv"
    if stale_watchlist_path.exists():
        stale_watchlist_path.unlink()
    write_selection_evidence_tables(candidate_matrix, report)
    write_docs(report, candidate_matrix)

    print(f"Wrote {candidate_matrix_path}")
    print(f"Wrote {candidate_manifest_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {shortlist_path}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'feature_selection_evidence_table.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'feature_correlation_pairs_ge_0p80.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'feature_correlation_prune_decisions.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'feature_screening_group_summary.csv'}")
    print(f"Wrote {DOCS_DIR / 'feature_screening_initial.md'}")


if __name__ == "__main__":
    main()
