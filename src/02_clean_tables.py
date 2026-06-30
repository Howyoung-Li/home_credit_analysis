from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

SPECIAL_DAY_SENTINEL = 365243
UNKNOWN_CATEGORY_VALUES = {"XNA", "Unknown"}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def read_application_file(dataset: str, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / file_name, low_memory=False)
    df.insert(0, "dataset", dataset)
    if "TARGET" not in df.columns:
        df.insert(2, "TARGET", np.nan)
    return df


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(UNKNOWN_CATEGORY_VALUES), col] = pd.NA
    return df


def clean_day_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    day_cols = [col for col in df.columns if col.startswith("DAYS")]
    for col in day_cols:
        if col not in df.columns:
            continue
        mask = df[col] == SPECIAL_DAY_SENTINEL
        sentinel_count = int(mask.sum())
        if sentinel_count:
            flag_col = f"{col}_SENTINEL_365243"
            df[flag_col] = mask.astype("int8")
            df.loc[mask, col] = np.nan
        rows.append(
            {
                "column": col,
                "sentinel_value": SPECIAL_DAY_SENTINEL,
                "sentinel_count": sentinel_count,
                "sentinel_rate": sentinel_count / len(df),
                "created_flag": sentinel_count > 0,
            }
        )
    return df, rows


def add_application_business_features(df: pd.DataFrame) -> pd.DataFrame:
    ext_source_cols = [col for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if col in df.columns]
    doc_flag_cols = [col for col in df.columns if col.startswith("FLAG_DOCUMENT_")]
    contact_cols = [col for col in ["FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "FLAG_MOBIL"] if col in df.columns]
    bureau_request_cols = [col for col in df.columns if col.startswith("AMT_REQ_CREDIT_BUREAU_")]
    raw_feature_cols = [
        col
        for col in df.columns
        if col not in {"dataset", "SK_ID_CURR", "TARGET"} and not col.endswith("_SENTINEL_365243")
    ]

    df["applicant_age_years"] = -df["DAYS_BIRTH"] / 365.25
    df["employment_years"] = -df["DAYS_EMPLOYED"] / 365.25
    df["registration_age_years"] = -df["DAYS_REGISTRATION"] / 365.25
    df["id_publish_years"] = -df["DAYS_ID_PUBLISH"] / 365.25
    df["last_phone_change_years"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25

    df["credit_to_income_ratio"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["annuity_to_income_ratio"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["credit_to_annuity_ratio"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["goods_to_credit_ratio"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["income_per_family_member"] = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["children_to_family_ratio"] = safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])
    df["employment_to_age_ratio"] = safe_divide(df["employment_years"], df["applicant_age_years"])

    if ext_source_cols:
        ext_source = df[ext_source_cols]
        df["ext_source_mean"] = ext_source.mean(axis=1)
        df["ext_source_min"] = ext_source.min(axis=1)
        df["ext_source_max"] = ext_source.max(axis=1)
        df["ext_source_std"] = ext_source.std(axis=1)
        df["ext_source_available_count"] = ext_source.notna().sum(axis=1)

    if doc_flag_cols:
        df["document_flag_count"] = df[doc_flag_cols].sum(axis=1)

    if contact_cols:
        df["contact_channel_count"] = df[contact_cols].sum(axis=1)

    if bureau_request_cols:
        recent_request_cols = [
            col
            for col in bureau_request_cols
            if col in {"AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK"}
        ]
        df["bureau_request_total_count"] = df[bureau_request_cols].sum(axis=1)
        df["bureau_request_recent_count"] = df[recent_request_cols].sum(axis=1) if recent_request_cols else np.nan

    df["application_missing_field_count"] = df[raw_feature_cols].isna().sum(axis=1)
    df["application_missing_field_ratio"] = df["application_missing_field_count"] / len(raw_feature_cols)
    return df


def write_quality_outputs(df: pd.DataFrame, sentinel_rows: list[dict[str, object]]) -> None:
    target_train = df[df["dataset"] == "train"]["TARGET"]
    quality = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "rows": len(part),
                "columns": part.shape[1],
                "target_bad_count": int(part["TARGET"].sum()) if dataset == "train" else np.nan,
                "target_bad_rate": float(part["TARGET"].mean()) if dataset == "train" else np.nan,
                "avg_missing_fields": float(part["application_missing_field_count"].mean()),
                "avg_missing_field_ratio": float(part["application_missing_field_ratio"].mean()),
            }
            for dataset, part in df.groupby("dataset")
        ]
    )
    quality.to_csv(OUTPUT_TABLE_DIR / "application_base_quality.csv", index=False)
    pd.DataFrame(sentinel_rows).to_csv(OUTPUT_TABLE_DIR / "application_day_sentinel_profile.csv", index=False)

    engineered_fields = [
        "applicant_age_years",
        "employment_years",
        "registration_age_years",
        "id_publish_years",
        "last_phone_change_years",
        "credit_to_income_ratio",
        "annuity_to_income_ratio",
        "credit_to_annuity_ratio",
        "goods_to_credit_ratio",
        "income_per_family_member",
        "children_to_family_ratio",
        "employment_to_age_ratio",
        "ext_source_mean",
        "ext_source_min",
        "ext_source_max",
        "ext_source_std",
        "ext_source_available_count",
        "document_flag_count",
        "contact_channel_count",
        "bureau_request_total_count",
        "bureau_request_recent_count",
        "application_missing_field_count",
        "application_missing_field_ratio",
    ]

    md = [
        "# 申请主表清洗口径",
        "",
        "本文件由 `src/02_clean_tables.py` 自动生成。",
        "",
        "## 清洗规则",
        "",
        f"- 将 `DAYS_*` 字段中的特殊值 `{SPECIAL_DAY_SENTINEL}` 视为缺失，并保留对应哨兵标记字段。",
        "- 将 `XNA`、`Unknown` 统一视为未知/缺失类别，避免把无效类别当作稳定业务分群。",
        "- 不修改 `data/raw/` 原始 CSV；清洗后的申请底座输出到 `data/interim/application_base.parquet`。",
        "- `TARGET` 只作为训练标签保留，不参与特征衍生。",
        "",
        "## 衍生业务字段",
        "",
    ]
    md.extend([f"- `{field}`" for field in engineered_fields if field in df.columns])
    md.extend(
        [
            "",
            "## 训练集标签概览",
            "",
            f"- 样本数：{target_train.notna().sum():,}",
            f"- 坏样本数：{int(target_train.sum()):,}",
            f"- 坏样本率：{target_train.mean():.2%}",
            "",
            "## 业务解释",
            "",
            "- 申请主表特征主要解释本次申请条件、客户基本偿债能力、信息完整度和外部评分可得性。",
            "- `credit_to_income_ratio`、`annuity_to_income_ratio` 对应授信压力和月供压力。",
            "- `application_missing_field_ratio` 可作为资料完整度或数据质量的代理变量。",
            "- 后续历史表特征会补充征信、历史申请、还款纪律和额度使用情况。",
            "",
        ]
    )
    (DOCS_DIR / "data_cleaning_rules.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    train = read_application_file("train", "application_train.csv")
    test = read_application_file("test", "application_test.csv")
    application = pd.concat([train, test], ignore_index=True)
    application = normalize_categories(application)
    application, sentinel_rows = clean_day_sentinels(application)
    application = add_application_business_features(application)

    out_path = INTERIM_DIR / "application_base.parquet"
    application.to_parquet(out_path, index=False)
    write_quality_outputs(application, sentinel_rows)

    print(f"Wrote {out_path}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'application_base_quality.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'application_day_sentinel_profile.csv'}")
    print(f"Wrote {DOCS_DIR / 'data_cleaning_rules.md'}")


if __name__ == "__main__":
    main()
