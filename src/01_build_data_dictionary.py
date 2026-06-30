from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]

DESCRIPTION_TABLE_MAP = {
    "application_train": "application_{train|test}.csv",
    "application_test": "application_{train|test}.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return next(csv.reader(f))


def load_descriptions() -> pd.DataFrame:
    path = RAW_DIR / "HomeCredit_columns_description.csv"
    desc = pd.read_csv(path, encoding="latin1")
    desc = desc.rename(columns={"Table": "source_table_doc", "Row": "column", "Description": "description"})
    return desc[["source_table_doc", "column", "description", "Special"]]


def classify_business_category(column: str) -> str:
    if column == "TARGET":
        return "label"
    if column.startswith("SK_ID"):
        return "join_key"
    if column.startswith("EXT_SOURCE"):
        return "external_score"
    if column.startswith("AMT"):
        return "amount_debt_income"
    if column.startswith("DAYS"):
        return "timing_recency"
    if column.startswith("CNT"):
        return "count_family_or_terms"
    if column.startswith("FLAG"):
        return "binary_flag"
    if "STATUS" in column:
        return "status_outcome"
    if column.startswith(("NAME", "CODE", "OCCUPATION", "ORGANIZATION", "WALLSMATERIAL", "FONDKAPREMONT", "HOUSETYPE")):
        return "categorical_profile"
    if column.startswith(("NUM", "OBS", "DEF")):
        return "count_or_credit_bureau_summary"
    if column.startswith(("RATE", "REGION")):
        return "rate_region_context"
    return "other"


def classify_data_domain(table: str) -> str:
    if table.startswith("application"):
        return "current_application"
    if table.startswith("bureau"):
        return "external_credit_history"
    if table == "previous_application":
        return "internal_previous_application"
    if table == "installments_payments":
        return "internal_repayment_history"
    if table == "credit_card_balance":
        return "internal_credit_card_history"
    if table == "POS_CASH_balance":
        return "internal_pos_cash_history"
    return "unknown"


def classify_availability(table: str, column: str) -> tuple[str, str]:
    if column == "TARGET":
        return "label_only_not_feature", "high"
    if column.startswith("SK_ID"):
        return "join_key_not_model_feature", "low"
    if table.startswith("application"):
        return "available_at_application", "low"
    return "historical_before_current_application_assumed", "medium"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    desc = load_descriptions()
    rows = []
    for file_name in RAW_FILES:
        table = file_name.replace(".csv", "")
        path = RAW_DIR / file_name
        columns = read_header(path)
        source_doc = DESCRIPTION_TABLE_MAP[table]
        table_desc = desc[desc["source_table_doc"] == source_doc].set_index("column")
        for col in columns:
            availability, leakage_risk = classify_availability(table, col)
            description = ""
            special = ""
            if col in table_desc.index:
                description = str(table_desc.loc[col, "description"])
                special = str(table_desc.loc[col, "Special"])
                if special == "nan":
                    special = ""
            rows.append(
                {
                    "table": table,
                    "column": col,
                    "data_domain": classify_data_domain(table),
                    "business_category": classify_business_category(col),
                    "application_time_availability": availability,
                    "leakage_risk_initial": leakage_risk,
                    "use_in_model_initial": "no" if col == "TARGET" or col.startswith("SK_ID") else "candidate",
                    "aggregation_needed": "no" if table.startswith("application") else "yes_to_SK_ID_CURR",
                    "description": description,
                    "special": special,
                    "review_note": "",
                }
            )

    dictionary = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "data_dictionary_initial.csv"
    dictionary.to_csv(out_path, index=False)

    risk_counts = (
        dictionary.groupby(["data_domain", "leakage_risk_initial"])
        .size()
        .reset_index(name="columns")
        .sort_values(["data_domain", "leakage_risk_initial"])
    )
    risk_counts.to_csv(PROCESSED_DIR / "leakage_risk_summary_initial.csv", index=False)

    md = [
        "# 初始防泄露审计",
        "",
        "本文件由 `src/01_build_data_dictionary.py` 生成。",
        "",
        "## 核心原则",
        "",
        "- `TARGET` 是贷后表现标签，只能用于训练监督信号，不能进入特征。",
        "- `SK_ID_*` 只用于连接和聚合，不作为模型业务特征。",
        "- `application_train/test` 中除标签和主键外，初步视为申请时点可见字段。",
        "- 历史表字段初步视为历史可见，但必须在特征工程中确认它们发生在当前申请之前。",
        "- 还款、余额、历史申请等表需要聚合到 `SK_ID_CURR` 申请粒度。",
        "",
        "## 初始泄露风险汇总",
        "",
        "| data_domain | leakage_risk_initial | columns |",
        "|---|---|---:|",
    ]
    for _, row in risk_counts.iterrows():
        md.append(f"| {row['data_domain']} | {row['leakage_risk_initial']} | {row['columns']} |")
    md.extend(
        [
            "",
            "## 下一步人工复核",
            "",
            "- 逐项检查 `DAYS_*` 字段是否相对当前申请日，避免使用当前申请后的表现。",
            "- 对历史还款、信用卡、POS余额表设置历史窗口，保留窗口口径。",
            "- 对缺失率极高字段决定是否作为缺失信号、分箱变量或剔除。",
            "- 对 `EXT_SOURCE_*` 作为外部评分单独解释，避免黑箱依赖。",
            "",
        ]
    )
    (DOCS_DIR / "leakage_audit_initial.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {PROCESSED_DIR / 'leakage_risk_summary_initial.csv'}")
    print(f"Wrote {DOCS_DIR / 'leakage_audit_initial.md'}")


if __name__ == "__main__":
    main()
