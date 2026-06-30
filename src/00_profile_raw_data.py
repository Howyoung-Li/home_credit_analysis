from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
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
    "HomeCredit_columns_description.csv",
]


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def sample_profile(path: Path, table_name: str, nrows: int = 5000) -> list[dict[str, object]]:
    sample = pd.read_csv(path, nrows=nrows, low_memory=False)
    rows: list[dict[str, object]] = []
    for col in sample.columns:
        values = sample[col].dropna().astype(str).head(3).tolist()
        rows.append(
            {
                "table": table_name,
                "column": col,
                "sample_dtype": str(sample[col].dtype),
                "sample_missing_pct": float(sample[col].isna().mean()),
                "sample_nunique": int(sample[col].nunique(dropna=True)),
                "example_values": " | ".join(values),
            }
        )
    return rows


def target_stats(path: Path) -> tuple[int | None, float | None]:
    header = read_header(path)
    if "TARGET" not in header:
        return None, None
    target = pd.read_csv(path, usecols=["TARGET"])
    bad_count = int(target["TARGET"].sum())
    bad_rate = float(target["TARGET"].mean())
    return bad_count, bad_rate


def infer_key_columns(columns: list[str]) -> str:
    candidates = [c for c in columns if c.startswith("SK_ID")]
    return ", ".join(candidates)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    table_rows = []
    column_rows = []

    for file_name in RAW_FILES:
        path = RAW_DIR / file_name
        if not path.exists():
            table_rows.append(
                {
                    "table": file_name.replace(".csv", ""),
                    "file_name": file_name,
                    "exists": False,
                    "rows": None,
                    "columns": None,
                    "file_mb": None,
                    "key_columns": "",
                    "has_target": False,
                    "bad_count": None,
                    "bad_rate": None,
                }
            )
            continue

        table_name = file_name.replace(".csv", "")
        columns = read_header(path)
        rows = count_csv_rows(path)
        bad_count, bad_rate = target_stats(path)
        table_rows.append(
            {
                "table": table_name,
                "file_name": file_name,
                "exists": True,
                "rows": rows,
                "columns": len(columns),
                "file_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "key_columns": infer_key_columns(columns),
                "has_target": "TARGET" in columns,
                "bad_count": bad_count,
                "bad_rate": bad_rate,
            }
        )

        if table_name != "HomeCredit_columns_description":
            column_rows.extend(sample_profile(path, table_name))

    table_profile = pd.DataFrame(table_rows)
    column_profile = pd.DataFrame(column_rows)

    table_profile.to_csv(TABLE_DIR / "raw_table_profile.csv", index=False)
    column_profile.to_csv(TABLE_DIR / "raw_column_profile_sample.csv", index=False)

    md_lines = [
        "# 原始数据盘点",
        "",
        "本报告由 `src/00_profile_raw_data.py` 自动生成。",
        "",
        "## 表级概览",
        "",
        "| table | rows | columns | file_mb | key_columns | bad_rate |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in table_rows:
        bad_rate_text = "" if row["bad_rate"] is None else f"{row['bad_rate']:.2%}"
        md_lines.append(
            f"| {row['table']} | {row['rows']} | {row['columns']} | "
            f"{row['file_mb']} | {row['key_columns']} | {bad_rate_text} |"
        )

    md_lines.extend(
        [
            "",
            "## 初步业务解读",
            "",
            "- `application_train` 是当前申请主表，`TARGET` 是贷后表现标签。",
            "- 其他表均为历史行为或外部征信数据，需要聚合到 `SK_ID_CURR` 申请粒度。",
            "- 后续建模前必须完成防泄露审计，确保特征在申请时点可见。",
            "- 项目评估应同时报告模型指标和准入策略指标。",
            "",
        ]
    )
    (DOCS_DIR / "raw_data_profile.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {TABLE_DIR / 'raw_table_profile.csv'}")
    print(f"Wrote {TABLE_DIR / 'raw_column_profile_sample.csv'}")
    print(f"Wrote {DOCS_DIR / 'raw_data_profile.md'}")


if __name__ == "__main__":
    main()
