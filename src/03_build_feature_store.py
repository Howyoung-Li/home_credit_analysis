from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

SPECIAL_DAY_SENTINEL = 365243
RECENT_DAY_WINDOWS = {"6m": 180, "12m": 365, "24m": 730}
RECENT_MONTH_WINDOWS = {"6m": 6, "12m": 12, "24m": 24}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def clean_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in [col for col in df.columns if col.startswith("DAYS")]:
        df.loc[df[col] == SPECIAL_DAY_SENTINEL, col] = np.nan
    return df


def add_manifest_rows(
    rows: list[dict[str, str]],
    columns: list[str],
    source_table: str,
    feature_group: str,
    business_meaning: str,
    aggregation_logic: str,
) -> None:
    for col in columns:
        rows.append(
            {
                "feature": col,
                "source_table": source_table,
                "feature_group": feature_group,
                "business_meaning": business_meaning,
                "aggregation_logic": aggregation_logic,
                "application_time_availability": "historical_or_application_time_available",
                "leakage_note": "初步视为申请时点可用；后续建模前继续复核时间窗口。",
            }
        )


def merge_feature_block(features: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    return features.merge(block, on="SK_ID_CURR", how="left")


def build_bureau_features(manifest_rows: list[dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_TYPE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
    ]
    bureau = pd.read_csv(RAW_DIR / "bureau.csv", usecols=usecols, low_memory=False)
    bureau = clean_day_columns(bureau)
    bureau["bureau_is_active"] = (bureau["CREDIT_ACTIVE"] == "Active").astype("int8")
    bureau["bureau_is_closed"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype("int8")
    bureau["bureau_has_overdue"] = (
        (bureau["CREDIT_DAY_OVERDUE"].fillna(0) > 0) | (bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0)
    ).astype("int8")
    bureau["bureau_debt_to_credit"] = safe_divide(bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"])
    bureau["bureau_limit_to_credit"] = safe_divide(bureau["AMT_CREDIT_SUM_LIMIT"], bureau["AMT_CREDIT_SUM"])
    bureau["bureau_is_consumer_credit"] = (bureau["CREDIT_TYPE"] == "Consumer credit").astype("int8")
    bureau["bureau_is_credit_card"] = (bureau["CREDIT_TYPE"] == "Credit card").astype("int8")

    features = bureau.groupby("SK_ID_CURR").agg(
        bureau_loan_count=("SK_ID_BUREAU", "count"),
        bureau_active_loan_count=("bureau_is_active", "sum"),
        bureau_closed_loan_count=("bureau_is_closed", "sum"),
        bureau_overdue_loan_count=("bureau_has_overdue", "sum"),
        bureau_max_days_overdue=("CREDIT_DAY_OVERDUE", "max"),
        bureau_sum_credit=("AMT_CREDIT_SUM", "sum"),
        bureau_mean_credit=("AMT_CREDIT_SUM", "mean"),
        bureau_max_credit=("AMT_CREDIT_SUM", "max"),
        bureau_sum_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_mean_debt=("AMT_CREDIT_SUM_DEBT", "mean"),
        bureau_sum_overdue_amount=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_max_overdue_amount=("AMT_CREDIT_SUM_OVERDUE", "max"),
        bureau_sum_credit_limit=("AMT_CREDIT_SUM_LIMIT", "sum"),
        bureau_max_credit_max_overdue=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_sum_credit_prolong=("CNT_CREDIT_PROLONG", "sum"),
        bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
        bureau_days_credit_min=("DAYS_CREDIT", "min"),
        bureau_days_credit_max=("DAYS_CREDIT", "max"),
        bureau_days_credit_update_max=("DAYS_CREDIT_UPDATE", "max"),
        bureau_credit_type_count=("CREDIT_TYPE", "nunique"),
        bureau_consumer_credit_count=("bureau_is_consumer_credit", "sum"),
        bureau_credit_card_count=("bureau_is_credit_card", "sum"),
        bureau_debt_to_credit_mean=("bureau_debt_to_credit", "mean"),
        bureau_debt_to_credit_max=("bureau_debt_to_credit", "max"),
        bureau_limit_to_credit_mean=("bureau_limit_to_credit", "mean"),
        bureau_annuity_sum=("AMT_ANNUITY", "sum"),
        bureau_annuity_mean=("AMT_ANNUITY", "mean"),
    ).reset_index()
    features["bureau_active_loan_ratio"] = safe_divide(features["bureau_active_loan_count"], features["bureau_loan_count"])
    features["bureau_overdue_loan_ratio"] = safe_divide(features["bureau_overdue_loan_count"], features["bureau_loan_count"])
    features["bureau_total_debt_to_credit_ratio"] = safe_divide(features["bureau_sum_debt"], features["bureau_sum_credit"])

    for status_name, status_mask in {
        "active": bureau["bureau_is_active"] == 1,
        "closed": bureau["bureau_is_closed"] == 1,
    }.items():
        part = bureau[status_mask]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"bureau_{status_name}_credit_sum": ("AMT_CREDIT_SUM", "sum"),
                f"bureau_{status_name}_debt_sum": ("AMT_CREDIT_SUM_DEBT", "sum"),
                f"bureau_{status_name}_overdue_amount_sum": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
                f"bureau_{status_name}_days_credit_mean": ("DAYS_CREDIT", "mean"),
                f"bureau_{status_name}_debt_to_credit_mean": ("bureau_debt_to_credit", "mean"),
                f"bureau_{status_name}_debt_to_credit_max": ("bureau_debt_to_credit", "max"),
            }
        ).reset_index()
        block[f"bureau_{status_name}_total_debt_to_credit_ratio"] = safe_divide(
            block[f"bureau_{status_name}_debt_sum"], block[f"bureau_{status_name}_credit_sum"]
        )
        features = merge_feature_block(features, block)

    for product_name, product_mask in {
        "consumer_credit": bureau["bureau_is_consumer_credit"] == 1,
        "credit_card": bureau["bureau_is_credit_card"] == 1,
    }.items():
        part = bureau[product_mask]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"bureau_{product_name}_credit_sum": ("AMT_CREDIT_SUM", "sum"),
                f"bureau_{product_name}_debt_sum": ("AMT_CREDIT_SUM_DEBT", "sum"),
                f"bureau_{product_name}_overdue_count": ("bureau_has_overdue", "sum"),
                f"bureau_{product_name}_debt_to_credit_mean": ("bureau_debt_to_credit", "mean"),
                f"bureau_{product_name}_debt_to_credit_max": ("bureau_debt_to_credit", "max"),
            }
        ).reset_index()
        block[f"bureau_{product_name}_total_debt_to_credit_ratio"] = safe_divide(
            block[f"bureau_{product_name}_debt_sum"], block[f"bureau_{product_name}_credit_sum"]
        )
        features = merge_feature_block(features, block)

    for window_name, days in RECENT_DAY_WINDOWS.items():
        part = bureau[(bureau["DAYS_CREDIT"] <= 0) & (bureau["DAYS_CREDIT"] >= -days)]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"bureau_recent_{window_name}_loan_count": ("SK_ID_BUREAU", "count"),
                f"bureau_recent_{window_name}_active_loan_count": ("bureau_is_active", "sum"),
                f"bureau_recent_{window_name}_overdue_loan_count": ("bureau_has_overdue", "sum"),
                f"bureau_recent_{window_name}_credit_sum": ("AMT_CREDIT_SUM", "sum"),
                f"bureau_recent_{window_name}_debt_sum": ("AMT_CREDIT_SUM_DEBT", "sum"),
                f"bureau_recent_{window_name}_debt_to_credit_mean": ("bureau_debt_to_credit", "mean"),
                f"bureau_recent_{window_name}_debt_to_credit_max": ("bureau_debt_to_credit", "max"),
                f"bureau_recent_{window_name}_max_days_overdue": ("CREDIT_DAY_OVERDUE", "max"),
            }
        ).reset_index()
        block[f"bureau_recent_{window_name}_overdue_loan_ratio"] = safe_divide(
            block[f"bureau_recent_{window_name}_overdue_loan_count"],
            block[f"bureau_recent_{window_name}_loan_count"],
        )
        block[f"bureau_recent_{window_name}_total_debt_to_credit_ratio"] = safe_divide(
            block[f"bureau_recent_{window_name}_debt_sum"],
            block[f"bureau_recent_{window_name}_credit_sum"],
        )
        features = merge_feature_block(features, block)

    add_manifest_rows(
        manifest_rows,
        [col for col in features.columns if col != "SK_ID_CURR"],
        "bureau.csv",
        "external_credit_history",
        "外部征信历史负债、逾期、活跃贷款、分产品结构和近期多头风险。",
        "按 SK_ID_CURR 聚合外部征信贷款记录，包含 active/closed、产品类型和 6/12/24 个月窗口。",
    )
    bureau_map = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    del bureau
    gc.collect()
    return features, bureau_map


def build_bureau_balance_features(
    bureau_map: pd.DataFrame, manifest_rows: list[dict[str, str]]
) -> pd.DataFrame:
    bb = pd.read_csv(
        RAW_DIR / "bureau_balance.csv",
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        dtype={"STATUS": "string"},
    )
    bb = bb.merge(bureau_map, on="SK_ID_BUREAU", how="left")
    bb["bureau_balance_status_num"] = bb["STATUS"].map({"C": 0, "X": np.nan, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}).astype("float")
    bb["bureau_balance_bad_month"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
    bb["bureau_balance_severe_bad_month"] = bb["STATUS"].isin(["3", "4", "5"]).astype("int8")
    bb["bureau_balance_closed_month"] = (bb["STATUS"] == "C").astype("int8")
    bb["bureau_balance_no_info_month"] = (bb["STATUS"] == "X").astype("int8")

    features = bb.groupby("SK_ID_CURR").agg(
        bureau_balance_month_count=("MONTHS_BALANCE", "count"),
        bureau_balance_account_count=("SK_ID_BUREAU", "nunique"),
        bureau_balance_bad_month_count=("bureau_balance_bad_month", "sum"),
        bureau_balance_severe_bad_month_count=("bureau_balance_severe_bad_month", "sum"),
        bureau_balance_closed_month_count=("bureau_balance_closed_month", "sum"),
        bureau_balance_no_info_month_count=("bureau_balance_no_info_month", "sum"),
        bureau_balance_status_max=("bureau_balance_status_num", "max"),
        bureau_balance_status_mean=("bureau_balance_status_num", "mean"),
        bureau_balance_months_min=("MONTHS_BALANCE", "min"),
        bureau_balance_months_max=("MONTHS_BALANCE", "max"),
    ).reset_index()
    features["bureau_balance_bad_month_ratio"] = safe_divide(
        features["bureau_balance_bad_month_count"], features["bureau_balance_month_count"]
    )
    features["bureau_balance_severe_bad_month_ratio"] = safe_divide(
        features["bureau_balance_severe_bad_month_count"], features["bureau_balance_month_count"]
    )

    for window_name, months in RECENT_MONTH_WINDOWS.items():
        part = bb[(bb["MONTHS_BALANCE"] <= 0) & (bb["MONTHS_BALANCE"] >= -months)]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"bureau_balance_recent_{window_name}_month_count": ("MONTHS_BALANCE", "count"),
                f"bureau_balance_recent_{window_name}_bad_month_count": ("bureau_balance_bad_month", "sum"),
                f"bureau_balance_recent_{window_name}_severe_bad_month_count": ("bureau_balance_severe_bad_month", "sum"),
                f"bureau_balance_recent_{window_name}_status_max": ("bureau_balance_status_num", "max"),
                f"bureau_balance_recent_{window_name}_status_mean": ("bureau_balance_status_num", "mean"),
            }
        ).reset_index()
        block[f"bureau_balance_recent_{window_name}_bad_month_ratio"] = safe_divide(
            block[f"bureau_balance_recent_{window_name}_bad_month_count"],
            block[f"bureau_balance_recent_{window_name}_month_count"],
        )
        block[f"bureau_balance_recent_{window_name}_severe_bad_month_ratio"] = safe_divide(
            block[f"bureau_balance_recent_{window_name}_severe_bad_month_count"],
            block[f"bureau_balance_recent_{window_name}_month_count"],
        )
        features = merge_feature_block(features, block)

    add_manifest_rows(
        manifest_rows,
        [col for col in features.columns if col != "SK_ID_CURR"],
        "bureau_balance.csv",
        "external_credit_monthly_status",
        "外部征信月度状态，衡量历史和近期逾期月份、严重逾期和账户状态完整度。",
        "先关联 bureau 的 SK_ID_CURR，再按申请人聚合全历史与 6/12/24 个月状态窗口。",
    )
    del bb
    gc.collect()
    return features


def build_previous_application_features(manifest_rows: list[dict[str, str]]) -> pd.DataFrame:
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "NAME_CONTRACT_TYPE",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        "CODE_REJECT_REASON",
        "NAME_CLIENT_TYPE",
        "CHANNEL_TYPE",
        "NAME_YIELD_GROUP",
        "NFLAG_INSURED_ON_APPROVAL",
    ]
    prev = pd.read_csv(RAW_DIR / "previous_application.csv", usecols=usecols, low_memory=False)
    prev = clean_day_columns(prev)
    prev["prev_is_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype("int8")
    prev["prev_is_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")
    prev["prev_is_canceled"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype("int8")
    prev["prev_is_cash"] = (prev["NAME_CONTRACT_TYPE"] == "Cash loans").astype("int8")
    prev["prev_is_revolving"] = (prev["NAME_CONTRACT_TYPE"] == "Revolving loans").astype("int8")
    prev["prev_credit_to_application"] = safe_divide(prev["AMT_CREDIT"], prev["AMT_APPLICATION"])
    prev["prev_down_payment_to_credit"] = safe_divide(prev["AMT_DOWN_PAYMENT"], prev["AMT_CREDIT"])

    features = prev.groupby("SK_ID_CURR").agg(
        previous_application_count=("SK_ID_PREV", "count"),
        previous_approved_count=("prev_is_approved", "sum"),
        previous_refused_count=("prev_is_refused", "sum"),
        previous_canceled_count=("prev_is_canceled", "sum"),
        previous_cash_count=("prev_is_cash", "sum"),
        previous_revolving_count=("prev_is_revolving", "sum"),
        previous_application_amount_sum=("AMT_APPLICATION", "sum"),
        previous_application_amount_mean=("AMT_APPLICATION", "mean"),
        previous_credit_amount_sum=("AMT_CREDIT", "sum"),
        previous_credit_amount_mean=("AMT_CREDIT", "mean"),
        previous_annuity_mean=("AMT_ANNUITY", "mean"),
        previous_down_payment_mean=("AMT_DOWN_PAYMENT", "mean"),
        previous_goods_price_mean=("AMT_GOODS_PRICE", "mean"),
        previous_days_decision_mean=("DAYS_DECISION", "mean"),
        previous_days_decision_max=("DAYS_DECISION", "max"),
        previous_cnt_payment_mean=("CNT_PAYMENT", "mean"),
        previous_credit_to_application_mean=("prev_credit_to_application", "mean"),
        previous_down_payment_to_credit_mean=("prev_down_payment_to_credit", "mean"),
        previous_reject_reason_count=("CODE_REJECT_REASON", "nunique"),
        previous_client_type_count=("NAME_CLIENT_TYPE", "nunique"),
        previous_channel_count=("CHANNEL_TYPE", "nunique"),
        previous_yield_group_count=("NAME_YIELD_GROUP", "nunique"),
        previous_insured_on_approval_rate=("NFLAG_INSURED_ON_APPROVAL", "mean"),
    ).reset_index()
    features["previous_approval_rate"] = safe_divide(features["previous_approved_count"], features["previous_application_count"])
    features["previous_refusal_rate"] = safe_divide(features["previous_refused_count"], features["previous_application_count"])
    features["previous_total_credit_to_current_application_ratio"] = safe_divide(
        features["previous_credit_amount_sum"], features["previous_application_amount_sum"]
    )

    for status_name, status_mask in {
        "approved": prev["prev_is_approved"] == 1,
        "refused": prev["prev_is_refused"] == 1,
        "canceled": prev["prev_is_canceled"] == 1,
    }.items():
        part = prev[status_mask]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"previous_{status_name}_amount_mean": ("AMT_APPLICATION", "mean"),
                f"previous_{status_name}_credit_mean": ("AMT_CREDIT", "mean"),
                f"previous_{status_name}_credit_sum": ("AMT_CREDIT", "sum"),
                f"previous_{status_name}_days_decision_max": ("DAYS_DECISION", "max"),
                f"previous_{status_name}_days_decision_mean": ("DAYS_DECISION", "mean"),
                f"previous_{status_name}_cnt_payment_mean": ("CNT_PAYMENT", "mean"),
                f"previous_{status_name}_credit_to_application_mean": ("prev_credit_to_application", "mean"),
            }
        ).reset_index()
        features = merge_feature_block(features, block)

    for product_name, product_mask in {
        "cash": prev["prev_is_cash"] == 1,
        "revolving": prev["prev_is_revolving"] == 1,
    }.items():
        part = prev[product_mask]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"previous_{product_name}_approved_count": ("prev_is_approved", "sum"),
                f"previous_{product_name}_refused_count": ("prev_is_refused", "sum"),
                f"previous_{product_name}_credit_mean": ("AMT_CREDIT", "mean"),
                f"previous_{product_name}_credit_sum": ("AMT_CREDIT", "sum"),
                f"previous_{product_name}_credit_to_application_mean": ("prev_credit_to_application", "mean"),
            }
        ).reset_index()
        block[f"previous_{product_name}_refusal_rate"] = safe_divide(
            block[f"previous_{product_name}_refused_count"],
            block[f"previous_{product_name}_approved_count"] + block[f"previous_{product_name}_refused_count"],
        )
        features = merge_feature_block(features, block)

    for window_name, days in RECENT_DAY_WINDOWS.items():
        part = prev[(prev["DAYS_DECISION"] <= 0) & (prev["DAYS_DECISION"] >= -days)]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"previous_recent_{window_name}_application_count": ("SK_ID_PREV", "count"),
                f"previous_recent_{window_name}_approved_count": ("prev_is_approved", "sum"),
                f"previous_recent_{window_name}_refused_count": ("prev_is_refused", "sum"),
                f"previous_recent_{window_name}_credit_mean": ("AMT_CREDIT", "mean"),
                f"previous_recent_{window_name}_credit_sum": ("AMT_CREDIT", "sum"),
                f"previous_recent_{window_name}_credit_to_application_mean": ("prev_credit_to_application", "mean"),
                f"previous_recent_{window_name}_days_decision_max": ("DAYS_DECISION", "max"),
            }
        ).reset_index()
        block[f"previous_recent_{window_name}_refusal_rate"] = safe_divide(
            block[f"previous_recent_{window_name}_refused_count"],
            block[f"previous_recent_{window_name}_application_count"],
        )
        features = merge_feature_block(features, block)

    add_manifest_rows(
        manifest_rows,
        [col for col in features.columns if col != "SK_ID_CURR"],
        "previous_application.csv",
        "internal_previous_application",
        "Home Credit 历史申请、近期申请、分状态通过/拒绝和分产品授信需求。",
        "按 SK_ID_CURR 聚合历史申请记录，包含 approved/refused/canceled、cash/revolving 和 6/12/24 个月窗口。",
    )
    del prev
    gc.collect()
    return features


def build_installment_features(manifest_rows: list[dict[str, str]]) -> pd.DataFrame:
    inst = pd.read_csv(
        RAW_DIR / "installments_payments.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
        low_memory=False,
    )
    inst = clean_day_columns(inst)
    inst["installment_days_late"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["installment_days_early"] = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    inst["installment_is_late"] = (inst["installment_days_late"].fillna(0) > 0).astype("int8")
    inst["installment_payment_ratio"] = safe_divide(inst["AMT_PAYMENT"], inst["AMT_INSTALMENT"])
    inst["installment_shortfall_amount"] = (inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]).clip(lower=0)
    inst["installment_has_shortfall"] = (inst["installment_shortfall_amount"].fillna(0) > 0).astype("int8")

    features = inst.groupby("SK_ID_CURR").agg(
        installment_record_count=("SK_ID_PREV", "count"),
        installment_contract_count=("SK_ID_PREV", "nunique"),
        installment_late_count=("installment_is_late", "sum"),
        installment_late_days_mean=("installment_days_late", "mean"),
        installment_late_days_max=("installment_days_late", "max"),
        installment_early_days_mean=("installment_days_early", "mean"),
        installment_payment_ratio_mean=("installment_payment_ratio", "mean"),
        installment_payment_ratio_min=("installment_payment_ratio", "min"),
        installment_shortfall_count=("installment_has_shortfall", "sum"),
        installment_shortfall_amount_sum=("installment_shortfall_amount", "sum"),
        installment_shortfall_amount_max=("installment_shortfall_amount", "max"),
        installment_amount_sum=("AMT_INSTALMENT", "sum"),
        installment_payment_sum=("AMT_PAYMENT", "sum"),
        installment_last_due_day=("DAYS_INSTALMENT", "max"),
        installment_first_due_day=("DAYS_INSTALMENT", "min"),
    ).reset_index()
    features["installment_late_ratio"] = safe_divide(features["installment_late_count"], features["installment_record_count"])
    features["installment_shortfall_ratio"] = safe_divide(
        features["installment_shortfall_count"], features["installment_record_count"]
    )
    features["installment_total_payment_ratio"] = safe_divide(
        features["installment_payment_sum"], features["installment_amount_sum"]
    )

    for window_name, days in RECENT_DAY_WINDOWS.items():
        part = inst[(inst["DAYS_INSTALMENT"] <= 0) & (inst["DAYS_INSTALMENT"] >= -days)]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"installment_recent_{window_name}_record_count": ("SK_ID_PREV", "count"),
                f"installment_recent_{window_name}_contract_count": ("SK_ID_PREV", "nunique"),
                f"installment_recent_{window_name}_late_count": ("installment_is_late", "sum"),
                f"installment_recent_{window_name}_late_days_mean": ("installment_days_late", "mean"),
                f"installment_recent_{window_name}_late_days_max": ("installment_days_late", "max"),
                f"installment_recent_{window_name}_payment_ratio_mean": ("installment_payment_ratio", "mean"),
                f"installment_recent_{window_name}_payment_ratio_min": ("installment_payment_ratio", "min"),
                f"installment_recent_{window_name}_shortfall_count": ("installment_has_shortfall", "sum"),
                f"installment_recent_{window_name}_shortfall_amount_sum": ("installment_shortfall_amount", "sum"),
                f"installment_recent_{window_name}_amount_sum": ("AMT_INSTALMENT", "sum"),
                f"installment_recent_{window_name}_payment_sum": ("AMT_PAYMENT", "sum"),
            }
        ).reset_index()
        block[f"installment_recent_{window_name}_late_ratio"] = safe_divide(
            block[f"installment_recent_{window_name}_late_count"],
            block[f"installment_recent_{window_name}_record_count"],
        )
        block[f"installment_recent_{window_name}_shortfall_ratio"] = safe_divide(
            block[f"installment_recent_{window_name}_shortfall_count"],
            block[f"installment_recent_{window_name}_record_count"],
        )
        block[f"installment_recent_{window_name}_total_payment_ratio"] = safe_divide(
            block[f"installment_recent_{window_name}_payment_sum"],
            block[f"installment_recent_{window_name}_amount_sum"],
        )
        features = merge_feature_block(features, block)

    add_manifest_rows(
        manifest_rows,
        [col for col in features.columns if col != "SK_ID_CURR"],
        "installments_payments.csv",
        "internal_repayment_history",
        "历史和近期分期还款纪律，衡量逾期、提前还款、短付和整体还款覆盖。",
        "按 SK_ID_CURR 聚合历史分期应还与实还记录，包含 6/12/24 个月还款窗口。",
    )
    del inst
    gc.collect()
    return features


def build_credit_card_features(manifest_rows: list[dict[str, str]]) -> pd.DataFrame:
    cc = pd.read_csv(
        RAW_DIR / "credit_card_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_CURRENT",
            "AMT_INST_MIN_REGULARITY",
            "AMT_PAYMENT_TOTAL_CURRENT",
            "CNT_DRAWINGS_CURRENT",
            "NAME_CONTRACT_STATUS",
            "SK_DPD",
            "SK_DPD_DEF",
        ],
        low_memory=False,
    )
    cc["credit_card_utilization"] = safe_divide(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["credit_card_payment_to_min_due"] = safe_divide(cc["AMT_PAYMENT_TOTAL_CURRENT"], cc["AMT_INST_MIN_REGULARITY"])
    cc["credit_card_dpd_month"] = (cc["SK_DPD"].fillna(0) > 0).astype("int8")
    cc["credit_card_def_dpd_month"] = (cc["SK_DPD_DEF"].fillna(0) > 0).astype("int8")
    cc["credit_card_active_month"] = (cc["NAME_CONTRACT_STATUS"] == "Active").astype("int8")

    features = cc.groupby("SK_ID_CURR").agg(
        credit_card_month_count=("MONTHS_BALANCE", "count"),
        credit_card_contract_count=("SK_ID_PREV", "nunique"),
        credit_card_active_month_count=("credit_card_active_month", "sum"),
        credit_card_balance_mean=("AMT_BALANCE", "mean"),
        credit_card_balance_max=("AMT_BALANCE", "max"),
        credit_card_credit_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        credit_card_credit_limit_max=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
        credit_card_utilization_mean=("credit_card_utilization", "mean"),
        credit_card_utilization_max=("credit_card_utilization", "max"),
        credit_card_drawings_sum=("AMT_DRAWINGS_CURRENT", "sum"),
        credit_card_drawings_mean=("AMT_DRAWINGS_CURRENT", "mean"),
        credit_card_drawing_count_sum=("CNT_DRAWINGS_CURRENT", "sum"),
        credit_card_payment_to_min_due_mean=("credit_card_payment_to_min_due", "mean"),
        credit_card_dpd_month_count=("credit_card_dpd_month", "sum"),
        credit_card_def_dpd_month_count=("credit_card_def_dpd_month", "sum"),
        credit_card_dpd_max=("SK_DPD", "max"),
        credit_card_def_dpd_max=("SK_DPD_DEF", "max"),
        credit_card_months_min=("MONTHS_BALANCE", "min"),
        credit_card_months_max=("MONTHS_BALANCE", "max"),
    ).reset_index()
    features["credit_card_dpd_month_ratio"] = safe_divide(
        features["credit_card_dpd_month_count"], features["credit_card_month_count"]
    )
    features["credit_card_def_dpd_month_ratio"] = safe_divide(
        features["credit_card_def_dpd_month_count"], features["credit_card_month_count"]
    )
    features["credit_card_active_month_ratio"] = safe_divide(
        features["credit_card_active_month_count"], features["credit_card_month_count"]
    )

    for window_name, months in RECENT_MONTH_WINDOWS.items():
        part = cc[(cc["MONTHS_BALANCE"] <= 0) & (cc["MONTHS_BALANCE"] >= -months)]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"credit_card_recent_{window_name}_month_count": ("MONTHS_BALANCE", "count"),
                f"credit_card_recent_{window_name}_contract_count": ("SK_ID_PREV", "nunique"),
                f"credit_card_recent_{window_name}_active_month_count": ("credit_card_active_month", "sum"),
                f"credit_card_recent_{window_name}_balance_mean": ("AMT_BALANCE", "mean"),
                f"credit_card_recent_{window_name}_balance_max": ("AMT_BALANCE", "max"),
                f"credit_card_recent_{window_name}_credit_limit_mean": ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
                f"credit_card_recent_{window_name}_utilization_mean": ("credit_card_utilization", "mean"),
                f"credit_card_recent_{window_name}_utilization_max": ("credit_card_utilization", "max"),
                f"credit_card_recent_{window_name}_drawings_sum": ("AMT_DRAWINGS_CURRENT", "sum"),
                f"credit_card_recent_{window_name}_payment_to_min_due_mean": ("credit_card_payment_to_min_due", "mean"),
                f"credit_card_recent_{window_name}_dpd_month_count": ("credit_card_dpd_month", "sum"),
                f"credit_card_recent_{window_name}_def_dpd_month_count": ("credit_card_def_dpd_month", "sum"),
                f"credit_card_recent_{window_name}_dpd_max": ("SK_DPD", "max"),
                f"credit_card_recent_{window_name}_def_dpd_max": ("SK_DPD_DEF", "max"),
            }
        ).reset_index()
        block[f"credit_card_recent_{window_name}_dpd_month_ratio"] = safe_divide(
            block[f"credit_card_recent_{window_name}_dpd_month_count"],
            block[f"credit_card_recent_{window_name}_month_count"],
        )
        block[f"credit_card_recent_{window_name}_def_dpd_month_ratio"] = safe_divide(
            block[f"credit_card_recent_{window_name}_def_dpd_month_count"],
            block[f"credit_card_recent_{window_name}_month_count"],
        )
        block[f"credit_card_recent_{window_name}_active_month_ratio"] = safe_divide(
            block[f"credit_card_recent_{window_name}_active_month_count"],
            block[f"credit_card_recent_{window_name}_month_count"],
        )
        features = merge_feature_block(features, block)

    add_manifest_rows(
        manifest_rows,
        [col for col in features.columns if col != "SK_ID_CURR"],
        "credit_card_balance.csv",
        "internal_credit_card_history",
        "历史和近期信用卡余额、额度使用、支取行为和逾期状态。",
        "按 SK_ID_CURR 聚合信用卡月度余额记录，包含 6/12/24 个月窗口。",
    )
    del cc
    gc.collect()
    return features


def build_pos_cash_features(manifest_rows: list[dict[str, str]]) -> pd.DataFrame:
    pos = pd.read_csv(
        RAW_DIR / "POS_CASH_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "MONTHS_BALANCE",
            "CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
            "NAME_CONTRACT_STATUS",
            "SK_DPD",
            "SK_DPD_DEF",
        ],
        low_memory=False,
    )
    pos["pos_dpd_month"] = (pos["SK_DPD"].fillna(0) > 0).astype("int8")
    pos["pos_def_dpd_month"] = (pos["SK_DPD_DEF"].fillna(0) > 0).astype("int8")
    pos["pos_active_month"] = (pos["NAME_CONTRACT_STATUS"] == "Active").astype("int8")
    pos["pos_completed_month"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype("int8")

    features = pos.groupby("SK_ID_CURR").agg(
        pos_month_count=("MONTHS_BALANCE", "count"),
        pos_contract_count=("SK_ID_PREV", "nunique"),
        pos_active_month_count=("pos_active_month", "sum"),
        pos_completed_month_count=("pos_completed_month", "sum"),
        pos_dpd_month_count=("pos_dpd_month", "sum"),
        pos_def_dpd_month_count=("pos_def_dpd_month", "sum"),
        pos_dpd_max=("SK_DPD", "max"),
        pos_def_dpd_max=("SK_DPD_DEF", "max"),
        pos_instalment_mean=("CNT_INSTALMENT", "mean"),
        pos_instalment_future_mean=("CNT_INSTALMENT_FUTURE", "mean"),
        pos_instalment_future_max=("CNT_INSTALMENT_FUTURE", "max"),
        pos_months_min=("MONTHS_BALANCE", "min"),
        pos_months_max=("MONTHS_BALANCE", "max"),
    ).reset_index()
    features["pos_dpd_month_ratio"] = safe_divide(features["pos_dpd_month_count"], features["pos_month_count"])
    features["pos_def_dpd_month_ratio"] = safe_divide(features["pos_def_dpd_month_count"], features["pos_month_count"])
    features["pos_active_month_ratio"] = safe_divide(features["pos_active_month_count"], features["pos_month_count"])

    for window_name, months in RECENT_MONTH_WINDOWS.items():
        part = pos[(pos["MONTHS_BALANCE"] <= 0) & (pos["MONTHS_BALANCE"] >= -months)]
        if part.empty:
            continue
        block = part.groupby("SK_ID_CURR").agg(
            **{
                f"pos_recent_{window_name}_month_count": ("MONTHS_BALANCE", "count"),
                f"pos_recent_{window_name}_contract_count": ("SK_ID_PREV", "nunique"),
                f"pos_recent_{window_name}_active_month_count": ("pos_active_month", "sum"),
                f"pos_recent_{window_name}_completed_month_count": ("pos_completed_month", "sum"),
                f"pos_recent_{window_name}_dpd_month_count": ("pos_dpd_month", "sum"),
                f"pos_recent_{window_name}_def_dpd_month_count": ("pos_def_dpd_month", "sum"),
                f"pos_recent_{window_name}_dpd_max": ("SK_DPD", "max"),
                f"pos_recent_{window_name}_def_dpd_max": ("SK_DPD_DEF", "max"),
                f"pos_recent_{window_name}_instalment_future_mean": ("CNT_INSTALMENT_FUTURE", "mean"),
                f"pos_recent_{window_name}_instalment_future_max": ("CNT_INSTALMENT_FUTURE", "max"),
            }
        ).reset_index()
        block[f"pos_recent_{window_name}_dpd_month_ratio"] = safe_divide(
            block[f"pos_recent_{window_name}_dpd_month_count"],
            block[f"pos_recent_{window_name}_month_count"],
        )
        block[f"pos_recent_{window_name}_def_dpd_month_ratio"] = safe_divide(
            block[f"pos_recent_{window_name}_def_dpd_month_count"],
            block[f"pos_recent_{window_name}_month_count"],
        )
        block[f"pos_recent_{window_name}_active_month_ratio"] = safe_divide(
            block[f"pos_recent_{window_name}_active_month_count"],
            block[f"pos_recent_{window_name}_month_count"],
        )
        features = merge_feature_block(features, block)

    add_manifest_rows(
        manifest_rows,
        [col for col in features.columns if col != "SK_ID_CURR"],
        "POS_CASH_balance.csv",
        "internal_pos_cash_history",
        "历史和近期 POS/cash loan 合约月度状态、剩余期数和逾期表现。",
        "按 SK_ID_CURR 聚合 POS/cash loan 月度记录，包含 6/12/24 个月窗口。",
    )
    del pos
    gc.collect()
    return features


def fill_no_history_counts(feature_store: pd.DataFrame) -> pd.DataFrame:
    history_count_columns = [
        "bureau_loan_count",
        "bureau_balance_month_count",
        "previous_application_count",
        "installment_record_count",
        "credit_card_month_count",
        "pos_month_count",
    ]
    for col in history_count_columns:
        if col in feature_store.columns:
            feature_store[f"has_{col.replace('_count', '')}_history"] = feature_store[col].notna().astype("int8")
            feature_store[col] = feature_store[col].fillna(0)

    zero_fill_keywords = (
        "_count",
        "_sum",
        "_month_count",
        "_loan_count",
        "_contract_count",
        "_application_count",
        "_record_count",
    )
    for col in feature_store.columns:
        if col.startswith(("bureau_", "previous_", "installment_", "credit_card_", "pos_")) and col.endswith(
            zero_fill_keywords
        ):
            feature_store[col] = feature_store[col].fillna(0)
    return feature_store


def write_outputs(feature_store: pd.DataFrame, manifest: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    feature_store_path = PROCESSED_DIR / "application_feature_store.parquet"
    feature_store.to_parquet(feature_store_path, index=False)
    feature_store.head(1000).to_csv(PROCESSED_DIR / "application_feature_store_preview.csv", index=False)
    manifest.to_csv(PROCESSED_DIR / "feature_manifest_initial.csv", index=False)

    profile_rows = []
    group_order = [
        "current_application",
        "external_credit_history",
        "external_credit_monthly_status",
        "internal_previous_application",
        "internal_repayment_history",
        "internal_credit_card_history",
        "internal_pos_cash_history",
        "history_coverage",
        "feature_store_quality",
    ]
    for label in group_order:
        cols = manifest.loc[manifest["feature_group"] == label, "feature"].tolist()
        cols = [col for col in cols if col in feature_store.columns]
        profile_rows.append(
            {
                "feature_group": label,
                "feature_count": len(cols),
                "avg_missing_rate": float(feature_store[cols].isna().mean().mean()) if cols else np.nan,
            }
        )
    profile = pd.DataFrame(profile_rows)
    profile.to_csv(OUTPUT_TABLE_DIR / "feature_store_group_profile.csv", index=False)

    train = feature_store[feature_store["dataset"] == "train"]
    md = [
        "# 申请级特征集市初版",
        "",
        "本文件由 `src/03_build_feature_store.py` 自动生成。",
        "",
        "## 输出文件",
        "",
        f"- `data/processed/application_feature_store.parquet`：申请级宽表，{feature_store.shape[0]:,} 行，{feature_store.shape[1]:,} 列。",
        "- `data/processed/application_feature_store_preview.csv`：前 1000 行预览。",
        "- `data/processed/feature_manifest_initial.csv`：初始特征说明表。",
        "- `outputs/tables/feature_store_group_profile.csv`：特征组覆盖率概览。",
        "",
        "## 样本与标签",
        "",
        f"- 训练集申请数：{len(train):,}",
        f"- 训练集坏样本率：{train['TARGET'].mean():.2%}",
        f"- 测试集申请数：{(feature_store['dataset'] == 'test').sum():,}",
        "",
        "## 特征组",
        "",
        "| feature_group | feature_count | avg_missing_rate |",
        "|---|---:|---:|",
    ]
    for _, row in profile.iterrows():
        md.append(f"| {row['feature_group']} | {int(row['feature_count'])} | {row['avg_missing_rate']:.2%} |")
    md.extend(
        [
            "",
            "## 业务使用方式",
            "",
            "- 当前申请字段解释客户基本画像、申请金额、偿债压力、资料完整度和外部评分可得性。",
            "- `bureau_*` 与 `bureau_balance_*` 对应外部征信历史，可用于判断多头借贷、历史逾期和活跃负债。",
            "- `previous_*` 对应 Home Credit 历史申请，可解释历史拒贷率、通过率、渠道和产品偏好。",
            "- `installment_*` 对应历史还款纪律，是贷前准入中最容易转成 reason code 的特征组。",
            "- `credit_card_*` 和 `pos_*` 补充额度使用、月度余额、DPD 和剩余期数。",
            "",
            "## 下一步",
            "",
            "- 用该宽表训练一个可解释 baseline 模型。",
            "- 输出 AUC、KS、Lift、Top 5%/10% 坏样本捕获率。",
            "- 把模型分数转成 approve / manual review / decline 三段式准入策略。",
            "",
        ]
    )
    (DOCS_DIR / "feature_store_initial.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {feature_store_path}")
    print(f"Wrote {PROCESSED_DIR / 'feature_manifest_initial.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'feature_store_group_profile.csv'}")
    print(f"Wrote {DOCS_DIR / 'feature_store_initial.md'}")


def main() -> None:
    manifest_rows: list[dict[str, str]] = []
    application_path = INTERIM_DIR / "application_base.parquet"
    if not application_path.exists():
        raise FileNotFoundError("请先运行 src/02_clean_tables.py 生成 data/interim/application_base.parquet")

    feature_store = pd.read_parquet(application_path)
    add_manifest_rows(
        manifest_rows,
        [col for col in feature_store.columns if col not in {"SK_ID_CURR", "TARGET", "dataset"}],
        "application_train/test.csv",
        "current_application",
        "当前申请资料、申请金额、偿债能力、资料完整度和外部评分。",
        "申请主表清洗与业务比率衍生。",
    )

    bureau_features, bureau_map = build_bureau_features(manifest_rows)
    feature_store = feature_store.merge(bureau_features, on="SK_ID_CURR", how="left")
    del bureau_features
    gc.collect()

    bureau_balance_features = build_bureau_balance_features(bureau_map, manifest_rows)
    feature_store = feature_store.merge(bureau_balance_features, on="SK_ID_CURR", how="left")
    del bureau_balance_features, bureau_map
    gc.collect()

    for builder in [
        build_previous_application_features,
        build_installment_features,
        build_credit_card_features,
        build_pos_cash_features,
    ]:
        features = builder(manifest_rows)
        feature_store = feature_store.merge(features, on="SK_ID_CURR", how="left")
        del features
        gc.collect()

    feature_store = fill_no_history_counts(feature_store)
    add_manifest_rows(
        manifest_rows,
        [col for col in feature_store.columns if col.startswith("has_") and col.endswith("_history")],
        "application_feature_store.parquet",
        "history_coverage",
        "申请人是否存在对应历史数据源记录，用于区分真实零值和无历史记录。",
        "多表聚合后根据关键计数字段是否缺失生成。",
    )
    feature_store["feature_store_missing_field_count"] = feature_store.drop(columns=["dataset", "SK_ID_CURR", "TARGET"]).isna().sum(axis=1)
    feature_store["feature_store_missing_field_ratio"] = safe_divide(
        feature_store["feature_store_missing_field_count"],
        pd.Series(feature_store.shape[1] - 3, index=feature_store.index),
    )

    manifest = pd.DataFrame(manifest_rows).drop_duplicates(subset=["feature"])
    add_manifest_rows(
        manifest_rows,
        ["feature_store_missing_field_count", "feature_store_missing_field_ratio"],
        "application_feature_store.parquet",
        "feature_store_quality",
        "整合后宽表字段完整度，可用于数据质量审计和稳定性监控。",
        "对申请级宽表缺失字段数量和比例进行统计。",
    )
    manifest = pd.DataFrame(manifest_rows).drop_duplicates(subset=["feature"])

    write_outputs(feature_store, manifest)


if __name__ == "__main__":
    main()
