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

PDO = 50
BASE_SCORE = 600
BASE_ODDS = 20
BASE_DECLINE_PCT = 0.10
BASE_MANUAL_REVIEW_PCT = 0.10
DECLINE_PCT_GRID = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
BAD_LOSS_TO_GOOD_PROFIT_GRID = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
STRATEGY_DECLINE_PCT_GRID = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
STRATEGY_MANUAL_REVIEW_PCT_GRID = [0.00, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
MAX_TOTAL_INTERVENTION_PCT = 0.50
COST_SCENARIOS = [
    {
        "cost_scenario": "low_bad_loss_low_manual_value",
        "bad_loss_cost": 3.0,
        "false_decline_good_cost": 1.0,
        "manual_review_cost": 0.10,
        "manual_bad_catch_rate": 0.35,
        "manual_good_false_decline_rate": 0.04,
        "max_direct_decline_pct": 0.10,
        "max_manual_review_pct": 0.10,
        "max_total_intervention_pct": 0.20,
        "description": "坏账损失相对温和，人工审核只拦住少量坏客户。",
    },
    {
        "cost_scenario": "balanced_base",
        "bad_loss_cost": 5.0,
        "false_decline_good_cost": 1.0,
        "manual_review_cost": 0.10,
        "manual_bad_catch_rate": 0.60,
        "manual_good_false_decline_rate": 0.06,
        "max_direct_decline_pct": 0.15,
        "max_manual_review_pct": 0.10,
        "max_total_intervention_pct": 0.25,
        "description": "默认场景：坏客户损失约等于误杀好客户成本的 5 倍，人审能拦住 60% 复核区坏客户，人审产能约 10%。",
    },
    {
        "cost_scenario": "high_bad_loss_strong_manual",
        "bad_loss_cost": 8.0,
        "false_decline_good_cost": 1.0,
        "manual_review_cost": 0.10,
        "manual_bad_catch_rate": 0.70,
        "manual_good_false_decline_rate": 0.08,
        "max_direct_decline_pct": 0.20,
        "max_manual_review_pct": 0.15,
        "max_total_intervention_pct": 0.35,
        "description": "高坏账损失场景，人工审核较有效。",
    },
    {
        "cost_scenario": "expensive_manual_conservative_reject",
        "bad_loss_cost": 5.0,
        "false_decline_good_cost": 1.5,
        "manual_review_cost": 0.25,
        "manual_bad_catch_rate": 0.50,
        "manual_good_false_decline_rate": 0.05,
        "max_direct_decline_pct": 0.10,
        "max_manual_review_pct": 0.10,
        "max_total_intervention_pct": 0.20,
        "description": "人工审核较贵且误杀好客户代价较高，偏保守拒绝。",
    },
]
AMOUNT_WEIGHTED_COST_SCENARIOS = [
    {
        "amount_cost_scenario": "low_lgd_low_margin",
        "lgd_rate": 0.35,
        "net_margin_rate": 0.05,
        "manual_review_cost_rate_of_median_credit": 0.0010,
        "manual_bad_catch_rate": 0.35,
        "manual_good_false_decline_rate": 0.04,
        "max_direct_decline_pct": 0.10,
        "max_manual_review_pct": 0.10,
        "max_total_intervention_pct": 0.20,
        "description": "较低 LGD 和较低净利差，偏保守的金额加权场景。",
    },
    {
        "amount_cost_scenario": "amount_balanced_base",
        "lgd_rate": 0.50,
        "net_margin_rate": 0.08,
        "manual_review_cost_rate_of_median_credit": 0.0010,
        "manual_bad_catch_rate": 0.60,
        "manual_good_false_decline_rate": 0.06,
        "max_direct_decline_pct": 0.15,
        "max_manual_review_pct": 0.10,
        "max_total_intervention_pct": 0.25,
        "description": "默认金额场景：用 AMT_CREDIT 作为 EAD proxy，LGD 50%，净利差 8%，人审产能 10%。",
    },
    {
        "amount_cost_scenario": "high_lgd_growth",
        "lgd_rate": 0.70,
        "net_margin_rate": 0.10,
        "manual_review_cost_rate_of_median_credit": 0.0010,
        "manual_bad_catch_rate": 0.70,
        "manual_good_false_decline_rate": 0.08,
        "max_direct_decline_pct": 0.20,
        "max_manual_review_pct": 0.15,
        "max_total_intervention_pct": 0.35,
        "description": "高 LGD 且坏账损失更敏感，允许更高风险干预。",
    },
    {
        "amount_cost_scenario": "high_margin_high_false_decline",
        "lgd_rate": 0.50,
        "net_margin_rate": 0.12,
        "manual_review_cost_rate_of_median_credit": 0.0020,
        "manual_bad_catch_rate": 0.50,
        "manual_good_false_decline_rate": 0.05,
        "max_direct_decline_pct": 0.10,
        "max_manual_review_pct": 0.10,
        "max_total_intervention_pct": 0.20,
        "description": "好客户利润更高且人审较贵，误杀成本更重。",
    },
]
SHAP_SPLITS = ["validation", "final_holdout"]
SHAP_TOP_N = 5
SHAP_CHUNK_SIZE = 5000
REASON_SAMPLE_PER_ACTION = 100


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def pd_to_score(pd_values: np.ndarray) -> np.ndarray:
    pd_values = np.clip(pd_values, 1e-6, 1 - 1e-6)
    odds = (1 - pd_values) / pd_values
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return offset + factor * np.log(odds)


def assign_base_strategy(scored: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    validation = scored[scored["screening_split"] == "validation"]
    decline_cutoff = float(np.quantile(validation["pd_lgbm"], 1 - BASE_DECLINE_PCT))
    manual_cutoff = float(np.quantile(validation["pd_lgbm"], 1 - BASE_DECLINE_PCT - BASE_MANUAL_REVIEW_PCT))
    scored = scored.copy()
    scored["strategy_action"] = np.select(
        [
            scored["pd_lgbm"] >= decline_cutoff,
            scored["pd_lgbm"] >= manual_cutoff,
        ],
        ["decline", "manual_review"],
        default="approve",
    )
    return scored, manual_cutoff, decline_cutoff


def reason_label_for_feature(feature: str) -> str:
    name = feature.lower()
    if "ext_source" in name:
        return "外部综合评分/外部数据风险信号"
    if "utilization" in name:
        return "近期信用卡额度使用率偏高"
    if "debt_to_credit" in name or "total_debt_to_credit" in name:
        return "征信负债相对授信额度偏高"
    if "late" in name or "dpd" in name or "past_due" in name:
        return "近期还款延迟或逾期行为"
    if "refusal" in name or "refused" in name:
        return "历史申请被拒记录较多"
    if "annuity_to_income" in name or "credit_to_income" in name or "payment_to_income" in name:
        return "还款压力相对收入偏高"
    if "employment" in name or "days_employed" in name:
        return "就业稳定性相关风险"
    if "days_birth" in name or "age" in name:
        return "年龄段相关风险"
    if "bureau" in name:
        return "外部征信历史风险信号"
    if "installment" in name:
        return "历史分期还款表现风险"
    if "previous" in name:
        return "历史申请与授信表现风险"
    if "credit_card" in name:
        return "信用卡历史使用表现风险"
    if "pos_" in name:
        return "POS/现金贷历史表现风险"
    return "模型识别的申请资料风险信号"


def load_scored_matrix() -> tuple[pd.DataFrame, list[str], lgb.Booster, int]:
    matrix = pd.read_parquet(PROCESSED_DIR / "candidate_feature_matrix.parquet")
    shortlist = pd.read_csv(OUTPUT_TABLE_DIR / "feature_screening_shortlist.csv")
    feature_names = shortlist["candidate_feature"].tolist()
    model_card = json.loads((MODEL_DIR / "lgbm_model_card.json").read_text(encoding="utf-8"))
    calibration_shift = float(model_card["calibration_intercept_shift"])
    best_iteration = int(model_card["best_iteration"])
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))

    x_all = matrix[feature_names].astype("float32")
    raw_score = model.predict(x_all, num_iteration=best_iteration, raw_score=True)
    pd_lgbm = sigmoid(raw_score + calibration_shift)

    amount_columns = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]
    scored = matrix[["SK_ID_CURR", "dataset", "screening_split", "TARGET", *amount_columns]].copy()
    scored["pd_lgbm"] = pd_lgbm
    scored["a_card_score"] = pd_to_score(pd_lgbm)
    scored, manual_cutoff, decline_cutoff = assign_base_strategy(scored)
    scored["base_manual_pd_cutoff"] = manual_cutoff
    scored["base_decline_pd_cutoff"] = decline_cutoff
    scored["base_manual_score_cutoff"] = pd_to_score(np.array([manual_cutoff]))[0]
    scored["base_decline_score_cutoff"] = pd_to_score(np.array([decline_cutoff]))[0]
    return scored, feature_names, model, best_iteration


def build_threshold_curve(scored: pd.DataFrame) -> pd.DataFrame:
    validation = scored[scored["screening_split"] == "validation"]
    labeled = scored[scored["TARGET"].notna()]
    rows = []
    for decline_pct in DECLINE_PCT_GRID:
        pd_cutoff = float(np.quantile(validation["pd_lgbm"], 1 - decline_pct))
        score_cutoff = float(pd_to_score(np.array([pd_cutoff]))[0])
        for split_name, part in labeled.groupby("screening_split"):
            declined = part[part["pd_lgbm"] >= pd_cutoff]
            approved = part[part["pd_lgbm"] < pd_cutoff]
            total_bad = int(part["TARGET"].sum())
            total_good = int(len(part) - total_bad)
            bad_rejected = int(declined["TARGET"].sum())
            good_rejected = int(len(declined) - bad_rejected)
            decline_bad_rate = float(declined["TARGET"].mean()) if len(declined) else np.nan
            approve_bad_rate = float(approved["TARGET"].mean()) if len(approved) else np.nan
            break_even = good_rejected / bad_rejected if bad_rejected > 0 else np.inf
            rows.append(
                {
                    "split": split_name,
                    "decline_pct_target_from_validation": decline_pct,
                    "pd_cutoff_from_validation": pd_cutoff,
                    "score_cutoff_from_validation": score_cutoff,
                    "rows": len(part),
                    "decline_rows": len(declined),
                    "approve_rows": len(approved),
                    "actual_decline_pct": len(declined) / len(part),
                    "bad_count": total_bad,
                    "good_count": total_good,
                    "bad_rejected_count": bad_rejected,
                    "good_rejected_count": good_rejected,
                    "decline_bad_rate": decline_bad_rate,
                    "approve_bad_rate": approve_bad_rate,
                    "decline_bad_capture": bad_rejected / max(total_bad, 1),
                    "good_rejected_capture": good_rejected / max(total_good, 1),
                    "decline_lift": decline_bad_rate / max(float(part["TARGET"].mean()), 1e-9)
                    if len(declined)
                    else np.nan,
                    "break_even_bad_loss_to_good_profit": break_even,
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "decline_pct_target_from_validation"])


def build_profit_curve(threshold_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in threshold_curve.iterrows():
        for loss_multiple in BAD_LOSS_TO_GOOD_PROFIT_GRID:
            value = row["bad_rejected_count"] * loss_multiple - row["good_rejected_count"]
            rows.append(
                {
                    "split": row["split"],
                    "decline_pct_target_from_validation": row["decline_pct_target_from_validation"],
                    "actual_decline_pct": row["actual_decline_pct"],
                    "bad_loss_to_good_profit": loss_multiple,
                    "net_value_vs_approve_all_units": value,
                    "net_value_per_10000_applications": value / row["rows"] * 10000,
                    "break_even_bad_loss_to_good_profit": row["break_even_bad_loss_to_good_profit"],
                    "decline_bad_rate": row["decline_bad_rate"],
                    "approve_bad_rate": row["approve_bad_rate"],
                    "decline_bad_capture": row["decline_bad_capture"],
                    "good_rejected_capture": row["good_rejected_capture"],
                    "bad_rejected_count": row["bad_rejected_count"],
                    "good_rejected_count": row["good_rejected_count"],
                }
            )
    profit_curve = pd.DataFrame(rows)
    summary_rows = []
    for (split_name, loss_multiple), part in profit_curve.groupby(["split", "bad_loss_to_good_profit"]):
        best = part.sort_values("net_value_vs_approve_all_units", ascending=False).iloc[0]
        summary_rows.append(best.to_dict())
    summary = pd.DataFrame(summary_rows).sort_values(["split", "bad_loss_to_good_profit"])
    return profit_curve, summary


def scenario_assumptions_frame() -> pd.DataFrame:
    return pd.DataFrame(COST_SCENARIOS)[
        [
            "cost_scenario",
            "bad_loss_cost",
            "false_decline_good_cost",
            "manual_review_cost",
            "manual_bad_catch_rate",
            "manual_good_false_decline_rate",
            "max_direct_decline_pct",
            "max_manual_review_pct",
            "max_total_intervention_pct",
            "description",
        ]
    ]


def cutoff_for_top_pct(validation_pd: np.ndarray, top_pct: float) -> float:
    if top_pct <= 0:
        return np.inf
    if top_pct >= 1:
        return -np.inf
    return float(np.quantile(validation_pd, 1 - top_pct))


def score_strategy_with_cost(
    split_name: str,
    part: pd.DataFrame,
    cost: dict[str, float | str],
    decline_pct: float,
    manual_review_pct: float,
    manual_cutoff: float,
    decline_cutoff: float,
) -> dict[str, float | int | str]:
    risk = part["pd_lgbm"].to_numpy(dtype=float)
    y = part["TARGET"].to_numpy(dtype=int)

    decline_mask = risk >= decline_cutoff
    manual_mask = (risk >= manual_cutoff) & ~decline_mask
    approve_mask = ~(decline_mask | manual_mask)

    bad_decline = int(y[decline_mask].sum())
    good_decline = int(decline_mask.sum() - bad_decline)
    bad_manual = int(y[manual_mask].sum())
    good_manual = int(manual_mask.sum() - bad_manual)
    bad_approve = int(y[approve_mask].sum())
    good_approve = int(approve_mask.sum() - bad_approve)
    total_bad = int(y.sum())
    total_good = int(len(y) - total_bad)

    bad_loss_cost = float(cost["bad_loss_cost"])
    false_decline_good_cost = float(cost["false_decline_good_cost"])
    manual_review_cost = float(cost["manual_review_cost"])
    manual_bad_catch_rate = float(cost["manual_bad_catch_rate"])
    manual_good_false_decline_rate = float(cost["manual_good_false_decline_rate"])

    expected_bad_saved = bad_decline + manual_bad_catch_rate * bad_manual
    expected_good_lost = good_decline + manual_good_false_decline_rate * good_manual
    manual_review_cost_total = manual_mask.sum() * manual_review_cost
    incremental_value = (
        expected_bad_saved * bad_loss_cost
        - expected_good_lost * false_decline_good_cost
        - manual_review_cost_total
    )

    expected_bad_approved = bad_approve + (1 - manual_bad_catch_rate) * bad_manual
    expected_good_approved = good_approve + (1 - manual_good_false_decline_rate) * good_manual
    expected_approved = expected_bad_approved + expected_good_approved
    expected_rejected = decline_mask.sum() + manual_bad_catch_rate * bad_manual + manual_good_false_decline_rate * good_manual

    return {
        "cost_scenario": str(cost["cost_scenario"]),
        "split": split_name,
        "decline_pct_target_from_validation": decline_pct,
        "manual_review_pct_target_from_validation": manual_review_pct,
        "total_intervention_pct_target": decline_pct + manual_review_pct,
        "pd_manual_cutoff_from_validation": manual_cutoff,
        "pd_decline_cutoff_from_validation": decline_cutoff,
        "score_manual_cutoff_from_validation": float(pd_to_score(np.array([manual_cutoff]))[0])
        if np.isfinite(manual_cutoff)
        else np.nan,
        "score_decline_cutoff_from_validation": float(pd_to_score(np.array([decline_cutoff]))[0])
        if np.isfinite(decline_cutoff)
        else np.nan,
        "rows": len(part),
        "bad_count": total_bad,
        "good_count": total_good,
        "direct_decline_rows": int(decline_mask.sum()),
        "manual_review_rows": int(manual_mask.sum()),
        "auto_approve_rows": int(approve_mask.sum()),
        "actual_direct_decline_pct": float(decline_mask.mean()),
        "actual_manual_review_pct": float(manual_mask.mean()),
        "bad_direct_decline": bad_decline,
        "good_direct_decline": good_decline,
        "bad_manual_review": bad_manual,
        "good_manual_review": good_manual,
        "bad_auto_approve": bad_approve,
        "good_auto_approve": good_approve,
        "direct_decline_bad_rate": bad_decline / max(int(decline_mask.sum()), 1),
        "manual_review_bad_rate": bad_manual / max(int(manual_mask.sum()), 1),
        "auto_approve_bad_rate": bad_approve / max(int(approve_mask.sum()), 1),
        "expected_bad_saved": expected_bad_saved,
        "expected_good_lost": expected_good_lost,
        "expected_bad_saved_capture": expected_bad_saved / max(total_bad, 1),
        "expected_good_lost_capture": expected_good_lost / max(total_good, 1),
        "expected_final_reject_pct": expected_rejected / len(part),
        "expected_final_approve_bad_rate": expected_bad_approved / max(expected_approved, 1e-9),
        "manual_review_cost_total": manual_review_cost_total,
        "incremental_value_vs_approve_all_units": incremental_value,
        "incremental_value_per_10000_applications": incremental_value / len(part) * 10000,
        "bad_loss_cost": bad_loss_cost,
        "false_decline_good_cost": false_decline_good_cost,
        "manual_review_cost": manual_review_cost,
        "manual_bad_catch_rate": manual_bad_catch_rate,
        "manual_good_false_decline_rate": manual_good_false_decline_rate,
        "max_direct_decline_pct": float(cost.get("max_direct_decline_pct", np.nan)),
        "max_manual_review_pct": float(cost.get("max_manual_review_pct", np.nan)),
        "max_total_intervention_pct": float(cost.get("max_total_intervention_pct", np.nan)),
    }


def build_cost_strategy_curve(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_pd = scored[scored["screening_split"] == "validation"]["pd_lgbm"].to_numpy(dtype=float)
    labeled = scored[scored["TARGET"].notna()]
    rows = []
    for cost in COST_SCENARIOS:
        max_direct_decline_pct = float(cost.get("max_direct_decline_pct", max(STRATEGY_DECLINE_PCT_GRID)))
        max_manual_review_pct = float(cost.get("max_manual_review_pct", max(STRATEGY_MANUAL_REVIEW_PCT_GRID)))
        max_total_intervention_pct = float(cost.get("max_total_intervention_pct", MAX_TOTAL_INTERVENTION_PCT))
        for decline_pct in STRATEGY_DECLINE_PCT_GRID:
            for manual_review_pct in STRATEGY_MANUAL_REVIEW_PCT_GRID:
                if decline_pct + manual_review_pct > MAX_TOTAL_INTERVENTION_PCT:
                    continue
                if decline_pct > max_direct_decline_pct:
                    continue
                if manual_review_pct > max_manual_review_pct:
                    continue
                if decline_pct + manual_review_pct > max_total_intervention_pct:
                    continue
                decline_cutoff = cutoff_for_top_pct(validation_pd, decline_pct)
                manual_cutoff = cutoff_for_top_pct(validation_pd, decline_pct + manual_review_pct)
                for split_name, part in labeled.groupby("screening_split"):
                    rows.append(
                        score_strategy_with_cost(
                            split_name=split_name,
                            part=part,
                            cost=cost,
                            decline_pct=decline_pct,
                            manual_review_pct=manual_review_pct,
                            manual_cutoff=manual_cutoff,
                            decline_cutoff=decline_cutoff,
                        )
                    )

    curve = pd.DataFrame(rows).sort_values(
        ["cost_scenario", "split", "decline_pct_target_from_validation", "manual_review_pct_target_from_validation"]
    )
    validation = curve[curve["split"] == "validation"].copy()
    validation["validation_rank"] = validation.groupby("cost_scenario")[
        "incremental_value_vs_approve_all_units"
    ].rank(method="first", ascending=False)
    best_keys = validation[validation["validation_rank"] == 1][
        ["cost_scenario", "decline_pct_target_from_validation", "manual_review_pct_target_from_validation"]
    ]
    optimal = curve.merge(
        best_keys,
        on=["cost_scenario", "decline_pct_target_from_validation", "manual_review_pct_target_from_validation"],
        how="inner",
    ).sort_values(["cost_scenario", "split"])
    return curve, optimal


def amount_cost_assumptions_frame() -> pd.DataFrame:
    return pd.DataFrame(AMOUNT_WEIGHTED_COST_SCENARIOS)[
        [
            "amount_cost_scenario",
            "lgd_rate",
            "net_margin_rate",
            "manual_review_cost_rate_of_median_credit",
            "manual_bad_catch_rate",
            "manual_good_false_decline_rate",
            "max_direct_decline_pct",
            "max_manual_review_pct",
            "max_total_intervention_pct",
            "description",
        ]
    ]


def score_strategy_with_amount_cost(
    split_name: str,
    part: pd.DataFrame,
    cost: dict[str, float | str],
    decline_pct: float,
    manual_review_pct: float,
    manual_cutoff: float,
    decline_cutoff: float,
    median_credit_for_cost: float,
) -> dict[str, float | int | str]:
    risk = part["pd_lgbm"].to_numpy(dtype=float)
    y = part["TARGET"].to_numpy(dtype=int)
    exposure = part["AMT_CREDIT"].fillna(median_credit_for_cost).clip(lower=0).to_numpy(dtype=float)

    decline_mask = risk >= decline_cutoff
    manual_mask = (risk >= manual_cutoff) & ~decline_mask
    approve_mask = ~(decline_mask | manual_mask)
    bad_mask = y == 1
    good_mask = ~bad_mask

    total_exposure = float(exposure.sum())
    total_bad_exposure = float(exposure[bad_mask].sum())
    total_good_exposure = float(exposure[good_mask].sum())
    direct_decline_exposure = float(exposure[decline_mask].sum())
    manual_review_exposure = float(exposure[manual_mask].sum())
    auto_approve_exposure = float(exposure[approve_mask].sum())
    bad_direct_decline_exposure = float(exposure[decline_mask & bad_mask].sum())
    good_direct_decline_exposure = float(exposure[decline_mask & good_mask].sum())
    bad_manual_exposure = float(exposure[manual_mask & bad_mask].sum())
    good_manual_exposure = float(exposure[manual_mask & good_mask].sum())
    bad_auto_approve_exposure = float(exposure[approve_mask & bad_mask].sum())
    good_auto_approve_exposure = float(exposure[approve_mask & good_mask].sum())

    lgd_rate = float(cost["lgd_rate"])
    net_margin_rate = float(cost["net_margin_rate"])
    manual_review_cost_rate = float(cost["manual_review_cost_rate_of_median_credit"])
    manual_bad_catch_rate = float(cost["manual_bad_catch_rate"])
    manual_good_false_decline_rate = float(cost["manual_good_false_decline_rate"])
    manual_review_unit_cost = median_credit_for_cost * manual_review_cost_rate

    expected_bad_exposure_saved = bad_direct_decline_exposure + manual_bad_catch_rate * bad_manual_exposure
    expected_good_exposure_lost = good_direct_decline_exposure + manual_good_false_decline_rate * good_manual_exposure
    expected_bad_exposure_approved = bad_auto_approve_exposure + (1 - manual_bad_catch_rate) * bad_manual_exposure
    expected_good_exposure_approved = good_auto_approve_exposure + (
        1 - manual_good_false_decline_rate
    ) * good_manual_exposure
    expected_approved_exposure = expected_bad_exposure_approved + expected_good_exposure_approved
    expected_rejected_exposure = direct_decline_exposure + manual_bad_catch_rate * bad_manual_exposure + (
        manual_good_false_decline_rate * good_manual_exposure
    )

    expected_loss_saved_amount = expected_bad_exposure_saved * lgd_rate
    expected_margin_lost_amount = expected_good_exposure_lost * net_margin_rate
    manual_review_cost_amount = manual_mask.sum() * manual_review_unit_cost
    incremental_profit_proxy = expected_loss_saved_amount - expected_margin_lost_amount - manual_review_cost_amount

    return {
        "amount_cost_scenario": str(cost["amount_cost_scenario"]),
        "split": split_name,
        "decline_pct_target_from_validation": decline_pct,
        "manual_review_pct_target_from_validation": manual_review_pct,
        "total_intervention_pct_target": decline_pct + manual_review_pct,
        "pd_manual_cutoff_from_validation": manual_cutoff,
        "pd_decline_cutoff_from_validation": decline_cutoff,
        "score_manual_cutoff_from_validation": float(pd_to_score(np.array([manual_cutoff]))[0])
        if np.isfinite(manual_cutoff)
        else np.nan,
        "score_decline_cutoff_from_validation": float(pd_to_score(np.array([decline_cutoff]))[0])
        if np.isfinite(decline_cutoff)
        else np.nan,
        "rows": len(part),
        "direct_decline_rows": int(decline_mask.sum()),
        "manual_review_rows": int(manual_mask.sum()),
        "auto_approve_rows": int(approve_mask.sum()),
        "actual_direct_decline_pct": float(decline_mask.mean()),
        "actual_manual_review_pct": float(manual_mask.mean()),
        "total_credit_exposure": total_exposure,
        "total_bad_credit_exposure": total_bad_exposure,
        "total_good_credit_exposure": total_good_exposure,
        "direct_decline_credit_exposure": direct_decline_exposure,
        "manual_review_credit_exposure": manual_review_exposure,
        "auto_approve_credit_exposure": auto_approve_exposure,
        "bad_direct_decline_credit_exposure": bad_direct_decline_exposure,
        "good_direct_decline_credit_exposure": good_direct_decline_exposure,
        "bad_manual_review_credit_exposure": bad_manual_exposure,
        "good_manual_review_credit_exposure": good_manual_exposure,
        "bad_auto_approve_credit_exposure": bad_auto_approve_exposure,
        "good_auto_approve_credit_exposure": good_auto_approve_exposure,
        "direct_decline_credit_share": direct_decline_exposure / max(total_exposure, 1e-9),
        "manual_review_credit_share": manual_review_exposure / max(total_exposure, 1e-9),
        "direct_decline_exposure_bad_rate": bad_direct_decline_exposure / max(direct_decline_exposure, 1e-9),
        "manual_review_exposure_bad_rate": bad_manual_exposure / max(manual_review_exposure, 1e-9),
        "auto_approve_exposure_bad_rate": bad_auto_approve_exposure / max(auto_approve_exposure, 1e-9),
        "expected_bad_exposure_saved": expected_bad_exposure_saved,
        "expected_good_exposure_lost": expected_good_exposure_lost,
        "expected_bad_exposure_saved_capture": expected_bad_exposure_saved / max(total_bad_exposure, 1e-9),
        "expected_good_exposure_lost_capture": expected_good_exposure_lost / max(total_good_exposure, 1e-9),
        "expected_final_reject_exposure_share": expected_rejected_exposure / max(total_exposure, 1e-9),
        "expected_final_approve_exposure_bad_rate": expected_bad_exposure_approved
        / max(expected_approved_exposure, 1e-9),
        "expected_loss_saved_amount": expected_loss_saved_amount,
        "expected_margin_lost_amount": expected_margin_lost_amount,
        "manual_review_cost_amount": manual_review_cost_amount,
        "incremental_profit_proxy_amount": incremental_profit_proxy,
        "incremental_profit_proxy_per_10000_applications": incremental_profit_proxy / len(part) * 10000,
        "incremental_profit_proxy_per_100m_credit": incremental_profit_proxy / max(total_exposure, 1e-9) * 100_000_000,
        "lgd_rate": lgd_rate,
        "net_margin_rate": net_margin_rate,
        "manual_review_cost_rate_of_median_credit": manual_review_cost_rate,
        "manual_review_unit_cost": manual_review_unit_cost,
        "manual_bad_catch_rate": manual_bad_catch_rate,
        "manual_good_false_decline_rate": manual_good_false_decline_rate,
        "max_direct_decline_pct": float(cost.get("max_direct_decline_pct", np.nan)),
        "max_manual_review_pct": float(cost.get("max_manual_review_pct", np.nan)),
        "max_total_intervention_pct": float(cost.get("max_total_intervention_pct", np.nan)),
    }


def build_amount_weighted_cost_curve(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = scored[scored["screening_split"] == "validation"]
    validation_pd = validation["pd_lgbm"].to_numpy(dtype=float)
    median_credit_for_cost = float(validation["AMT_CREDIT"].median())
    labeled = scored[scored["TARGET"].notna()]
    rows = []
    for cost in AMOUNT_WEIGHTED_COST_SCENARIOS:
        max_direct_decline_pct = float(cost.get("max_direct_decline_pct", max(STRATEGY_DECLINE_PCT_GRID)))
        max_manual_review_pct = float(cost.get("max_manual_review_pct", max(STRATEGY_MANUAL_REVIEW_PCT_GRID)))
        max_total_intervention_pct = float(cost.get("max_total_intervention_pct", MAX_TOTAL_INTERVENTION_PCT))
        for decline_pct in STRATEGY_DECLINE_PCT_GRID:
            for manual_review_pct in STRATEGY_MANUAL_REVIEW_PCT_GRID:
                if decline_pct + manual_review_pct > MAX_TOTAL_INTERVENTION_PCT:
                    continue
                if decline_pct > max_direct_decline_pct:
                    continue
                if manual_review_pct > max_manual_review_pct:
                    continue
                if decline_pct + manual_review_pct > max_total_intervention_pct:
                    continue
                decline_cutoff = cutoff_for_top_pct(validation_pd, decline_pct)
                manual_cutoff = cutoff_for_top_pct(validation_pd, decline_pct + manual_review_pct)
                for split_name, part in labeled.groupby("screening_split"):
                    rows.append(
                        score_strategy_with_amount_cost(
                            split_name=split_name,
                            part=part,
                            cost=cost,
                            decline_pct=decline_pct,
                            manual_review_pct=manual_review_pct,
                            manual_cutoff=manual_cutoff,
                            decline_cutoff=decline_cutoff,
                            median_credit_for_cost=median_credit_for_cost,
                        )
                    )

    curve = pd.DataFrame(rows).sort_values(
        [
            "amount_cost_scenario",
            "split",
            "decline_pct_target_from_validation",
            "manual_review_pct_target_from_validation",
        ]
    )
    validation_curve = curve[curve["split"] == "validation"].copy()
    validation_curve["validation_rank"] = validation_curve.groupby("amount_cost_scenario")[
        "incremental_profit_proxy_amount"
    ].rank(method="first", ascending=False)
    best_keys = validation_curve[validation_curve["validation_rank"] == 1][
        ["amount_cost_scenario", "decline_pct_target_from_validation", "manual_review_pct_target_from_validation"]
    ]
    optimal = curve.merge(
        best_keys,
        on=["amount_cost_scenario", "decline_pct_target_from_validation", "manual_review_pct_target_from_validation"],
        how="inner",
    ).sort_values(["amount_cost_scenario", "split"])
    return curve, optimal


def build_shap_outputs(
    scored: pd.DataFrame,
    feature_names: list[str],
    model: lgb.Booster,
    best_iteration: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_parquet(PROCESSED_DIR / "candidate_feature_matrix.parquet")
    evidence = pd.read_csv(OUTPUT_TABLE_DIR / "feature_selection_evidence_table.csv")
    importance = pd.read_csv(OUTPUT_TABLE_DIR / "lgbm_feature_importance.csv")

    explain_mask = scored["screening_split"].isin(SHAP_SPLITS)
    explain_scored = scored.loc[explain_mask].reset_index(drop=True)
    x_explain = matrix.loc[explain_mask, feature_names].astype("float32").reset_index(drop=True)
    feature_count = len(feature_names)
    sum_abs = np.zeros(feature_count, dtype=np.float64)
    sum_signed = np.zeros(feature_count, dtype=np.float64)
    sum_positive = np.zeros(feature_count, dtype=np.float64)
    positive_count = np.zeros(feature_count, dtype=np.float64)
    reason_rows = []

    for start in range(0, len(x_explain), SHAP_CHUNK_SIZE):
        stop = min(start + SHAP_CHUNK_SIZE, len(x_explain))
        x_chunk = x_explain.iloc[start:stop]
        meta_chunk = explain_scored.iloc[start:stop].reset_index(drop=True)
        contrib = model.predict(x_chunk, num_iteration=best_iteration, pred_contrib=True)
        feature_contrib = np.asarray(contrib[:, :-1], dtype=np.float64)
        sum_abs += np.abs(feature_contrib).sum(axis=0)
        sum_signed += feature_contrib.sum(axis=0)
        positive = np.maximum(feature_contrib, 0)
        sum_positive += positive.sum(axis=0)
        positive_count += (feature_contrib > 0).sum(axis=0)

        review_mask = meta_chunk["strategy_action"].isin(["decline", "manual_review"]).to_numpy()
        if not review_mask.any():
            continue
        review_positions = np.flatnonzero(review_mask)
        review_contrib = feature_contrib[review_positions]
        review_values = x_chunk.iloc[review_positions].to_numpy()
        review_meta = meta_chunk.iloc[review_positions].reset_index(drop=True)
        top_idx = np.argpartition(-review_contrib, kth=min(SHAP_TOP_N, feature_count) - 1, axis=1)[:, :SHAP_TOP_N]
        top_vals = np.take_along_axis(review_contrib, top_idx, axis=1)
        order = np.argsort(-top_vals, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_vals = np.take_along_axis(top_vals, order, axis=1)

        for row_pos in range(len(review_meta)):
            meta = review_meta.iloc[row_pos]
            for rank in range(SHAP_TOP_N):
                feature_idx = int(top_idx[row_pos, rank])
                shap_value = float(top_vals[row_pos, rank])
                if shap_value <= 0:
                    continue
                feature = feature_names[feature_idx]
                reason_rows.append(
                    {
                        "SK_ID_CURR": int(meta["SK_ID_CURR"]),
                        "split": meta["screening_split"],
                        "strategy_action": meta["strategy_action"],
                        "TARGET": meta["TARGET"],
                        "pd_lgbm": float(meta["pd_lgbm"]),
                        "a_card_score": float(meta["a_card_score"]),
                        "reason_rank": rank + 1,
                        "candidate_feature": feature,
                        "reason_label": reason_label_for_feature(feature),
                        "shap_value_raw_log_odds": shap_value,
                        "feature_value": float(review_values[row_pos, feature_idx])
                        if np.isfinite(review_values[row_pos, feature_idx])
                        else np.nan,
                    }
                )

    n = max(len(x_explain), 1)
    shap_global = pd.DataFrame(
        {
            "candidate_feature": feature_names,
            "mean_abs_shap": sum_abs / n,
            "mean_shap": sum_signed / n,
            "mean_positive_shap": sum_positive / n,
            "positive_shap_rate": positive_count / n,
        }
    )
    shap_global = shap_global.merge(
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
    shap_global = shap_global.merge(
        importance[["candidate_feature", "importance_gain", "importance_gain_pct", "importance_split"]],
        on="candidate_feature",
        how="left",
    )
    shap_global["reason_label"] = shap_global["candidate_feature"].map(reason_label_for_feature)
    shap_global = shap_global.sort_values("mean_abs_shap", ascending=False)

    reason_long = pd.DataFrame(reason_rows)
    if reason_long.empty:
        reason_summary = pd.DataFrame()
        reason_sample = pd.DataFrame()
    else:
        action_sizes = (
            explain_scored[explain_scored["strategy_action"].isin(["decline", "manual_review"])]
            .groupby(["screening_split", "strategy_action"])["SK_ID_CURR"]
            .nunique()
            .rename("action_applicant_count")
            .reset_index()
        )
        reason_summary = (
            reason_long.groupby(["split", "strategy_action", "candidate_feature", "reason_label"])
            .agg(
                applicant_count=("SK_ID_CURR", "nunique"),
                avg_reason_rank=("reason_rank", "mean"),
                avg_shap_value=("shap_value_raw_log_odds", "mean"),
                avg_pd=("pd_lgbm", "mean"),
                bad_rate=("TARGET", "mean"),
                avg_feature_value=("feature_value", "mean"),
            )
            .reset_index()
            .merge(
                action_sizes,
                left_on=["split", "strategy_action"],
                right_on=["screening_split", "strategy_action"],
                how="left",
            )
            .drop(columns=["screening_split"])
        )
        reason_summary["applicant_share_in_action"] = (
            reason_summary["applicant_count"] / reason_summary["action_applicant_count"].clip(lower=1)
        )
        reason_summary = reason_summary.sort_values(
            ["split", "strategy_action", "applicant_count", "avg_shap_value"],
            ascending=[True, True, False, False],
        )

        sample_base = (
            reason_long[reason_long["reason_rank"] == 1]
            .sort_values(["split", "strategy_action", "pd_lgbm"], ascending=[True, True, False])
            .groupby(["split", "strategy_action"], group_keys=False)
            .head(REASON_SAMPLE_PER_ACTION)
        )
        sample_keys = sample_base[["SK_ID_CURR", "split", "strategy_action"]]
        reason_sample = reason_long.merge(sample_keys, on=["SK_ID_CURR", "split", "strategy_action"], how="inner")
        reason_sample = reason_sample.sort_values(["split", "strategy_action", "pd_lgbm", "SK_ID_CURR", "reason_rank"])

    return shap_global, reason_long, reason_summary, reason_sample


def write_markdown(
    threshold_curve: pd.DataFrame,
    profit_summary: pd.DataFrame,
    cost_assumptions: pd.DataFrame,
    cost_optimal: pd.DataFrame,
    amount_cost_assumptions: pd.DataFrame,
    amount_cost_optimal: pd.DataFrame,
    shap_global: pd.DataFrame,
    reason_summary: pd.DataFrame,
) -> None:
    holdout_10 = threshold_curve[
        (threshold_curve["split"] == "final_holdout")
        & np.isclose(threshold_curve["decline_pct_target_from_validation"], BASE_DECLINE_PCT)
    ].iloc[0]
    validation_10 = threshold_curve[
        (threshold_curve["split"] == "validation")
        & np.isclose(threshold_curve["decline_pct_target_from_validation"], BASE_DECLINE_PCT)
    ].iloc[0]
    holdout_profit = profit_summary[profit_summary["split"] == "final_holdout"]
    cost_optimal_validation = cost_optimal[cost_optimal["split"] == "validation"]
    cost_optimal_holdout = cost_optimal[cost_optimal["split"] == "final_holdout"]
    balanced_validation = cost_optimal_validation[cost_optimal_validation["cost_scenario"] == "balanced_base"].iloc[0]
    balanced_holdout = cost_optimal_holdout[cost_optimal_holdout["cost_scenario"] == "balanced_base"].iloc[0]
    amount_optimal_validation = amount_cost_optimal[amount_cost_optimal["split"] == "validation"]
    amount_optimal_holdout = amount_cost_optimal[amount_cost_optimal["split"] == "final_holdout"]
    amount_balanced_validation = amount_optimal_validation[
        amount_optimal_validation["amount_cost_scenario"] == "amount_balanced_base"
    ].iloc[0]
    amount_balanced_holdout = amount_optimal_holdout[
        amount_optimal_holdout["amount_cost_scenario"] == "amount_balanced_base"
    ].iloc[0]
    top_shap = shap_global.head(15)
    top_holdout_decline_reasons = reason_summary[
        (reason_summary["split"] == "final_holdout") & (reason_summary["strategy_action"] == "decline")
    ].head(12)

    md = [
        "# 策略阈值、收益曲线与 SHAP 原因码",
        "",
        "本文件由 `src/08_explain_monitor.py` 自动生成。",
        "",
        "## 口径",
        "",
        "- 阈值从 `validation` 分布确定，`final_holdout` 只用于只读检验。",
        "- 收益曲线使用相对经济口径：拦住 1 个坏客户的价值 = `bad_loss_to_good_profit`，误拒 1 个好客户的机会成本 = 1。",
        "- 因为 Home Credit 数据没有真实利率、额度、LGD、资金成本，本页不声称绝对利润，只给 break-even 和相对收益。",
        "- SHAP 使用 LightGBM `pred_contrib=True`，解释的是模型 raw log-odds 风险贡献；正值表示推高预测违约风险。",
        "",
        "## 10% 拒绝策略是否划算",
        "",
        "| split | actual_decline_pct | decline_bad_rate | approve_bad_rate | bad_rejected | good_rejected | bad_capture | break_even_bad_loss/good_profit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| validation | {validation_10['actual_decline_pct']:.2%} | {validation_10['decline_bad_rate']:.2%} | "
            f"{validation_10['approve_bad_rate']:.2%} | {int(validation_10['bad_rejected_count'])} | "
            f"{int(validation_10['good_rejected_count'])} | {validation_10['decline_bad_capture']:.2%} | "
            f"{validation_10['break_even_bad_loss_to_good_profit']:.2f} |"
        ),
        (
            f"| final_holdout | {holdout_10['actual_decline_pct']:.2%} | {holdout_10['decline_bad_rate']:.2%} | "
            f"{holdout_10['approve_bad_rate']:.2%} | {int(holdout_10['bad_rejected_count'])} | "
            f"{int(holdout_10['good_rejected_count'])} | {holdout_10['decline_bad_capture']:.2%} | "
            f"{holdout_10['break_even_bad_loss_to_good_profit']:.2f} |"
        ),
        "",
        (
            f"- 在 final_holdout 上，拒绝高风险约 10% 客群会拦住 {int(holdout_10['bad_rejected_count'])} 个坏客户，"
            f"同时误拒 {int(holdout_10['good_rejected_count'])} 个好客户。"
        ),
        (
            f"- break-even 为 {holdout_10['break_even_bad_loss_to_good_profit']:.2f}：只要一个坏客户带来的净损失超过"
            f"一个好客户净收益的 {holdout_10['break_even_bad_loss_to_good_profit']:.2f} 倍，这条 10% 拒绝线在经济上就是正收益。"
        ),
        "",
        "## 不同坏账损失倍数下的最优拒绝率",
        "",
        "| split | bad_loss/good_profit | best_decline_pct | approve_bad_rate | decline_bad_rate | net_value_per_10k_apps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in holdout_profit.iterrows():
        md.append(
            f"| final_holdout | {row['bad_loss_to_good_profit']:.1f} | "
            f"{row['decline_pct_target_from_validation']:.0%} | {row['approve_bad_rate']:.2%} | "
            f"{row['decline_bad_rate']:.2%} | {row['net_value_per_10000_applications']:.1f} |"
        )

    md.extend(
        [
            "",
            "## 三段式成本收益模型",
            "",
            "这里把阈值选择改成 approve / manual_review / decline 的统一成本函数，所有数值都用相对单位表示：",
            "",
            "- 漏放坏客户成本 `bad_loss_cost`：坏客户如果被放款造成的预期净损失。",
            "- 误杀好客户成本 `false_decline_good_cost`：好客户被拒造成的利润、获客和关系损失。",
            "- 人工审核成本 `manual_review_cost`：每进入人工审核一单的运营成本。",
            "- 人审坏客户拦截率 `manual_bad_catch_rate`：人工审核能识别并拒绝多少复核区坏客户。",
            "- 人审好客户误杀率 `manual_good_false_decline_rate`：人工审核误拒多少复核区好客户。",
            "- 产能约束：限制最大直接拒绝比例、最大人工审核比例和总干预比例，避免成本函数给出运营上不可落地的策略。",
            "",
            "增量收益相对 `approve-all` 计算：",
            "",
            "`bad_saved * bad_loss_cost - good_lost * false_decline_good_cost - manual_review_rows * manual_review_cost`",
            "",
            "## 成本参数场景",
            "",
            "| scenario | bad_loss | false_decline | manual_cost | manual_bad_catch | manual_good_false_decline | max_decline | max_manual |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in cost_assumptions.iterrows():
        md.append(
            f"| {row['cost_scenario']} | {row['bad_loss_cost']:.2f} | "
            f"{row['false_decline_good_cost']:.2f} | {row['manual_review_cost']:.2f} | "
            f"{row['manual_bad_catch_rate']:.0%} | {row['manual_good_false_decline_rate']:.0%} | "
            f"{row['max_direct_decline_pct']:.0%} | {row['max_manual_review_pct']:.0%} |"
        )

    md.extend(
        [
            "",
            "## 按成本收益选择阈值",
            "",
            (
                f"- 默认 `balanced_base` 场景在 validation 上选择：直接拒绝 "
                f"{balanced_validation['decline_pct_target_from_validation']:.0%}，人工审核 "
                f"{balanced_validation['manual_review_pct_target_from_validation']:.0%}。"
            ),
            (
                f"- 同一组阈值在 final_holdout 上的期望最终通过池坏账率为 "
                f"{balanced_holdout['expected_final_approve_bad_rate']:.2%}，"
                f"期望拦截坏客户覆盖率为 {balanced_holdout['expected_bad_saved_capture']:.2%}，"
                f"每万申请增量收益为 {balanced_holdout['incremental_value_per_10000_applications']:.1f} 个成本单位。"
            ),
            "",
            "| scenario | selected_decline | selected_manual_review | holdout_approve_bad_rate | bad_saved_capture | final_reject_pct | value_per_10k_apps |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in cost_optimal_holdout.iterrows():
        md.append(
            f"| {row['cost_scenario']} | {row['decline_pct_target_from_validation']:.0%} | "
            f"{row['manual_review_pct_target_from_validation']:.0%} | "
            f"{row['expected_final_approve_bad_rate']:.2%} | {row['expected_bad_saved_capture']:.2%} | "
            f"{row['expected_final_reject_pct']:.2%} | {row['incremental_value_per_10000_applications']:.1f} |"
        )

    md.extend(
        [
            "",
            "## 金额加权成本模型",
            "",
            "当前数据有 `AMT_CREDIT`，因此可以把它作为 EAD proxy 做金额加权策略评估。这个口径仍不是绝对利润，因为数据没有真实利率、资金成本、回收金额、催收成本和客户 LTV。",
            "",
            "- 漏放坏客户损失 proxy：`AMT_CREDIT * LGD`。",
            "- 误杀好客户机会成本 proxy：`AMT_CREDIT * net_margin_rate`。",
            "- 人工审核成本 proxy：`validation AMT_CREDIT 中位数 * manual_review_cost_rate_of_median_credit`。",
            "- 金额加权阈值同样只在 `validation` 上选择，`final_holdout` 只做验证。",
            "",
            "## 金额成本参数场景",
            "",
            "| scenario | LGD | net_margin | manual_cost_rate | manual_bad_catch | manual_good_false_decline | max_decline | max_manual |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in amount_cost_assumptions.iterrows():
        md.append(
            f"| {row['amount_cost_scenario']} | {row['lgd_rate']:.0%} | {row['net_margin_rate']:.0%} | "
            f"{row['manual_review_cost_rate_of_median_credit']:.2%} | {row['manual_bad_catch_rate']:.0%} | "
            f"{row['manual_good_false_decline_rate']:.0%} | {row['max_direct_decline_pct']:.0%} | "
            f"{row['max_manual_review_pct']:.0%} |"
        )

    md.extend(
        [
            "",
            "## 按金额加权成本选择阈值",
            "",
            (
                f"- 默认 `amount_balanced_base` 场景在 validation 上选择：直接拒绝 "
                f"{amount_balanced_validation['decline_pct_target_from_validation']:.0%}，人工审核 "
                f"{amount_balanced_validation['manual_review_pct_target_from_validation']:.0%}。"
            ),
            (
                f"- 同一组阈值在 final_holdout 上的期望通过敞口坏账率为 "
                f"{amount_balanced_holdout['expected_final_approve_exposure_bad_rate']:.2%}，"
                f"期望拦截坏客户敞口覆盖率为 {amount_balanced_holdout['expected_bad_exposure_saved_capture']:.2%}，"
                f"每 1 亿授信敞口增量利润 proxy 为 "
                f"{amount_balanced_holdout['incremental_profit_proxy_per_100m_credit']:,.0f}。"
            ),
            "",
            "| scenario | selected_decline | selected_manual_review | holdout_approve_exposure_bad_rate | bad_exposure_saved | reject_exposure_share | profit_proxy_per_100m_credit |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in amount_optimal_holdout.iterrows():
        md.append(
            f"| {row['amount_cost_scenario']} | {row['decline_pct_target_from_validation']:.0%} | "
            f"{row['manual_review_pct_target_from_validation']:.0%} | "
            f"{row['expected_final_approve_exposure_bad_rate']:.2%} | "
            f"{row['expected_bad_exposure_saved_capture']:.2%} | "
            f"{row['expected_final_reject_exposure_share']:.2%} | "
            f"{row['incremental_profit_proxy_per_100m_credit']:,.0f} |"
        )

    md.extend(
        [
            "",
            "## SHAP 全局 Top 特征",
            "",
            "| feature | reason_label | group | mean_abs_shap | gain_pct |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in top_shap.iterrows():
        md.append(
            f"| `{row['candidate_feature']}` | {row['reason_label']} | {row['feature_group']} | "
            f"{row['mean_abs_shap']:.5f} | {row['importance_gain_pct']:.2%} |"
        )

    md.extend(
        [
            "",
            "## final_holdout 拒绝客群主要原因码",
            "",
            "| reason_label | feature | applicant_share | bad_rate | avg_shap |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in top_holdout_decline_reasons.iterrows():
        md.append(
            f"| {row['reason_label']} | `{row['candidate_feature']}` | "
            f"{row['applicant_share_in_action']:.2%} | {row['bad_rate']:.2%} | {row['avg_shap_value']:.4f} |"
        )

    md.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `outputs/tables/strategy_threshold_curve.csv`：不同拒绝率的坏账率、捕获率、误拒率、break-even。",
            "- `outputs/tables/strategy_profit_curve.csv`：不同坏账损失倍数下的相对收益曲线。",
            "- `outputs/tables/strategy_profit_curve_summary.csv`：每个损失倍数下的最优拒绝率。",
            "- `outputs/tables/strategy_cost_assumptions.csv`：人工审核、漏放、误杀成本参数场景。",
            "- `outputs/tables/strategy_cost_sensitivity_curve.csv`：三段式策略成本收益全网格。",
            "- `outputs/tables/strategy_cost_optimal_thresholds.csv`：validation 选阈值后在各 split 的表现。",
            "- `outputs/tables/strategy_amount_cost_assumptions.csv`：LGD、净利差、人审成本率参数场景。",
            "- `outputs/tables/strategy_amount_weighted_cost_curve.csv`：金额加权成本收益全网格。",
            "- `outputs/tables/strategy_amount_weighted_optimal_thresholds.csv`：金额加权 validation 选阈值后在各 split 的表现。",
            "- `outputs/tables/shap_global_importance.csv`：SHAP 全局重要性。",
            "- `outputs/tables/shap_reason_code_summary.csv`：decline/manual_review 客群原因码汇总。",
            "- `outputs/tables/shap_reason_code_long.csv`：decline/manual_review 样本 Top 原因长表。",
            "- `outputs/tables/a_card_reason_code_sample.csv`：面试展示用原因码样例。",
            "",
        ]
    )
    (DOCS_DIR / "strategy_profit_shap_initial.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    scored, feature_names, model, best_iteration = load_scored_matrix()
    threshold_curve = build_threshold_curve(scored)
    profit_curve, profit_summary = build_profit_curve(threshold_curve)
    cost_assumptions = scenario_assumptions_frame()
    cost_curve, cost_optimal = build_cost_strategy_curve(scored)
    amount_cost_assumptions = amount_cost_assumptions_frame()
    amount_cost_curve, amount_cost_optimal = build_amount_weighted_cost_curve(scored)
    shap_global, reason_long, reason_summary, reason_sample = build_shap_outputs(
        scored=scored,
        feature_names=feature_names,
        model=model,
        best_iteration=best_iteration,
    )

    threshold_curve.to_csv(OUTPUT_TABLE_DIR / "strategy_threshold_curve.csv", index=False)
    profit_curve.to_csv(OUTPUT_TABLE_DIR / "strategy_profit_curve.csv", index=False)
    profit_summary.to_csv(OUTPUT_TABLE_DIR / "strategy_profit_curve_summary.csv", index=False)
    cost_assumptions.to_csv(OUTPUT_TABLE_DIR / "strategy_cost_assumptions.csv", index=False)
    cost_curve.to_csv(OUTPUT_TABLE_DIR / "strategy_cost_sensitivity_curve.csv", index=False)
    cost_optimal.to_csv(OUTPUT_TABLE_DIR / "strategy_cost_optimal_thresholds.csv", index=False)
    amount_cost_assumptions.to_csv(OUTPUT_TABLE_DIR / "strategy_amount_cost_assumptions.csv", index=False)
    amount_cost_curve.to_csv(OUTPUT_TABLE_DIR / "strategy_amount_weighted_cost_curve.csv", index=False)
    amount_cost_optimal.to_csv(OUTPUT_TABLE_DIR / "strategy_amount_weighted_optimal_thresholds.csv", index=False)
    shap_global.to_csv(OUTPUT_TABLE_DIR / "shap_global_importance.csv", index=False)
    reason_long.to_csv(OUTPUT_TABLE_DIR / "shap_reason_code_long.csv", index=False)
    reason_summary.to_csv(OUTPUT_TABLE_DIR / "shap_reason_code_summary.csv", index=False)
    reason_sample.to_csv(OUTPUT_TABLE_DIR / "a_card_reason_code_sample.csv", index=False)
    write_markdown(
        threshold_curve,
        profit_summary,
        cost_assumptions,
        cost_optimal,
        amount_cost_assumptions,
        amount_cost_optimal,
        shap_global,
        reason_summary,
    )

    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_threshold_curve.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_profit_curve.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_profit_curve_summary.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_cost_assumptions.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_cost_sensitivity_curve.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_cost_optimal_thresholds.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_amount_cost_assumptions.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_amount_weighted_cost_curve.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'strategy_amount_weighted_optimal_thresholds.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'shap_global_importance.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'shap_reason_code_summary.csv'}")
    print(f"Wrote {OUTPUT_TABLE_DIR / 'a_card_reason_code_sample.csv'}")
    print(f"Wrote {DOCS_DIR / 'strategy_profit_shap_initial.md'}")


if __name__ == "__main__":
    main()
