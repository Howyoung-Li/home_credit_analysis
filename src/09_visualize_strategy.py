from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

CAPACITY_COLUMNS = [
    "SK_ID_CURR",
    "screening_split",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "credit_to_income_ratio",
    "annuity_to_income_ratio",
    "income_per_family_member",
    "employment_years",
    "bureau_total_debt_to_credit_ratio",
    "bureau_active_total_debt_to_credit_ratio",
    "bureau_recent_24m_total_debt_to_credit_ratio",
    "bureau_sum_debt",
    "bureau_recent_24m_debt_sum",
    "credit_card_recent_6m_utilization_max",
    "credit_card_recent_12m_utilization_max",
    "installment_recent_12m_late_ratio",
    "installment_recent_12m_total_payment_ratio",
    "installment_recent_12m_shortfall_ratio",
    "previous_refusal_rate",
    "previous_recent_12m_refusal_rate",
    "ext_source_mean",
    "has_bureau_loan_history",
    "has_installment_record_history",
    "has_credit_card_month_history",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
]

PALETTE = {
    "ink": "#1f2937",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "bg": "#f8fafc",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "amber": "#d97706",
    "red": "#dc2626",
    "violet": "#7c3aed",
    "slate": "#475569",
}

ACTION_LABELS = {
    "approve": "自动通过",
    "manual_review": "人工复核",
    "decline": "直接拒绝",
}


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}%}"


def num(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return ""
    return f"{value:,.{digits}f}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, content: object, size: int = 14, fill: str = "#1f2937", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif" '
        f'text-anchor="{anchor}">{esc(content)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, rx: float = 0, stroke: str = "none") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" />'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#1f2937", width: float = 1.5) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}" />'


def svg_wrap(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            rect(0, 0, width, height, "#ffffff"),
            *body,
            "</svg>",
        ]
    )


def write_svg(name: str, width: int, height: int, body: list[str]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    path.write_text(svg_wrap(width, height, body), encoding="utf-8")
    return path


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.5
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
    # blue -> amber -> red, kept readable on white.
    if t < 0.5:
        u = t / 0.5
        a = np.array([219, 234, 254])
        b = np.array([254, 243, 199])
    else:
        u = (t - 0.5) / 0.5
        a = np.array([254, 243, 199])
        b = np.array([248, 113, 113])
    rgb = (a * (1 - u) + b * u).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y_ranked = y_true[order]
    positives = y_ranked.sum()
    if positives == 0:
        return np.nan
    tp = np.cumsum(y_ranked)
    precision_at_k = tp / np.arange(1, len(y_ranked) + 1)
    return float((precision_at_k * y_ranked).sum() / positives)


def build_topk_precision_recall(split: str = "final_holdout") -> pd.DataFrame:
    predictions = pd.read_csv(TABLE_DIR / "lgbm_predictions.csv")
    predictions = predictions[predictions["screening_split"] == split].copy()
    predictions = predictions.sort_values("pred_risk", ascending=False)
    total_bad = int(predictions["TARGET"].sum())
    overall_bad_rate = float(predictions["TARGET"].mean())
    rows = []
    for topk in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]:
        n = int(np.ceil(len(predictions) * topk))
        selected = predictions.head(n)
        bad_count = int(selected["TARGET"].sum())
        bad_rate = float(selected["TARGET"].mean())
        bad_capture = bad_count / total_bad if total_bad else np.nan
        rows.append(
            {
                "split": split,
                "topk_pct": topk,
                "rows": n,
                "bad_count": bad_count,
                "bad_rate_precision": bad_rate,
                "bad_capture_recall": bad_capture,
                "lift": bad_rate / overall_bad_rate if overall_bad_rate else np.nan,
            }
        )
    topk_df = pd.DataFrame(rows)
    topk_df.to_csv(TABLE_DIR / "lgbm_topk_precision_recall.csv", index=False)
    return topk_df


def level_score(value: float, medium: float, high: float, reverse: bool = False) -> int:
    if pd.isna(value):
        return 0
    if reverse:
        if value <= high:
            return 2
        if value <= medium:
            return 1
        return 0
    if value >= high:
        return 2
    if value >= medium:
        return 1
    return 0


def level_label(score: int) -> str:
    return ["低", "中", "高"][int(np.clip(score, 0, 2))]


def format_ratio_value(value: float, digits: int = 1) -> str:
    return "缺失" if pd.isna(value) else pct(value, digits)


def format_amount(value: float) -> str:
    if pd.isna(value):
        return "缺失"
    if abs(value) >= 10_000:
        return f"{value / 10_000:,.1f}万"
    return f"{value:,.0f}"


def add_capacity_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["capacity_repayment_pressure_score"] = [
        max(level_score(a, 0.25, 0.35), level_score(c, 4.0, 6.0))
        for a, c in zip(result["annuity_to_income_ratio"], result["credit_to_income_ratio"])
    ]
    result["capacity_external_debt_score"] = [
        max(level_score(a, 0.45, 0.75), level_score(b, 0.45, 0.75), level_score(c, 0.45, 0.75))
        for a, b, c in zip(
            result["bureau_total_debt_to_credit_ratio"],
            result["bureau_active_total_debt_to_credit_ratio"],
            result["bureau_recent_24m_total_debt_to_credit_ratio"],
        )
    ]
    result["capacity_credit_card_score"] = [
        max(level_score(a, 0.75, 0.95), level_score(b, 0.75, 0.95))
        for a, b in zip(result["credit_card_recent_6m_utilization_max"], result["credit_card_recent_12m_utilization_max"])
    ]
    result["capacity_repayment_history_score"] = [
        max(level_score(l, 0.08, 0.18), level_score(p, 0.98, 0.92, reverse=True), level_score(s, 0.08, 0.15))
        for l, p, s in zip(
            result["installment_recent_12m_late_ratio"],
            result["installment_recent_12m_total_payment_ratio"],
            result["installment_recent_12m_shortfall_ratio"],
        )
    ]
    result["capacity_application_history_score"] = [
        max(level_score(a, 0.20, 0.50), level_score(b, 0.20, 0.50))
        for a, b in zip(result["previous_refusal_rate"], result["previous_recent_12m_refusal_rate"])
    ]
    result["capacity_external_score_score"] = [level_score(v, 0.50, 0.35, reverse=True) for v in result["ext_source_mean"]]
    score_cols = [c for c in result.columns if c.startswith("capacity_") and c.endswith("_score")]
    result["capacity_max_score"] = result[score_cols].max(axis=1)
    result["capacity_high_dimension_count"] = (result[score_cols] >= 2).sum(axis=1)
    result["capacity_medium_plus_dimension_count"] = (result[score_cols] >= 1).sum(axis=1)
    result["capacity_level"] = result["capacity_max_score"].map(level_label)
    return result


def build_risk_dimension_text(row: pd.Series) -> str:
    dimensions = [
        (
            "还款压力",
            row["capacity_repayment_pressure_score"],
            f"年金/收入 {format_ratio_value(row['annuity_to_income_ratio'])}，授信/收入 {row['credit_to_income_ratio']:.2f}x"
            if not pd.isna(row["credit_to_income_ratio"])
            else f"年金/收入 {format_ratio_value(row['annuity_to_income_ratio'])}，授信/收入 缺失",
        ),
        (
            "外部征信负债",
            row["capacity_external_debt_score"],
            f"总债务/授信 {format_ratio_value(row['bureau_total_debt_to_credit_ratio'])}，近24月债务/授信 {format_ratio_value(row['bureau_recent_24m_total_debt_to_credit_ratio'])}",
        ),
        (
            "信用卡使用",
            row["capacity_credit_card_score"],
            f"近6月最高利用率 {format_ratio_value(row['credit_card_recent_6m_utilization_max'])}",
        ),
        (
            "历史还款纪律",
            row["capacity_repayment_history_score"],
            f"近12月逾期占比 {format_ratio_value(row['installment_recent_12m_late_ratio'])}，付款覆盖率 {format_ratio_value(row['installment_recent_12m_total_payment_ratio'])}",
        ),
        (
            "历史申请",
            row["capacity_application_history_score"],
            f"历史拒绝率 {format_ratio_value(row['previous_refusal_rate'])}，近12月拒绝率 {format_ratio_value(row['previous_recent_12m_refusal_rate'])}",
        ),
        (
            "外部评分",
            row["capacity_external_score_score"],
            f"外部综合评分 {row['ext_source_mean']:.3f}" if not pd.isna(row["ext_source_mean"]) else "外部综合评分 缺失",
        ),
    ]
    dimensions = sorted(dimensions, key=lambda item: item[1], reverse=True)
    return "；".join(f"{name}{level_label(score)}：{detail}" for name, score, detail in dimensions[:4])


def build_repayment_capacity_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "candidate_feature_matrix.parquet", columns=CAPACITY_COLUMNS)
    scored = pd.read_csv(TABLE_DIR / "a_card_scored_population.csv")
    scored = scored[scored["screening_split"] == "final_holdout"].copy()
    base = scored.merge(matrix, on=["SK_ID_CURR", "screening_split", "TARGET"], how="left")
    base = add_capacity_dimensions(base)

    base["bureau_debt_ratio_proxy"] = base[
        [
            "bureau_total_debt_to_credit_ratio",
            "bureau_active_total_debt_to_credit_ratio",
            "bureau_recent_24m_total_debt_to_credit_ratio",
        ]
    ].max(axis=1)
    base["credit_card_utilization_proxy"] = base[
        ["credit_card_recent_6m_utilization_max", "credit_card_recent_12m_utilization_max"]
    ].max(axis=1)
    base["previous_refusal_proxy"] = base[["previous_refusal_rate", "previous_recent_12m_refusal_rate"]].max(axis=1)
    base["external_score_risk_proxy"] = 1 - base["ext_source_mean"]

    action_order = ["approve", "manual_review", "decline"]
    summary_rows = []
    for action in action_order:
        part = base[base["strategy_action"] == action]
        summary_rows.append(
            {
                "strategy_action": action,
                "strategy_action_label": ACTION_LABELS[action],
                "rows": int(len(part)),
                "bad_rate": float(part["TARGET"].mean()),
                "avg_pd": float(part["pd_lgbm"].mean()),
                "avg_score": float(part["a_card_score"].mean()),
                "annuity_to_income_mean": float(part["annuity_to_income_ratio"].mean()),
                "credit_to_income_mean": float(part["credit_to_income_ratio"].mean()),
                "bureau_debt_ratio_proxy_mean": float(part["bureau_debt_ratio_proxy"].mean()),
                "credit_card_utilization_proxy_mean": float(part["credit_card_utilization_proxy"].mean()),
                "installment_recent_12m_late_ratio_mean": float(part["installment_recent_12m_late_ratio"].mean()),
                "previous_refusal_proxy_mean": float(part["previous_refusal_proxy"].mean()),
                "external_score_risk_proxy_mean": float(part["external_score_risk_proxy"].mean()),
                "high_capacity_dimension_share": float((part["capacity_high_dimension_count"] >= 2).mean()),
                "medium_plus_capacity_dimension_share": float((part["capacity_medium_plus_dimension_count"] >= 1).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "risk_agent_capacity_segment_summary.csv", index=False)

    case_frames = []
    for action in action_order:
        part = base[base["strategy_action"] == action].copy()
        if action == "decline":
            chosen = part.sort_values("pd_lgbm", ascending=False).head(1)
            case_type = "高风险拒绝样例"
        elif action == "manual_review":
            median_pd = part["pd_lgbm"].median()
            chosen = part.iloc[(part["pd_lgbm"] - median_pd).abs().argsort()[:1]]
            case_type = "边界人工复核样例"
        else:
            chosen = part.sort_values("pd_lgbm", ascending=True).head(1)
            case_type = "低风险通过样例"
        chosen = chosen.copy()
        chosen["case_type"] = case_type
        case_frames.append(chosen)
    cases = pd.concat(case_frames, ignore_index=True)

    reason_long = pd.read_csv(TABLE_DIR / "shap_reason_code_long.csv")
    case_rows = []
    for row in cases.itertuples(index=False):
        case = pd.Series(row._asdict())
        reasons = reason_long[reason_long["SK_ID_CURR"] == case["SK_ID_CURR"]].sort_values("reason_rank").head(3)
        reason_text = "；".join(
            f"{int(r.reason_rank)}.{r.reason_label}({r.candidate_feature}={r.feature_value:.3g})"
            for r in reasons.itertuples(index=False)
        )
        if not reason_text:
            reason_text = "低风险样例未触发主要高风险原因码；以模型低PD、较高A卡分和画像指标作为自动通过依据。"
        risk_dimensions = build_risk_dimension_text(case)
        action = case["strategy_action"]
        if action == "decline":
            recommendation = "建议直接拒绝；若业务需要保留，可转强验证并要求补充收入、负债与历史征信材料。"
            checklist = "核验收入真实性、外部负债压力、历史拒绝原因、近期还款异常；复核是否存在资料缺失或异常值。"
        elif action == "manual_review":
            recommendation = "建议进入人工复核；模型分处于边界区间，需要结合偿债能力、征信负债和历史还款纪律确认。"
            checklist = "补充核验收入稳定性、最近12个月还款覆盖率、信用卡额度使用和历史申请拒绝原因。"
        else:
            recommendation = "建议自动通过；保持额度约束和贷后监控，关注后续征信负债与还款表现变化。"
            checklist = "常规准入校验即可；保留模型原因码和策略版本，进入贷后表现监控。"
        case_rows.append(
            {
                "case_type": case["case_type"],
                "SK_ID_CURR": int(case["SK_ID_CURR"]),
                "strategy_action": action,
                "strategy_action_label": ACTION_LABELS[action],
                "pd_lgbm": float(case["pd_lgbm"]),
                "a_card_score": float(case["a_card_score"]),
                "target_in_holdout": int(case["TARGET"]),
                "income": float(case["AMT_INCOME_TOTAL"]),
                "credit": float(case["AMT_CREDIT"]),
                "annuity": float(case["AMT_ANNUITY"]),
                "annuity_to_income_ratio": float(case["annuity_to_income_ratio"]),
                "credit_to_income_ratio": float(case["credit_to_income_ratio"]),
                "capacity_level": case["capacity_level"],
                "capacity_high_dimension_count": int(case["capacity_high_dimension_count"]),
                "risk_dimensions": risk_dimensions,
                "top_reasons": reason_text,
                "agent_recommendation": recommendation,
                "review_checklist": checklist,
            }
        )
    case_table = pd.DataFrame(case_rows)
    case_table.to_csv(TABLE_DIR / "risk_agent_case_studies.csv", index=False)
    return summary, case_table


def horizontal_bar_chart(
    path_name: str,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    value_formatter,
    color: str,
    width: int = 1120,
    left: int = 360,
) -> Path:
    row_h = 34
    top = 96
    right = 150
    height = top + row_h * len(labels) + 48
    chart_w = width - left - right
    vmax = max(values) if values else 1
    vmax = vmax if vmax > 0 else 1
    body = [
        text(32, 38, title, 24, PALETTE["ink"]),
        text(32, 66, subtitle, 13, PALETTE["muted"]),
        line(left, top - 20, width - right + 10, top - 20, PALETTE["grid"], 1),
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_h
        bar_w = chart_w * value / vmax
        body.append(text(32, y + 18, label[:44], 13, PALETTE["ink"]))
        body.append(rect(left, y, chart_w, 18, "#f1f5f9", 4))
        body.append(rect(left, y, bar_w, 18, color, 4))
        body.append(text(left + bar_w + 10, y + 14, value_formatter(value), 12, PALETTE["ink"]))
    return write_svg(path_name, width, height, body)


def draw_metric_cards() -> Path:
    metrics = pd.read_csv(TABLE_DIR / "lgbm_metrics.csv")
    metrics = metrics[metrics["split"].isin(["validation", "final_holdout"])].copy()
    rows = []
    for _, row in metrics.iterrows():
        rows.extend(
            [
                (row["split"], "AUC", row["auc"], "{:.4f}", PALETTE["blue"]),
                (row["split"], "KS", row["ks"], "{:.4f}", PALETTE["cyan"]),
                (row["split"], "Top 10% 坏账捕获", row["top_10pct_bad_capture"], "{:.1%}", PALETTE["red"]),
                (row["split"], "Score PSI vs dev", row["score_psi_vs_development"], "{:.4f}", PALETTE["green"]),
            ]
        )
    width, height = 1120, 390
    body = [
        text(32, 38, "模型验证指标", 24, PALETTE["ink"]),
        text(32, 66, "validation 用于早停，final_holdout 只做最终只读验证", 13, PALETTE["muted"]),
    ]
    card_w, card_h = 248, 94
    start_x, start_y = 32, 102
    gap_x, gap_y = 24, 32
    for idx, (split, name, value, fmt, color) in enumerate(rows):
        col = idx % 4
        row_no = idx // 4
        x = start_x + col * (card_w + gap_x)
        y = start_y + row_no * (card_h + gap_y)
        body.append(rect(x, y, card_w, card_h, "#f8fafc", 8, "#e5e7eb"))
        body.append(rect(x, y, 6, card_h, color, 6))
        body.append(text(x + 20, y + 28, split, 13, PALETTE["muted"]))
        body.append(text(x + 20, y + 58, fmt.format(value), 24, PALETTE["ink"]))
        body.append(text(x + 20, y + 80, name, 13, PALETTE["slate"]))
    return write_svg("01_model_validation_metrics.svg", width, height, body)


def draw_strategy_action_profile() -> Path:
    df = pd.read_csv(TABLE_DIR / "a_card_strategy_action_metrics.csv")
    df = df[df["split"] == "final_holdout"].copy()
    order = ["approve", "manual_review", "decline"]
    df["strategy_action"] = pd.Categorical(df["strategy_action"], order, ordered=True)
    df = df.sort_values("strategy_action")
    width, height = 1120, 460
    body = [
        text(32, 38, "A 卡三段策略表现", 24, PALETTE["ink"]),
        text(32, 66, "final_holdout：自动通过、人工复核、直接拒绝三段客群的风险浓度", 13, PALETTE["muted"]),
    ]
    panels = [
        ("population_pct", "人群占比", PALETTE["blue"], 80, 0.85, pct),
        ("bad_rate", "坏账率", PALETTE["red"], 610, 0.35, pct),
    ]
    for col, title, color, x0, vmax, formatter in panels:
        body.append(text(x0, 110, title, 17, PALETTE["ink"]))
        chart_x, chart_y, chart_w, chart_h = x0, 135, 420, 230
        body.append(line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, PALETTE["grid"], 1))
        bar_w = 84
        gap = 54
        for i, row in enumerate(df.itertuples(index=False)):
            value = getattr(row, col)
            h = chart_h * min(value / vmax, 1)
            x = chart_x + 28 + i * (bar_w + gap)
            y = chart_y + chart_h - h
            body.append(rect(x, y, bar_w, h, color, 5))
            body.append(text(x + bar_w / 2, y - 8, formatter(value), 13, PALETTE["ink"], "middle"))
            body.append(text(x + bar_w / 2, chart_y + chart_h + 28, row.strategy_action, 12, PALETTE["slate"], "middle"))
        for tick in np.linspace(0, vmax, 5):
            y = chart_y + chart_h - chart_h * tick / vmax
            body.append(line(chart_x, y, chart_x + chart_w, y, "#f1f5f9", 1))
            body.append(text(chart_x - 8, y + 4, formatter(tick), 11, PALETTE["muted"], "end"))
    return write_svg("02_strategy_action_profile.svg", width, height, body)


def draw_lift_curve() -> Path:
    df = pd.read_csv(TABLE_DIR / "lgbm_lift_table.csv")
    df = df[df["split"] == "final_holdout"].sort_values("score_band").copy()
    width, height = 1120, 500
    x0, y0, chart_w, chart_h = 80, 110, 960, 290
    max_bad_rate = max(df["bad_rate"].max() * 1.15, 0.1)
    body = [
        text(32, 38, "风险十分位 Lift", 24, PALETTE["ink"]),
        text(32, 66, "final_holdout：按预测风险从高到低分为十组，观察坏账率与累计坏样本捕获", 13, PALETTE["muted"]),
        line(x0, y0 + chart_h, x0 + chart_w, y0 + chart_h, PALETTE["ink"], 1.2),
        line(x0, y0, x0, y0 + chart_h, PALETTE["ink"], 1.2),
    ]
    for tick in np.linspace(0, max_bad_rate, 5):
        y = y0 + chart_h - chart_h * tick / max_bad_rate
        body.append(line(x0, y, x0 + chart_w, y, "#f1f5f9", 1))
        body.append(text(x0 - 10, y + 4, pct(tick, 0), 11, PALETTE["muted"], "end"))
    bar_gap = 14
    bar_w = (chart_w - bar_gap * (len(df) + 1)) / len(df)
    points = []
    for i, row in enumerate(df.itertuples(index=False)):
        x = x0 + bar_gap + i * (bar_w + bar_gap)
        h = chart_h * row.bad_rate / max_bad_rate
        y = y0 + chart_h - h
        body.append(rect(x, y, bar_w, h, PALETTE["red"], 4))
        body.append(text(x + bar_w / 2, y - 6, pct(row.bad_rate, 1), 10, PALETTE["ink"], "middle"))
        body.append(text(x + bar_w / 2, y0 + chart_h + 26, f"D{int(row.score_band)}", 11, PALETTE["slate"], "middle"))
        cx = x + bar_w / 2
        cy = y0 + chart_h - chart_h * row.cumulative_bad_capture
        points.append((cx, cy))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    body.append(f'<polyline points="{polyline}" fill="none" stroke="{PALETTE["blue"]}" stroke-width="3" />')
    for x, y in points:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{PALETTE["blue"]}" />')
    body.append(text(840, 96, "红柱：坏账率", 13, PALETTE["red"]))
    body.append(text(840, 118, "蓝线：累计坏样本捕获", 13, PALETTE["blue"]))
    return write_svg("03_lift_decile_curve.svg", width, height, body)


def draw_topk_precision_recall() -> Path:
    df = build_topk_precision_recall("final_holdout")
    width, height = 1120, 500
    x0, y0 = 56, 116
    row_h = 58
    columns = [
        ("审核TopK", 0, 110),
        ("样本数", 115, 110),
        ("Lift", 220, 70),
        ("Precision / 坏账率", 325, 220),
        ("Recall / 坏账捕获", 610, 220),
    ]
    max_precision = max(float(df["bad_rate_precision"].max()), 0.01)
    body = [
        text(32, 38, "TopK审核阈值表现", 24, PALETTE["ink"]),
        text(32, 66, "final_holdout：按预测风险从高到低截取不同审核比例，观察风险浓度与坏样本捕获", 13, PALETTE["muted"]),
        text(32, 88, "Precision 在这里等价于 TopK 客群坏账率；Recall 等价于坏账捕获率。", 12, PALETTE["muted"]),
    ]
    for label, x_off, _ in columns:
        body.append(text(x0 + x_off, y0 - 14, label, 13, PALETTE["slate"]))
    body.append(line(x0, y0, width - 56, y0, PALETTE["grid"], 1))
    for i, row in enumerate(df.itertuples(index=False)):
        y = y0 + 20 + i * row_h
        body.append(line(x0, y + 30, width - 56, y + 30, "#f1f5f9", 1))
        body.append(text(x0, y, f"Top {pct(row.topk_pct, 0)}", 15, PALETTE["ink"]))
        body.append(text(x0 + 115, y, f"{row.rows:,}", 14, PALETTE["ink"]))
        body.append(text(x0 + 220, y, f"{row.lift:.2f}x", 14, PALETTE["green"]))

        p_bar_w = 150 * row.bad_rate_precision / max_precision
        body.append(rect(x0 + 325, y - 15, 150, 18, "#f1f5f9", 4))
        body.append(rect(x0 + 325, y - 15, p_bar_w, 18, PALETTE["red"], 4))
        body.append(text(x0 + 325 + 164, y, pct(row.bad_rate_precision, 1), 13, PALETTE["ink"]))

        body.append(text(x0 + 610, y, pct(row.bad_capture_recall, 1), 14, PALETTE["blue"]))
    return write_svg("04_topk_precision_recall.svg", width, height, body)


def draw_amount_cost_heatmap() -> Path:
    df = pd.read_csv(TABLE_DIR / "strategy_amount_weighted_cost_curve.csv")
    df = df[(df["amount_cost_scenario"] == "amount_balanced_base") & (df["split"] == "validation")].copy()
    opt = pd.read_csv(TABLE_DIR / "strategy_amount_weighted_optimal_thresholds.csv")
    opt = opt[(opt["amount_cost_scenario"] == "amount_balanced_base") & (opt["split"] == "validation")].iloc[0]
    declines = sorted(df["decline_pct_target_from_validation"].unique())
    manuals = sorted(df["manual_review_pct_target_from_validation"].unique())
    pivot = df.pivot_table(
        index="manual_review_pct_target_from_validation",
        columns="decline_pct_target_from_validation",
        values="incremental_profit_proxy_per_100m_credit",
        aggfunc="mean",
    )
    values = df["incremental_profit_proxy_per_100m_credit"]
    vmin, vmax = float(values.min()), float(values.max())
    width, height = 1120, 620
    x0, y0 = 150, 120
    cell_w, cell_h = 82, 58
    body = [
        text(32, 38, "金额加权成本收益热力图", 24, PALETTE["ink"]),
        text(32, 66, "amount_balanced_base / validation：颜色越红表示每 1 亿授信敞口增量利润 proxy 越高", 13, PALETTE["muted"]),
        text(x0 + cell_w * len(declines) / 2, 100, "直接拒绝比例", 14, PALETTE["ink"], "middle"),
        text(34, y0 + cell_h * len(manuals) / 2, "人工审核比例", 14, PALETTE["ink"]),
    ]
    for j, d in enumerate(declines):
        body.append(text(x0 + j * cell_w + cell_w / 2, y0 - 14, pct(d, 0), 11, PALETTE["slate"], "middle"))
    for i, m in enumerate(manuals):
        body.append(text(x0 - 12, y0 + i * cell_h + cell_h / 2 + 4, pct(m, 0), 11, PALETTE["slate"], "end"))
        for j, d in enumerate(declines):
            val = pivot.loc[m, d] if d in pivot.columns and m in pivot.index else np.nan
            x, y = x0 + j * cell_w, y0 + i * cell_h
            if pd.isna(val):
                body.append(rect(x, y, cell_w - 4, cell_h - 4, "#f8fafc", 4, "#e5e7eb"))
                continue
            fill = color_scale(float(val), vmin, vmax)
            is_best = np.isclose(d, opt["decline_pct_target_from_validation"]) and np.isclose(
                m, opt["manual_review_pct_target_from_validation"]
            )
            body.append(rect(x, y, cell_w - 4, cell_h - 4, fill, 5, PALETTE["ink"] if is_best else "#ffffff"))
            body.append(text(x + cell_w / 2 - 2, y + cell_h / 2 + 4, f"{val / 1_000_000:.1f}M", 11, PALETTE["ink"], "middle"))
            if is_best:
                body.append(text(x + cell_w / 2 - 2, y + 16, "BEST", 9, PALETTE["ink"], "middle"))
    body.append(text(820, 150, f"最优：decline {pct(opt['decline_pct_target_from_validation'], 0)} + manual {pct(opt['manual_review_pct_target_from_validation'], 0)}", 16, PALETTE["ink"]))
    body.append(text(820, 180, f"验证集收益：{num(opt['incremental_profit_proxy_per_100m_credit'], 0)} / 1亿授信", 13, PALETTE["muted"]))
    body.append(text(820, 210, f"通过敞口坏账率：{pct(opt['expected_final_approve_exposure_bad_rate'], 2)}", 13, PALETTE["muted"]))
    return write_svg("04_amount_cost_heatmap.svg", width, height, body)


def draw_amount_cost_scenarios() -> Path:
    df = pd.read_csv(TABLE_DIR / "strategy_amount_weighted_optimal_thresholds.csv")
    df = df[df["split"] == "final_holdout"].copy()
    labels = df["amount_cost_scenario"].tolist()
    width, height = 1120, 520
    x0, y0, chart_w, chart_h = 320, 110, 680, 300
    vmax = max(df["decline_pct_target_from_validation"].add(df["manual_review_pct_target_from_validation"]).max(), 0.3)
    body = [
        text(32, 38, "金额成本场景下的最优阈值", 24, PALETTE["ink"]),
        text(32, 66, "不同 LGD / 净利差 / 人审效果假设下，validation 选阈值并在 final_holdout 验证", 13, PALETTE["muted"]),
    ]
    row_h = 66
    for i, row in enumerate(df.itertuples(index=False)):
        y = y0 + i * row_h
        decline = row.decline_pct_target_from_validation
        manual = row.manual_review_pct_target_from_validation
        body.append(text(32, y + 24, labels[i], 12, PALETTE["ink"]))
        body.append(rect(x0, y, chart_w, 24, "#f1f5f9", 4))
        d_w = chart_w * decline / vmax
        m_w = chart_w * manual / vmax
        body.append(rect(x0, y, d_w, 24, PALETTE["red"], 4))
        body.append(rect(x0 + d_w, y, m_w, 24, PALETTE["amber"], 4))
        body.append(text(x0 + d_w + m_w + 8, y + 18, f"D {pct(decline,0)} / M {pct(manual,0)}", 12, PALETTE["ink"]))
        body.append(text(x0, y + 46, f"通过敞口坏账率 {pct(row.expected_final_approve_exposure_bad_rate,2)} · 坏敞口拦截 {pct(row.expected_bad_exposure_saved_capture,1)} · 每1亿收益 {num(row.incremental_profit_proxy_per_100m_credit,0)}", 11, PALETTE["muted"]))
    body.append(text(860, 445, "红：直接拒绝", 12, PALETTE["red"]))
    body.append(text(960, 445, "黄：人工审核", 12, PALETTE["amber"]))
    return write_svg("05_amount_cost_scenarios.svg", width, height, body)


def draw_repayment_capacity_segments() -> Path:
    summary, cases = build_repayment_capacity_outputs()
    write_intelligent_agent_page(summary, cases)
    write_intelligent_agent_doc()

    metrics = [
        ("annuity_to_income_mean", "年金/收入", PALETTE["red"]),
        ("bureau_debt_ratio_proxy_mean", "外部债务/授信", PALETTE["amber"]),
        ("credit_card_utilization_proxy_mean", "信用卡利用率", PALETTE["cyan"]),
        ("external_score_risk_proxy_mean", "外部评分风险", PALETTE["blue"]),
    ]
    width, height = 1120, 560
    x0, y0 = 210, 126
    row_h = 92
    metric_gap = 150
    bar_w = 88
    body = [
        text(32, 38, "偿债能力与KYC画像分层", 24, PALETTE["ink"]),
        text(32, 66, "final_holdout：按A卡策略动作对比还款压力、征信负债、信用卡使用、历史拒绝和外部评分风险", 13, PALETTE["muted"]),
        text(32, 88, "该组件用于辅助人工审核和Agent摘要，不直接替代模型风险排序。", 12, PALETTE["muted"]),
    ]
    for i, (_, label, _) in enumerate(metrics):
        body.append(text(x0 + i * metric_gap, y0 - 20, label, 12, PALETTE["slate"], "middle"))
    for i, row in enumerate(summary.itertuples(index=False)):
        y = y0 + i * row_h
        body.append(line(32, y + 50, width - 48, y + 50, "#f1f5f9", 1))
        body.append(text(32, y + 2, row.strategy_action_label, 16, PALETTE["ink"]))
        body.append(text(32, y + 24, f"坏账率 {pct(row.bad_rate,1)} / 均分 {row.avg_score:.0f}", 12, PALETTE["muted"]))
        for j, (col, _, color) in enumerate(metrics):
            value = getattr(row, col)
            value = 0 if pd.isna(value) else float(np.clip(value, 0, 1))
            x = x0 + j * metric_gap - bar_w / 2
            body.append(rect(x, y - 8, bar_w, 18, "#f1f5f9", 4))
            body.append(rect(x, y - 8, bar_w * value, 18, color, 4))
            body.append(text(x + bar_w / 2, y + 28, pct(value, 1), 12, PALETTE["ink"], "middle"))
    body.append(text(32, 505, "外部评分风险 = 1 - ext_source_mean；Home Credit无真实流水/OCR，本页将KYC定义为申请画像与偿债能力代理指标。", 12, PALETTE["muted"]))
    return write_svg("08_repayment_capacity_segments.svg", width, height, body)


def draw_shap_and_reasons() -> tuple[Path, Path]:
    shap = pd.read_csv(TABLE_DIR / "shap_global_importance.csv").head(15)
    shap_labels = [f"{r.candidate_feature} | {r.reason_label}" for r in shap.itertuples(index=False)]
    shap_path = horizontal_bar_chart(
        "06_shap_top_features.svg",
        "SHAP 全局重要性",
        "validation + final_holdout：mean absolute SHAP，解释模型整体风险排序来源",
        shap_labels,
        shap["mean_abs_shap"].tolist(),
        lambda v: f"{v:.4f}",
        PALETTE["violet"],
        left=470,
    )

    reason = pd.read_csv(TABLE_DIR / "shap_reason_code_summary.csv")
    reason = reason[(reason["split"] == "final_holdout") & (reason["strategy_action"] == "decline")].head(12)
    labels = [f"{r.reason_label} | {r.candidate_feature}" for r in reason.itertuples(index=False)]
    values = reason["applicant_share_in_action"].tolist()
    reason_path = horizontal_bar_chart(
        "07_decline_reason_codes.svg",
        "拒绝客群原因码覆盖",
        "final_holdout / decline：原因码在拒绝客群中的覆盖率，旁边数值为覆盖比例",
        labels,
        values,
        lambda v: pct(v, 1),
        PALETTE["cyan"],
        left=500,
    )
    return shap_path, reason_path


def write_intelligent_agent_page(summary: pd.DataFrame, cases: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = "\n".join(
        "<tr>"
        f"<td>{esc(row.strategy_action_label)}</td>"
        f"<td>{row.rows:,}</td>"
        f"<td>{pct(row.bad_rate, 1)}</td>"
        f"<td>{pct(row.annuity_to_income_mean, 1)}</td>"
        f"<td>{pct(row.bureau_debt_ratio_proxy_mean, 1)}</td>"
        f"<td>{pct(row.credit_card_utilization_proxy_mean, 1)}</td>"
        f"<td>{pct(row.previous_refusal_proxy_mean, 1)}</td>"
        f"<td>{pct(row.high_capacity_dimension_share, 1)}</td>"
        "</tr>"
        for row in summary.itertuples(index=False)
    )
    case_cards = []
    for row in cases.itertuples(index=False):
        case_cards.append(
            f"""
    <article class="case-card">
      <div class="case-head">
        <div>
          <span class="eyebrow">{esc(row.case_type)}</span>
          <h3>{esc(row.strategy_action_label)} · SK_ID_CURR {int(row.SK_ID_CURR)}</h3>
        </div>
        <strong>{pct(row.pd_lgbm, 1)} PD / {row.a_card_score:.0f}分</strong>
      </div>
      <div class="mini-grid">
        <div><span>授信金额</span><b>{esc(format_amount(row.credit))}</b></div>
        <div><span>收入</span><b>{esc(format_amount(row.income))}</b></div>
        <div><span>年金/收入</span><b>{pct(row.annuity_to_income_ratio, 1)}</b></div>
        <div><span>授信/收入</span><b>{row.credit_to_income_ratio:.2f}x</b></div>
        <div><span>偿债风险等级</span><b>{esc(row.capacity_level)}</b></div>
        <div><span>验证标签</span><b>{int(row.target_in_holdout)}</b></div>
      </div>
      <p><b>画像与偿债能力：</b>{esc(row.risk_dimensions)}</p>
      <p><b>模型原因码：</b>{esc(row.top_reasons)}</p>
      <p><b>Agent审核摘要：</b>{esc(row.agent_recommendation)}</p>
      <p><b>人工复核清单：</b>{esc(row.review_checklist)}</p>
    </article>
"""
        )
    case_html = "\n".join(case_cards)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>智能风控Agent审核摘要</title>
  <style>
    body {{ margin: 0; background: #f8fafc; color: #1f2937; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; }}
    h3 {{ margin: 2px 0 0; font-size: 18px; }}
    p {{ color: #475569; line-height: 1.7; }}
    a {{ color: #2563eb; text-decoration: none; }}
    .nav {{ margin: 16px 0 24px; }}
    .arch {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .arch div, .case-card, .note {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }}
    .arch span, .mini-grid span, .eyebrow {{ display: block; color: #64748b; font-size: 13px; margin-bottom: 6px; }}
    .arch strong {{ font-size: 16px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }}
    th {{ color: #475569; background: #f8fafc; font-weight: 650; }}
    .case-card {{ margin-bottom: 14px; }}
    .case-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }}
    .case-head strong {{ color: #dc2626; white-space: nowrap; }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }}
    .mini-grid div {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
    .mini-grid b {{ font-size: 15px; }}
    @media (max-width: 820px) {{ .arch, .mini-grid {{ grid-template-columns: 1fr 1fr; }} .case-head {{ display: block; }} }}
  </style>
</head>
<body>
<main>
  <h1>大模型辅助贷前准入智能风控组件</h1>
  <p>本页把A卡评分、偿债能力/KYC画像、SHAP原因码和策略证据组织成可复核的审核摘要。LLM/Agent定位为分析编排层，负责证据检索、结构化总结和人工复核提示，不直接替代可验证的模型评分与准入规则。</p>
  <div class="nav"><a href="./strategy_visual_dashboard.html">返回主Dashboard</a></div>
  <section class="arch">
    <div><span>1. 数据底座</span><strong>多表信贷申请、征信、历史申请与还款数据</strong></div>
    <div><span>2. 风险排序</span><strong>LightGBM A卡模型 + score band</strong></div>
    <div><span>3. 策略组件</span><strong>approve / manual review / decline</strong></div>
    <div><span>4. Agent输出</span><strong>风险画像、证据引用、复核清单</strong></div>
  </section>

  <h2>偿债能力与KYC画像分层</h2>
  <table>
    <thead>
      <tr><th>策略动作</th><th>样本数</th><th>坏账率</th><th>年金/收入</th><th>外部债务/授信</th><th>信用卡利用率</th><th>历史拒绝率</th><th>2+红灯维度占比</th></tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Agent审核摘要样例</h2>
  {case_html}

  <div class="note">
    <p><b>面试口径：</b>Home Credit没有真实流水账单、OCR影像或完整KYC文档，因此本项目不声称做了图像KYC或真实流水解析；这里把KYC定义为申请人画像和偿债能力代理指标。若接入真实业务，可把流水账单解析、KYC影像识别、关系网络和贷后文本记录作为Agent的外部工具与证据源。</p>
  </div>
</main>
</body>
</html>
"""
    path = REPORT_DIR / "intelligent_risk_agent_demo.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def write_intelligent_agent_doc() -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    content = """
# 面试深挖：大模型辅助贷前准入智能风控组件

本文件由 `src/09_visualize_strategy.py` 自动生成，用于准备“风控策略工程师（大模型+智能风控）”方向的项目深挖。

核心原则：不要把这个项目讲成“LLM直接做信贷审批”，而要讲成“传统A卡风控模型 + 策略阈值 + 原因码 + 大模型Agent审核摘要”的组合。信贷风控里最重要的是可验证、可监控、可复盘，LLM应当做分析编排和证据总结，而不是替代可审计的风险排序模型。

## 一句话定位

这是一个大模型辅助的贷前准入A卡智能风控组件：底层用多表信贷数据构建A卡风险评分模型，上层用三段准入策略做通过、人工复核和拒绝决策，再用偿债能力/KYC画像、SHAP原因码和大模型Agent生成可复核的审核摘要。

## 30秒项目介绍

我做了一个贷前准入A卡风控项目，数据来自Home Credit多表信贷申请和历史履约数据。项目先从当前申请、外部征信、历史申请、分期还款、信用卡余额和POS现金贷等表构建申请级特征集市，再通过缺失率、PSI、单变量AUC/KS/IV和相关性剪枝筛选变量，训练Logit baseline和LightGBM主模型。模型在final_holdout上AUC 0.7960、KS 0.4494、PR-AUC/AP 0.2913，Top 10%高风险客群坏账率30.8%、坏账捕获38.1%、Lift 3.81x。

在模型之上，我把它扩展成一个智能风控组件：模型负责风险排序，策略模块负责approve/manual_review/decline三段准入，金额加权成本收益用于选择阈值，SHAP和原因码负责解释，大模型Agent负责把客户画像、模型分、原因码、策略规则和人工复核清单组织成结构化审核摘要。

## 和岗位JD的对应

| JD能力点 | 项目对应内容 | 面试表达 |
|---|---|---|
| 大模型+传统风控 | LLM/Agent不直接打分，而是编排模型、规则、证据和审核摘要 | 大模型做非结构化理解和决策辅助，传统模型负责可验证排序 |
| 金融信贷风控 | Home Credit贷前准入A卡评分 | 明确是申请时点准入，不是贷后催收或反欺诈 |
| 流水账单解析 | 数据无真实流水，用收入、年金、授信、外部债务、还款表现做代理 | 不夸大，说明真实业务可接入流水解析工具 |
| KYC智能画像 | 职业、家庭、资产居住、收入负债、征信历史、还款纪律 | 把KYC定义为申请人画像，不声称做了OCR影像识别 |
| 偿债能力评估 | 年金/收入、授信/收入、外部债务/授信、信用卡利用率、历史还款 | 对应贷前审批中的还款能力、债务压力、多头风险 |
| 风险排序能力 | AUC、KS、PR-AUC/AP、TopK坏账率、坏账捕获、Lift | 不只看AUC，也看TopK产能下的风险浓度 |
| 稳定性 | PSI、score drift、final_holdout只读验证 | 说明没有用最终测试集做筛选或调参 |
| 业务协同 | 三段准入、人审清单、金额加权成本收益 | 把模型结果转成业务动作 |

## 项目架构怎么讲

可以按四层讲：

1. 数据层：多表信贷数据，包括当前申请、外部征信、历史申请、分期还款、信用卡余额、POS现金贷。
2. 模型层：申请级特征集市、变量筛选、Logit baseline、LightGBM A卡模型、模型分和A卡分数。
3. 策略层：approve/manual_review/decline三段准入、TopK审核、金额加权成本收益、阈值选择。
4. Agent层：读取客户画像、模型分、SHAP原因码、策略阈值和变量证据，输出审核摘要、复核清单和限制说明。

面试表达：

> 我把大模型放在最上层做解释和流程编排，不把它放在底层直接判断好坏客户。因为底层风险排序必须可量化、可回测、可监控；大模型更适合把结构化和半结构化证据转成审核人员可读的摘要。

## 为什么不用LLM直接做信用评分

回答重点：

- 信贷审批属于高风险决策，需要稳定、可解释、可审计。
- LLM输出受prompt、上下文和模型版本影响，不适合作为唯一风险排序器。
- 传统模型可以做AUC、KS、PSI、TopK、Lift、成本收益回测，LLM很难直接提供同等稳定的统计验证。
- LLM适合处理复杂文本、政策条款、审核摘要、材料核验和人审清单。

推荐话术：

> 我不会让LLM直接替代评分卡或GBDT模型。我的设计是“模型出分、规则定边界、LLM做解释和审核辅助”。LLM生成的内容必须引用模型分、原因码、策略阈值和变量证据，不能凭空给审批结论。

## 偿债能力/KYC画像怎么构建

Home Credit没有真实流水、OCR影像或完整KYC材料，所以项目里不能说做了真实流水解析或图像KYC。这里的KYC画像指申请人画像和偿债能力代理变量。

### 当前数据能做什么

| 业务问题 | 代理变量 | 解释 |
|---|---|---|
| 收入能否覆盖还款 | `annuity_to_income_ratio` | 年金/收入，近似月供压力 |
| 授信额度是否过高 | `credit_to_income_ratio` | 授信/收入，衡量额度相对收入是否偏重 |
| 外部负债压力 | `bureau_total_debt_to_credit_ratio`、`bureau_recent_24m_total_debt_to_credit_ratio` | 外部征信负债占授信比例 |
| 信用卡资金压力 | `credit_card_recent_6m_utilization_max` | 近期信用卡额度使用率 |
| 历史还款纪律 | `installment_recent_12m_late_ratio`、`installment_recent_12m_total_payment_ratio` | 是否晚还、是否足额还款 |
| 历史申请质量 | `previous_refusal_rate` | 历史被拒比例 |
| 外部综合风险 | `ext_source_mean` | 外部评分综合信号 |

### 当前结果怎么解释

| 策略动作 | 坏账率 | 均分 | 2+红灯维度占比 |
|---|---:|---:|---:|
| 自动通过 | 4.2% | 631 | 15.3% |
| 人工复核 | 16.0% | 511 | 42.5% |
| 直接拒绝 | 30.7% | 444 | 63.7% |

面试解释：

> 可以看到策略动作越严格，坏账率越高，平均A卡分越低，偿债能力画像里的多红灯比例也越高。这说明三段策略不是只凭一个模型分数硬切，而是和还款压力、外部征信、信用卡利用率、历史拒绝、外部评分等业务维度方向一致。

## Agent审核摘要怎么生成

Agent输入：

- 客户基础画像：收入、授信金额、年金、就业、家庭、外部评分。
- 偿债能力指标：年金/收入、授信/收入、外部债务/授信、信用卡利用率、历史还款纪律。
- 模型输出：LightGBM PD、A卡分数、策略动作。
- 解释证据：SHAP原因码、变量值、特征筛选证据。
- 策略证据：三段准入阈值、TopK表现、成本收益假设。

Agent输出：

- 建议动作：自动通过、人工复核、直接拒绝。
- 风险摘要：客户主要风险维度。
- 原因码：模型识别的关键风险来源。
- 人审清单：需要补充核验的材料或业务问题。
- 限制说明：数据缺失、代理变量口径、不能直接替代人工判断的地方。

关键边界：

> Agent只能组织证据和生成建议，不能越过模型和策略规则直接做审批。Agent输出要能追溯到字段、模型分、原因码或策略文档。

## 三个样例怎么讲

### 自动通过样例

特点：

- 模型PD很低，A卡分高。
- 年金/收入和授信/收入压力较低。
- 外部征信负债低。
- 未触发主要高风险原因码。

话术：

> 对低风险客户，Agent不会硬凑风险点，而是说明模型低PD、A卡分高、画像指标没有明显异常，建议自动通过并进入贷后监控。

### 人工复核样例

特点：

- 模型分处在边界区间。
- 历史拒绝率或外部评分有一定风险。
- 不适合直接拒绝，因为部分偿债指标可能仍然较好。

话术：

> 这类客户最适合体现智能风控Agent价值。模型给出边界风险，Agent把“为什么要人审”转成清单，例如核验收入稳定性、历史拒绝原因、近期信用卡使用和还款覆盖率。

### 直接拒绝样例

特点：

- 模型PD高，A卡分低。
- 外部评分差。
- 历史申请拒绝率高。
- 外部征信或逾期相关原因码明显。

话术：

> 对拒绝客户，我会让Agent输出明确的风险原因和复核边界。如果业务希望保留客户，也可以转强验证或补充材料，而不是让模型结果变成不可解释的黑箱拒绝。

## 指标被追问怎么答

### 为什么AUC 0.796不算特别高，项目还可以讲

回答：

> 信贷风控不只看AUC。AUC说明整体排序能力，KS说明好坏区分，TopK坏账率和坏账捕获说明在有限人审/拒绝产能下能不能集中风险。这个项目Top10%高风险客群坏账率30.8%，相对整体8.07%坏账率有3.81倍Lift，说明排序是有效的。

### Precision、Recall、F1怎么看

回答：

> 在风控里precision就是TopK客群坏账率，recall就是坏账捕获率。F1不是主指标，因为误杀好客户和漏放坏客户成本不对称。更合理的是看不同TopK下的坏账率、坏账捕获、Lift，再结合人工审核成本、误拒机会成本和漏放损失选阈值。

### PR-AUC为什么要看

回答：

> 样本坏账率只有约8%，正负样本不平衡。PR-AUC/AP能补充ROC-AUC，观察模型对少数坏样本的识别能力。但业务落地还是要回到TopK和成本收益。

### PSI怎么用

回答：

> PSI用于开发期稳定性检查和上线后漂移监控。我没有用final_holdout或官方test提前筛变量，避免把最终样本分布泄露进建模流程。模型冻结后，可以在final_holdout或外部样本上做漂移诊断和变量ablation。

## 成本收益被追问怎么答

当前数据有`AMT_CREDIT`，但没有真实利率、资金成本、回收金额、催收成本和客户LTV，所以不能直接算真实利润。

项目里的做法：

- 用`AMT_CREDIT`作为EAD proxy。
- 漏放坏客户损失 proxy = `AMT_CREDIT * LGD`。
- 误杀好客户机会成本 proxy = `AMT_CREDIT * net_margin_rate`。
- 人工审核成本 proxy = 固定人审成本或按授信中位数比例估计。
- 在不同拒绝率/人审率下做金额加权收益曲线。

面试话术：

> 我不会说这个收益就是公司真实利润，因为缺少真实定价、资金成本、回收和LTV。它是一个阈值选择框架，用来说明如何把模型分数转成业务决策。真实落地时需要接入实际利率、期限、资金成本、催收回收率和客户生命周期价值。

## 数据泄露被追问怎么答

容易被问：

- 你用了历史还款表现，会不会泄露？
- 你用了PSI筛变量，会不会泄露？
- 你用了Kaggle test吗？

回答：

> 我的口径是申请时点可得信息。历史还款和历史申请只代表当前申请之前已经发生的记录，不能使用当前申请之后的表现。官方test没有标签，只做预测和展示，不参与筛选、调参。final_holdout只做最终只读验证，不参与变量筛选和阈值学习。

补充：

> 真实业务中还需要有严格的时间戳和特征回溯机制，每个特征都要证明在审批时点已经可得。这个项目是公开数据，所以我用DAYS/MONTHS字段和内部holdout尽量模拟这个流程。

## 如果被问“没有真实流水和KYC，为什么能叫智能风控”

推荐回答：

> 我会明确区分当前项目能力和真实业务可扩展能力。当前数据没有真实流水、OCR影像和KYC文档，所以我没有声称做了真实流水解析或图像KYC。我现在做的是偿债能力代理画像：收入、年金、授信、外部征信负债、信用卡利用率、历史还款纪律和历史拒绝。真实业务中，可以把流水解析、OCR、反欺诈设备指纹、关系网络和客服文本作为Agent工具接入同一个框架。

继续展开：

> 这个项目的价值在于框架已经搭好：结构化模型负责排序，策略模块负责阈值，Agent负责整合多源证据和生成审核摘要。换成真实业务数据时，只需要把更多工具和证据源接入Agent，而不是推翻整个流程。

## 如果被问“LLM怎么评估效果”

回答方向：

- 不用LLM的文本生成质量替代模型指标。
- 模型效果看AUC、KS、PR-AUC、TopK、Lift、成本收益。
- Agent效果看审核效率、一致性、证据引用准确率、原因码覆盖率、人工采纳率、幻觉率。

可设计的评测：

| 评测维度 | 指标 |
|---|---|
| 证据准确性 | 输出中的字段、模型分、原因码是否能在表中找到 |
| 决策一致性 | Agent建议是否与策略动作一致 |
| 完整性 | 是否包含风险摘要、原因码、复核清单、限制说明 |
| 幻觉控制 | 是否出现数据中不存在的流水/OCR/KYC结论 |
| 人审效率 | 审核人员阅读摘要后处理时间是否下降 |

推荐话术：

> 我会把Agent评测和模型评测分开。模型评估风险排序，Agent评估证据组织和审核辅助质量。Agent不能编造证据，所有结论都要能追溯到字段、原因码或策略文档。

## 如果被问“怎么上线”

可以按上线流程回答：

1. 离线开发：特征回溯、样本切分、模型训练、稳定性验证、策略回测。
2. 灰度验证：小流量影子模式，只输出建议，不影响真实审批。
3. 人审辅助：先用于人工复核队列摘要和原因码解释。
4. 监控体系：AUC/KS、通过率、坏账率、拒绝率、人审命中率、PSI、原因码分布、Agent幻觉率。
5. 策略迭代：根据业务成本、产能和贷后表现调整阈值。
6. 合规审计：保存模型版本、策略版本、prompt版本、证据引用和输出日志。

面试话术：

> 我会先让Agent在影子模式和人审辅助场景落地，不直接影响自动审批。等证据准确率、人工采纳率和幻觉率稳定后，再逐步扩大使用范围。

## 如果被问“相比纯规则或纯模型有什么价值”

纯规则问题：

- 可解释但覆盖有限。
- 容易被绕过。
- 难以捕捉多变量交互。

纯模型问题：

- 排序能力强，但业务人员不一定能直接理解。
- 人审时需要额外解释。
- 阈值选择必须结合成本和产能。

本项目组合价值：

- GBDT模型捕捉复杂非线性风险。
- 三段策略把分数转成业务动作。
- 成本收益把阈值选择业务化。
- SHAP原因码把模型结果翻译成风险语言。
- Agent把多源证据组织成审核摘要，提高人审效率和复盘能力。

## 可能被追问的薄弱点与回答

### 1. Home Credit是Kaggle数据，和真实业务有什么差距

回答：

> 差距主要在数据实时性、真实审批时间戳、定价成本、贷后回收、KYC材料和流水信息。我的项目重点不是声称完全复刻业务，而是展示我理解真实风控流程：数据盘点、防泄露、特征构建、模型验证、策略阈值、原因解释和监控。真实业务接入后，我会优先补时间戳、真实成本和外部数据源。

### 2. 你的偿债能力阈值是不是拍脑袋

回答：

> 当前阈值是为了展示审核摘要和分层逻辑，不作为最终准入规则。真正上线时应结合政策规则、历史分布、坏账率单调性和业务专家经验校准。模型排序仍然由LightGBM和策略回测决定，偿债能力红灯主要用于解释和人审清单。

### 3. 为什么外部评分这么重要，会不会不可解释

回答：

> `ext_source_mean`确实是强变量，但我没有只依赖它。一方面保留它提升排序能力，另一方面通过SHAP原因码和偿债能力画像补充解释。如果业务要求强解释或合规，可以做去除外部评分的ablation，看性能损失和策略稳定性。

### 4. 为什么不是端到端深度学习

回答：

> 信贷表格数据里GBDT通常更稳、更容易解释、更适合中小规模结构化数据。深度学习可以用于文本、图像、关系网络或序列行为，但对当前结构化A卡数据，LightGBM更符合可解释和快速迭代要求。

### 5. 这个Agent有没有真的调用大模型

回答：

> 当前页面是Agent输出形态和证据组织流程的原型，用规则模板模拟了LLM应输出的结构，重点验证输入、证据、边界和审核摘要设计。下一步可以接入LLM，把这些表格、原因码、策略文档作为RAG证据库，让模型生成同样结构的JSON输出，并用自动化评测检查证据准确性和幻觉。

## 不要这样说

- 不要说“LLM提升了AUC”。AUC来自LightGBM，LLM负责解释和审核辅助。
- 不要说“做了真实流水解析”。当前没有流水，只是偿债能力代理指标。
- 不要说“做了OCR KYC”。当前没有图像KYC，只是申请画像。
- 不要说“成本收益是真实利润”。这是金额加权proxy，真实业务需要利率、资金成本、回收和LTV。
- 不要说“final_holdout参与筛选”。final_holdout只做最终验证。
- 不要把F1作为主指标。风控主讲AUC、KS、TopK、Lift、成本收益。

## 3分钟讲述模板

> 这个项目我会定位成大模型辅助贷前准入智能风控组件。底层是Home Credit多表信贷数据，我先从当前申请、外部征信、历史申请、分期还款、信用卡和POS现金贷构建申请级特征集市，再做缺失率、PSI、AUC/KS/IV和相关性筛选，最终训练Logit baseline和LightGBM A卡模型。模型在final_holdout上AUC 0.7960、KS 0.4494，Top10%高风险客群坏账率30.8%、坏账捕获38.1%、Lift 3.81x。
>
> 模型之外，我重点做了业务落地：把模型分转成approve、manual_review、decline三段准入策略，并用AMT_CREDIT作为EAD proxy设计金额加权成本收益框架，考虑LGD、净利差、人审成本、误杀和漏放。为了贴近大模型智能风控，我又加了一层Agent审核摘要：Agent读取模型分、A卡分、偿债能力/KYC画像、SHAP原因码和策略证据，输出审核建议、风险摘要和人工复核清单。
>
> 我不会让LLM直接判断客户好坏，因为信贷审批需要可验证、可监控和可审计。我的设计是传统模型负责风险排序，策略模块负责阈值和动作，大模型负责证据整合、审核摘要和复盘提效。

## 8分钟深挖讲述结构

1. 业务目标：贷前准入A卡，不是刷榜，而是把模型转成审批策略。
2. 数据选取：当前申请、外部征信、历史申请、历史还款、信用卡、POS现金贷。
3. 防泄露：申请时点可得、final_holdout只读、官方test不参与筛选。
4. 特征构建：收入负债、授信压力、近期窗口、外部债务、还款纪律、信用卡利用率、历史拒绝。
5. 特征筛选：缺失率、PSI、单变量AUC/KS/IV、相关性剪枝，674到368。
6. 模型结果：AUC、KS、PR-AUC/AP、TopK坏账率、坏账捕获、Lift。
7. 策略落地：三段准入、TopK审核、人审产能、金额加权成本收益。
8. 智能风控：偿债能力/KYC画像、SHAP原因码、Agent审核摘要。
9. 边界说明：无真实流水/OCR，LLM不直接打分，收益是proxy。
10. 上线方案：影子模式、人审辅助、监控、审计、迭代。

## 输出文件

- `reports/intelligent_risk_agent_demo.html`：智能风控Agent审核摘要页面。
- `outputs/figures/08_repayment_capacity_segments.svg`：偿债能力与KYC画像分层图。
- `outputs/tables/risk_agent_capacity_segment_summary.csv`：三段策略客群画像指标汇总。
- `outputs/tables/risk_agent_case_studies.csv`：三类样例的结构化审核摘要。
- `reports/strategy_visual_dashboard.html`：主Dashboard。
- `docs/interview_data_feature_deep_dive.md`：数据选取、特征构建与筛选深挖。
"""
    path = DOCS_DIR / "intelligent_risk_agent_demo.md"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def build_project_overview() -> list[tuple[str, str]]:
    split_summary = pd.read_csv(TABLE_DIR / "feature_screening_split_summary.csv")
    feature_groups = pd.read_csv(TABLE_DIR / "feature_screening_group_summary.csv")
    feature_store = pd.read_csv(TABLE_DIR / "feature_store_group_profile.csv")
    metrics = pd.read_csv(TABLE_DIR / "lgbm_metrics.csv")

    labeled_rows = int(split_summary[split_summary["dataset"] == "train"]["rows"].sum())
    external_rows = int(split_summary[split_summary["screening_split"] == "external_unlabeled"]["rows"].iloc[0])
    dev = metrics[metrics["split"] == "development"].iloc[0]
    val = metrics[metrics["split"] == "validation"].iloc[0]
    holdout = metrics[metrics["split"] == "final_holdout"].iloc[0]
    feature_store_count = int(feature_store["feature_count"].sum())
    candidate_count = int(feature_groups["candidate_count"].sum())
    shortlist_count = int(feature_groups["final_shortlist_count"].sum())

    return [
        ("数据来源", "Home Credit Default Risk（Kaggle）多表信贷申请与历史履约数据"),
        ("样本量", f"有标签 train {labeled_rows:,}；官方无标签 test {external_rows:,}"),
        ("特征构建", f"申请级特征集市 {feature_store_count:,} 个字段；建模候选 {candidate_count:,} 个特征"),
        ("特征筛选", f"质量、单变量、PSI 和相关性筛选后保留 {shortlist_count:,} 个入模特征"),
        ("模型", "Logit baseline + LightGBM GBDT；主模型使用 LightGBM"),
        (
            "验证口径",
            f"development {int(dev['rows']):,} / validation {int(val['rows']):,} / final_holdout {int(holdout['rows']):,}；坏账率 {pct(holdout['bad_rate'], 2)}",
        ),
    ]


def write_dashboard(figures: list[Path]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(TABLE_DIR / "lgbm_metrics.csv")
    holdout = metrics[metrics["split"] == "final_holdout"].iloc[0]
    amount_opt = pd.read_csv(TABLE_DIR / "strategy_amount_weighted_optimal_thresholds.csv")
    amount_holdout = amount_opt[
        (amount_opt["amount_cost_scenario"] == "amount_balanced_base") & (amount_opt["split"] == "final_holdout")
    ].iloc[0]
    predictions = pd.read_csv(TABLE_DIR / "lgbm_predictions.csv")
    holdout_predictions = predictions[predictions["screening_split"] == "final_holdout"]
    holdout_ap = average_precision(
        holdout_predictions["TARGET"].to_numpy(),
        holdout_predictions["pred_risk"].to_numpy(),
    )
    topk = build_topk_precision_recall("final_holdout")
    top10 = topk[np.isclose(topk["topk_pct"], 0.10)].iloc[0]
    cards = [
        ("final_holdout AUC", f"{holdout['auc']:.4f}"),
        ("KS", f"{holdout['ks']:.4f}"),
        ("PR-AUC / AP", f"{holdout_ap:.4f}"),
        ("Top10 Precision / Recall", f"{pct(top10['bad_rate_precision'], 1)} / {pct(top10['bad_capture_recall'], 1)}"),
        ("金额成本最优策略", f"D {pct(amount_holdout['decline_pct_target_from_validation'],0)} / M {pct(amount_holdout['manual_review_pct_target_from_validation'],0)}"),
        ("通过敞口坏账率", pct(amount_holdout["expected_final_approve_exposure_bad_rate"], 2)),
        ("每1亿授信收益 proxy", num(amount_holdout["incremental_profit_proxy_per_100m_credit"], 0)),
        ("智能风控Agent页面", "已生成"),
    ]
    rel_figures = [Path("../outputs/figures") / fig.name for fig in figures]
    card_html = "\n".join(
        f'<div class="card"><span>{esc(name)}</span><strong>{esc(value)}</strong></div>' for name, value in cards
    )
    overview_html = "\n".join(
        f'<div class="overview-item"><span>{esc(name)}</span><strong>{esc(value)}</strong></div>'
        for name, value in build_project_overview()
    )
    fig_html = "\n".join(
        f'<section><img src="{esc(path.as_posix())}" alt="{esc(path.stem)}" /></section>' for path in rel_figures
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Home Credit A卡风控策略可视化</title>
  <style>
    body {{ margin: 0; background: #f8fafc; color: #1f2937; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 56px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    p {{ margin: 0 0 22px; color: #64748b; line-height: 1.6; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    .overview {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 18px 0 26px; }}
    .overview-item {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; }}
    .overview-item span {{ display: block; color: #64748b; font-size: 13px; margin-bottom: 7px; }}
    .overview-item strong {{ font-size: 15px; line-height: 1.5; font-weight: 650; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 22px 0 28px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 18px; }}
    .card span {{ display: block; color: #64748b; font-size: 13px; margin-bottom: 8px; }}
    .card strong {{ font-size: 24px; }}
    .links {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 0 0 22px; }}
    .links a {{ background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; color: #2563eb; padding: 10px 14px; text-decoration: none; font-weight: 650; }}
    section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 18px; }}
    img {{ width: 100%; height: auto; display: block; }}
    @media (max-width: 760px) {{ .overview, .cards {{ grid-template-columns: 1fr; }} main {{ padding: 20px 12px 36px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Home Credit A卡风控策略可视化</h1>
  <p>图表聚焦模型排序、TopK审核表现、偿债能力/KYC画像、三段准入策略、金额加权成本收益、SHAP解释和原因码。</p>
  <div class="links"><a href="./intelligent_risk_agent_demo.html">查看智能风控Agent审核摘要</a></div>
  <h2>项目概览</h2>
  <div class="overview">{overview_html}</div>
  <h2>核心结果</h2>
  <div class="cards">{card_html}</div>
  {fig_html}
</main>
</body>
</html>
"""
    path = REPORT_DIR / "strategy_visual_dashboard.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def write_doc(figures: list[Path], dashboard_path: Path) -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 可视化报告初版",
        "",
        "本文件由 `src/09_visualize_strategy.py` 自动生成。",
        "",
        "## 设计目标",
        "",
        "- 用图表证明模型在 `final_holdout` 上有稳定排序能力。",
        "- 展示不同 TopK 审核比例下的坏账率、坏账捕获率和 Lift。",
        "- 展示偿债能力与KYC画像分层，并生成智能风控Agent审核摘要页面。",
        "- 把 approve / manual_review / decline 的风险差异展示出来。",
        "- 用金额加权成本热力图解释阈值不是拍脑袋，而是由 LGD、净利差、人审成本和产能共同决定。",
        "- 用 SHAP 和 reason code 把模型风险分数翻译成信贷业务语言。",
        "",
        "## 输出文件",
        "",
        f"- `{dashboard_path.relative_to(PROJECT_ROOT)}`：HTML dashboard。",
        "- `reports/intelligent_risk_agent_demo.html`：智能风控Agent审核摘要页面。",
    ]
    for fig in figures:
        lines.append(f"- `{fig.relative_to(PROJECT_ROOT)}`")
    lines.extend(
        [
            "",
            "## 推荐讲述顺序",
            "",
            "1. 先看模型指标：final_holdout AUC、KS、PR-AUC/AP。",
            "2. 再看 TopK 审核表现：Top 10% 客群坏账率约 30%，坏账捕获约 38%，说明排序方向和风险浓度正常。",
            "3. 然后看偿债能力与KYC画像分层：说明还款压力、外部债务、信用卡利用和历史拒绝如何辅助人审。",
            "4. 再看三段策略：拒绝池和人工审核池坏账率明显高于通过池。",
            "5. 接着看金额成本热力图：在默认金额场景下选择 12% decline + 10% manual review。",
            "6. 最后用 SHAP、原因码和Agent审核摘要解释为什么这些客户被识别为高风险。",
            "",
        ]
    )
    path = DOCS_DIR / "visualization_initial.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    figures = [
        draw_metric_cards(),
        draw_strategy_action_profile(),
        draw_lift_curve(),
        draw_topk_precision_recall(),
        draw_repayment_capacity_segments(),
        draw_amount_cost_heatmap(),
        draw_amount_cost_scenarios(),
    ]
    figures.extend(draw_shap_and_reasons())
    dashboard = write_dashboard(figures)
    doc = write_doc(figures, dashboard)
    print(f"Wrote {dashboard}")
    print(f"Wrote {doc}")
    for fig in figures:
        print(f"Wrote {fig}")


if __name__ == "__main__":
    main()
