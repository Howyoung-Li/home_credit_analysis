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
    cards = [
        ("final_holdout AUC", f"{holdout['auc']:.4f}"),
        ("KS", f"{holdout['ks']:.4f}"),
        ("Top 10% 坏账捕获", pct(holdout["top_10pct_bad_capture"], 1)),
        ("金额成本最优策略", f"D {pct(amount_holdout['decline_pct_target_from_validation'],0)} / M {pct(amount_holdout['manual_review_pct_target_from_validation'],0)}"),
        ("通过敞口坏账率", pct(amount_holdout["expected_final_approve_exposure_bad_rate"], 2)),
        ("每1亿授信收益 proxy", num(amount_holdout["incremental_profit_proxy_per_100m_credit"], 0)),
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
    section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 18px; }}
    img {{ width: 100%; height: auto; display: block; }}
    @media (max-width: 760px) {{ .overview, .cards {{ grid-template-columns: 1fr; }} main {{ padding: 20px 12px 36px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Home Credit A卡风控策略可视化</h1>
  <p>图表聚焦模型排序、三段准入策略、金额加权成本收益、SHAP解释和原因码。</p>
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
        "- 把 approve / manual_review / decline 的风险差异展示出来。",
        "- 用金额加权成本热力图解释阈值不是拍脑袋，而是由 LGD、净利差、人审成本和产能共同决定。",
        "- 用 SHAP 和 reason code 把模型风险分数翻译成信贷业务语言。",
        "",
        "## 输出文件",
        "",
        f"- `{dashboard_path.relative_to(PROJECT_ROOT)}`：HTML dashboard。",
    ]
    for fig in figures:
        lines.append(f"- `{fig.relative_to(PROJECT_ROOT)}`")
    lines.extend(
        [
            "",
            "## 推荐讲述顺序",
            "",
            "1. 先看模型指标：final_holdout AUC、KS、Top 10% 坏账捕获。",
            "2. 再看三段策略：拒绝池和人工审核池坏账率明显高于通过池。",
            "3. 然后看金额成本热力图：在默认金额场景下选择 12% decline + 10% manual review。",
            "4. 最后用 SHAP 和原因码解释为什么这些客户被识别为高风险。",
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
