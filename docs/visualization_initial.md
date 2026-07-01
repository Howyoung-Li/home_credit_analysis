# 可视化报告初版

本文件由 `src/09_visualize_strategy.py` 自动生成。

## 设计目标

- 用图表证明模型在 `final_holdout` 上有稳定排序能力。
- 展示不同 TopK 审核比例下的坏账率、坏账捕获率和 Lift。
- 展示偿债能力与KYC画像分层，并生成智能风控Agent审核摘要页面。
- 把 approve / manual_review / decline 的风险差异展示出来。
- 用金额加权成本热力图解释阈值不是拍脑袋，而是由 LGD、净利差、人审成本和产能共同决定。
- 用 SHAP 和 reason code 把模型风险分数翻译成信贷业务语言。

## 输出文件

- `reports/strategy_visual_dashboard.html`：HTML dashboard。
- `reports/intelligent_risk_agent_demo.html`：智能风控Agent审核摘要页面。
- `outputs/figures/01_model_validation_metrics.svg`
- `outputs/figures/02_strategy_action_profile.svg`
- `outputs/figures/03_lift_decile_curve.svg`
- `outputs/figures/04_topk_precision_recall.svg`
- `outputs/figures/08_repayment_capacity_segments.svg`
- `outputs/figures/04_amount_cost_heatmap.svg`
- `outputs/figures/05_amount_cost_scenarios.svg`
- `outputs/figures/06_shap_top_features.svg`
- `outputs/figures/07_decline_reason_codes.svg`

## 推荐讲述顺序

1. 先看模型指标：final_holdout AUC、KS、PR-AUC/AP。
2. 再看 TopK 审核表现：Top 10% 客群坏账率约 30%，坏账捕获约 38%，说明排序方向和风险浓度正常。
3. 然后看偿债能力与KYC画像分层：说明还款压力、外部债务、信用卡利用和历史拒绝如何辅助人审。
4. 再看三段策略：拒绝池和人工审核池坏账率明显高于通过池。
5. 接着看金额成本热力图：在默认金额场景下选择 12% decline + 10% manual review。
6. 最后用 SHAP、原因码和Agent审核摘要解释为什么这些客户被识别为高风险。
