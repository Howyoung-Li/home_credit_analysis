# 可视化报告初版

本文件由 `src/09_visualize_strategy.py` 自动生成。

## 设计目标

- 用图表证明模型在 `final_holdout` 上有稳定排序能力。
- 把 approve / manual_review / decline 的风险差异展示出来。
- 用金额加权成本热力图解释阈值不是拍脑袋，而是由 LGD、净利差、人审成本和产能共同决定。
- 用 SHAP 和 reason code 把模型风险分数翻译成信贷业务语言。

## 输出文件

- `reports/strategy_visual_dashboard.html`：HTML dashboard。
- `outputs/figures/01_model_validation_metrics.svg`
- `outputs/figures/02_strategy_action_profile.svg`
- `outputs/figures/03_lift_decile_curve.svg`
- `outputs/figures/04_amount_cost_heatmap.svg`
- `outputs/figures/05_amount_cost_scenarios.svg`
- `outputs/figures/06_shap_top_features.svg`
- `outputs/figures/07_decline_reason_codes.svg`

## 推荐讲述顺序

1. 先看模型指标：final_holdout AUC、KS、Top 10% 坏账捕获。
2. 再看三段策略：拒绝池和人工审核池坏账率明显高于通过池。
3. 然后看金额成本热力图：在默认金额场景下选择 12% decline + 10% manual review。
4. 最后用 SHAP 和原因码解释为什么这些客户被识别为高风险。
