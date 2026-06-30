# 策略阈值、收益曲线与 SHAP 原因码

本文件由 `src/08_explain_monitor.py` 自动生成。

## 口径

- 阈值从 `validation` 分布确定，`final_holdout` 只用于只读检验。
- 收益曲线使用相对经济口径：拦住 1 个坏客户的价值 = `bad_loss_to_good_profit`，误拒 1 个好客户的机会成本 = 1。
- 因为 Home Credit 数据没有真实利率、额度、LGD、资金成本，本页不声称绝对利润，只给 break-even 和相对收益。
- SHAP 使用 LightGBM `pred_contrib=True`，解释的是模型 raw log-odds 风险贡献；正值表示推高预测违约风险。

## 10% 拒绝策略是否划算

| split | actual_decline_pct | decline_bad_rate | approve_bad_rate | bad_rejected | good_rejected | bad_capture | break_even_bad_loss/good_profit |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 10.00% | 31.67% | 5.45% | 1948 | 4203 | 39.23% | 2.16 |
| final_holdout | 10.06% | 30.74% | 5.54% | 1901 | 4284 | 38.29% | 2.25 |

- 在 final_holdout 上，拒绝高风险约 10% 客群会拦住 1901 个坏客户，同时误拒 4284 个好客户。
- break-even 为 2.25：只要一个坏客户带来的净损失超过一个好客户净收益的 2.25 倍，这条 10% 拒绝线在经济上就是正收益。

## 不同坏账损失倍数下的最优拒绝率

| split | bad_loss/good_profit | best_decline_pct | approve_bad_rate | decline_bad_rate | net_value_per_10k_apps |
|---|---:|---:|---:|---:|---:|
| final_holdout | 1.0 | 1% | 7.60% | 54.17% | 8.5 |
| final_holdout | 2.0 | 3% | 7.02% | 43.14% | 86.0 |
| final_holdout | 3.0 | 5% | 6.46% | 38.61% | 272.3 |
| final_holdout | 5.0 | 12% | 5.24% | 28.94% | 879.2 |
| final_holdout | 8.0 | 25% | 3.74% | 21.05% | 2238.5 |
| final_holdout | 10.0 | 30% | 3.32% | 19.14% | 3318.9 |

## 三段式成本收益模型

这里把阈值选择改成 approve / manual_review / decline 的统一成本函数，所有数值都用相对单位表示：

- 漏放坏客户成本 `bad_loss_cost`：坏客户如果被放款造成的预期净损失。
- 误杀好客户成本 `false_decline_good_cost`：好客户被拒造成的利润、获客和关系损失。
- 人工审核成本 `manual_review_cost`：每进入人工审核一单的运营成本。
- 人审坏客户拦截率 `manual_bad_catch_rate`：人工审核能识别并拒绝多少复核区坏客户。
- 人审好客户误杀率 `manual_good_false_decline_rate`：人工审核误拒多少复核区好客户。
- 产能约束：限制最大直接拒绝比例、最大人工审核比例和总干预比例，避免成本函数给出运营上不可落地的策略。

增量收益相对 `approve-all` 计算：

`bad_saved * bad_loss_cost - good_lost * false_decline_good_cost - manual_review_rows * manual_review_cost`

## 成本参数场景

| scenario | bad_loss | false_decline | manual_cost | manual_bad_catch | manual_good_false_decline | max_decline | max_manual |
|---|---:|---:|---:|---:|---:|---:|---:|
| low_bad_loss_low_manual_value | 3.00 | 1.00 | 0.10 | 35% | 4% | 10% | 10% |
| balanced_base | 5.00 | 1.00 | 0.10 | 60% | 6% | 15% | 10% |
| high_bad_loss_strong_manual | 8.00 | 1.00 | 0.10 | 70% | 8% | 20% | 15% |
| expensive_manual_conservative_reject | 5.00 | 1.50 | 0.25 | 50% | 5% | 10% | 10% |

## 按成本收益选择阈值

- 默认 `balanced_base` 场景在 validation 上选择：直接拒绝 10%，人工审核 10%。
- 同一组阈值在 final_holdout 上的期望最终通过池坏账率为 4.55%，期望拦截坏客户覆盖率为 50.18%，每万申请增量收益为 1178.6 个成本单位。

| scenario | selected_decline | selected_manual_review | holdout_approve_bad_rate | bad_saved_capture | final_reject_pct | value_per_10k_apps |
|---|---:|---:|---:|---:|---:|---:|
| balanced_base | 10% | 10% | 4.55% | 50.18% | 11.52% | 1178.6 |
| expensive_manual_conservative_reject | 5% | 10% | 5.48% | 36.52% | 6.42% | 702.3 |
| high_bad_loss_strong_manual | 15% | 15% | 3.46% | 64.57% | 17.35% | 2806.9 |
| low_bad_loss_low_manual_value | 5% | 10% | 5.78% | 32.74% | 6.03% | 353.4 |

## 金额加权成本模型

当前数据有 `AMT_CREDIT`，因此可以把它作为 EAD proxy 做金额加权策略评估。这个口径仍不是绝对利润，因为数据没有真实利率、资金成本、回收金额、催收成本和客户 LTV。

- 漏放坏客户损失 proxy：`AMT_CREDIT * LGD`。
- 误杀好客户机会成本 proxy：`AMT_CREDIT * net_margin_rate`。
- 人工审核成本 proxy：`validation AMT_CREDIT 中位数 * manual_review_cost_rate_of_median_credit`。
- 金额加权阈值同样只在 `validation` 上选择，`final_holdout` 只做验证。

## 金额成本参数场景

| scenario | LGD | net_margin | manual_cost_rate | manual_bad_catch | manual_good_false_decline | max_decline | max_manual |
|---|---:|---:|---:|---:|---:|---:|---:|
| low_lgd_low_margin | 35% | 5% | 0.10% | 35% | 4% | 10% | 10% |
| amount_balanced_base | 50% | 8% | 0.10% | 60% | 6% | 15% | 10% |
| high_lgd_growth | 70% | 10% | 0.10% | 70% | 8% | 20% | 15% |
| high_margin_high_false_decline | 50% | 12% | 0.20% | 50% | 5% | 10% | 10% |

## 按金额加权成本选择阈值

- 默认 `amount_balanced_base` 场景在 validation 上选择：直接拒绝 12%，人工审核 10%。
- 同一组阈值在 final_holdout 上的期望通过敞口坏账率为 4.27%，期望拦截坏客户敞口覆盖率为 49.72%，每 1 亿授信敞口增量利润 proxy 为 1,217,397。

| scenario | selected_decline | selected_manual_review | holdout_approve_exposure_bad_rate | bad_exposure_saved | reject_exposure_share | profit_proxy_per_100m_credit |
|---|---:|---:|---:|---:|---:|---:|
| amount_balanced_base | 12% | 10% | 4.27% | 49.72% | 11.70% | 1,217,397 |
| high_lgd_growth | 10% | 15% | 3.94% | 53.22% | 11.04% | 2,075,641 |
| high_margin_high_false_decline | 7% | 10% | 4.99% | 38.24% | 7.19% | 897,153 |
| low_lgd_low_margin | 10% | 10% | 4.85% | 41.51% | 9.52% | 760,401 |

## SHAP 全局 Top 特征

| feature | reason_label | group | mean_abs_shap | gain_pct |
|---|---|---|---:|---:|
| `ext_source_mean` | 外部综合评分/外部数据风险信号 | current_application | 0.40687 | 17.34% |
| `employment_years` | 就业稳定性相关风险 | current_application | 0.11625 | 2.03% |
| `catfreq__CODE_GENDER` | 模型识别的申请资料风险信号 | current_application | 0.10856 | 0.66% |
| `AMT_ANNUITY` | 模型识别的申请资料风险信号 | current_application | 0.10237 | 1.19% |
| `credit_to_annuity_ratio` | 模型识别的申请资料风险信号 | current_application | 0.08802 | 2.17% |
| `goods_to_credit_ratio` | 模型识别的申请资料风险信号 | current_application | 0.08736 | 1.34% |
| `OWN_CAR_AGE` | 年龄段相关风险 | current_application | 0.07411 | 0.76% |
| `installment_payment_sum` | 历史分期还款表现风险 | internal_repayment_history | 0.06511 | 1.01% |
| `EXT_SOURCE_1` | 外部综合评分/外部数据风险信号 | current_application | 0.06052 | 1.11% |
| `DAYS_BIRTH` | 年龄段相关风险 | current_application | 0.06047 | 1.36% |
| `catoh__NAME_FAMILY_STATUS__01_Married` | 模型识别的申请资料风险信号 | current_application | 0.05906 | 0.26% |
| `ext_source_max` | 外部综合评分/外部数据风险信号 | current_application | 0.05668 | 2.34% |
| `bureau_recent_24m_debt_to_credit_max` | 征信负债相对授信额度偏高 | external_credit_history | 0.05584 | 1.58% |
| `catoh__NAME_EDUCATION_TYPE__02_Higher_education` | 模型识别的申请资料风险信号 | current_application | 0.05174 | 0.43% |
| `id_publish_years` | 模型识别的申请资料风险信号 | current_application | 0.04942 | 0.87% |

## final_holdout 拒绝客群主要原因码

| reason_label | feature | applicant_share | bad_rate | avg_shap |
|---|---|---:|---:|---:|
| 外部综合评分/外部数据风险信号 | `ext_source_mean` | 91.56% | 31.22% | 0.7457 |
| 就业稳定性相关风险 | `employment_years` | 42.73% | 29.97% | 0.1543 |
| 模型识别的申请资料风险信号 | `catfreq__CODE_GENDER` | 36.23% | 31.33% | 0.1797 |
| 外部综合评分/外部数据风险信号 | `ext_source_min` | 30.67% | 33.53% | 0.1990 |
| 模型识别的申请资料风险信号 | `credit_to_annuity_ratio` | 30.06% | 31.47% | 0.1720 |
| 模型识别的申请资料风险信号 | `goods_to_credit_ratio` | 28.38% | 33.45% | 0.1795 |
| 模型识别的申请资料风险信号 | `AMT_ANNUITY` | 24.66% | 29.31% | 0.1719 |
| 历史分期还款表现风险 | `installment_payment_sum` | 19.98% | 29.77% | 0.1636 |
| 征信负债相对授信额度偏高 | `bureau_recent_24m_debt_to_credit_max` | 18.77% | 30.49% | 0.1849 |
| 近期信用卡额度使用率偏高 | `credit_card_recent_6m_utilization_max` | 17.66% | 34.98% | 0.2725 |
| 外部综合评分/外部数据风险信号 | `EXT_SOURCE_3` | 14.84% | 31.81% | 0.1549 |
| 历史申请被拒记录较多 | `previous_refusal_rate` | 13.52% | 33.85% | 0.1643 |

## 输出文件

- `outputs/tables/strategy_threshold_curve.csv`：不同拒绝率的坏账率、捕获率、误拒率、break-even。
- `outputs/tables/strategy_profit_curve.csv`：不同坏账损失倍数下的相对收益曲线。
- `outputs/tables/strategy_profit_curve_summary.csv`：每个损失倍数下的最优拒绝率。
- `outputs/tables/strategy_cost_assumptions.csv`：人工审核、漏放、误杀成本参数场景。
- `outputs/tables/strategy_cost_sensitivity_curve.csv`：三段式策略成本收益全网格。
- `outputs/tables/strategy_cost_optimal_thresholds.csv`：validation 选阈值后在各 split 的表现。
- `outputs/tables/strategy_amount_cost_assumptions.csv`：LGD、净利差、人审成本率参数场景。
- `outputs/tables/strategy_amount_weighted_cost_curve.csv`：金额加权成本收益全网格。
- `outputs/tables/strategy_amount_weighted_optimal_thresholds.csv`：金额加权 validation 选阈值后在各 split 的表现。
- `outputs/tables/shap_global_importance.csv`：SHAP 全局重要性。
- `outputs/tables/shap_reason_code_summary.csv`：decline/manual_review 客群原因码汇总。
- `outputs/tables/shap_reason_code_long.csv`：decline/manual_review 样本 Top 原因长表。
- `outputs/tables/a_card_reason_code_sample.csv`：面试展示用原因码样例。
