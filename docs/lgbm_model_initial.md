# LightGBM 模型初版

本文件由 `src/06_train_lgbm.py` 自动生成。

## 建模口径

- 使用 `feature_screening_shortlist.csv` 中的统计短名单变量。
- 只在 `development` 上训练 LightGBM。
- 使用 `validation` 做 early stopping。
- `final_holdout` 不参与训练、筛选或调参，只做最终只读评估。
- 训练后只用 development 做 raw score 截距校准，使平均预测 PD 对齐 development 坏样本率。

## 核心指标

| split | rows | bad_rate | AUC | KS | top_5pct_bad_capture | top_10pct_bad_capture | score_PSI_vs_dev |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 184507 | 8.07% | 0.9134 | 0.6607 | 42.20% | 61.80% | 0.0000 |
| validation | 61502 | 8.07% | 0.7904 | 0.4395 | 23.99% | 39.23% | 0.0008 |
| final_holdout | 61502 | 8.07% | 0.7960 | 0.4494 | 23.93% | 38.11% | 0.0006 |

## 输出文件

- `outputs/tables/lgbm_metrics.csv`：development / validation / final_holdout 指标。
- `outputs/tables/lgbm_lift_table.csv`：十分位 Lift 和累计坏样本捕获。
- `outputs/tables/lgbm_feature_importance.csv`：gain/split 特征重要性。
- `outputs/tables/lgbm_predictions.csv`：三个有标签 split 的预测分数。
- `outputs/tables/model_comparison_baseline_lgbm.csv`：Logit 与 LightGBM 同口径对比。
- `outputs/models/lgbm_model.txt`：LightGBM 模型文件。

## Top Gain 特征

| feature | gain_pct | split | group | univariate_auc_power |
|---|---:|---:|---|---:|
| `ext_source_mean` | 17.34% | 377 | current_application | 0.7166 |
| `ext_source_min` | 2.44% | 248 | current_application | 0.6912 |
| `ext_source_max` | 2.34% | 293 | current_application | 0.6858 |
| `credit_to_annuity_ratio` | 2.17% | 625 | current_application | 0.5322 |
| `employment_years` | 2.03% | 552 | current_application | 0.5831 |
| `EXT_SOURCE_3` | 1.61% | 367 | current_application | 0.6818 |
| `bureau_recent_24m_debt_to_credit_max` | 1.58% | 258 | external_credit_history | 0.6209 |
| `DAYS_BIRTH` | 1.36% | 487 | current_application | 0.5832 |
| `goods_to_credit_ratio` | 1.34% | 270 | current_application | 0.5675 |
| `AMT_ANNUITY` | 1.19% | 383 | current_application | 0.5017 |
| `EXT_SOURCE_1` | 1.11% | 367 | current_application | 0.6638 |
| `credit_card_recent_6m_utilization_max` | 1.06% | 128 | internal_credit_card_history | 0.6529 |
| `installment_payment_sum` | 1.01% | 254 | internal_repayment_history | 0.5336 |
| `bureau_credit_card_debt_to_credit_max` | 0.97% | 194 | external_credit_history | 0.6381 |
| `installment_recent_24m_late_days_mean` | 0.96% | 155 | internal_repayment_history | 0.5823 |

## 当前解读

- validation AUC 为 0.7904，KS 为 0.4395。
- final_holdout AUC 为 0.7960，KS 为 0.4494。
- best_iteration 为 1055。
- 下一步应把 LightGBM 分数转成 approve / manual review / decline，并在 final_holdout 上做策略评估。
