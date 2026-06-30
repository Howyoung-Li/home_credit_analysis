# A 卡评分与准入策略初版

本文件由 `src/07_evaluate_strategy.py` 自动生成。

## Test 口径说明

- `final_holdout`：从 `application_train` 内部留出的有标签 test，可计算 AUC、KS、Lift、坏样本捕获等模型指标。
- `external_unlabeled`：Kaggle 官方 `application_test`，没有 `TARGET`，只能输出预测 PD、A 卡分数、风险等级和策略动作，不能计算真实 AUC/KS。
- `final_holdout` 和官方 test 都没有参与特征筛选、模型训练和 early stopping。

## A 卡刻度

- Base score：600 分，对应 good:bad odds = 20:1。
- PDO：50，坏账 odds 翻倍时分数下降 50 分。
- 分数方向：分数越高，预测违约风险越低。
- 当前策略：validation 风险最高 10% 拒绝，之后 10% 人工复核，其余自动通过。

## 内部有标签 Test 指标

| split | rows | bad_rate | AUC | KS | top_5pct_bad_capture | top_10pct_bad_capture | score_PSI_vs_dev |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 184507 | 8.07% | 0.9134 | 0.6607 | 42.20% | 61.80% | 0.0000 |
| final_holdout | 61502 | 8.07% | 0.7960 | 0.4494 | 23.93% | 38.11% | 0.0006 |
| validation | 61502 | 8.07% | 0.7904 | 0.4395 | 23.99% | 39.23% | 0.0008 |

## 策略动作表现

| split | action | population_pct | bad_rate | bad_capture | avg_score |
|---|---|---:|---:|---:|---:|
| validation | approve | 80.00% | 4.26% | 42.26% | 630.0 |
| validation | decline | 10.00% | 31.67% | 39.23% | 443.8 |
| validation | manual_review | 10.00% | 14.94% | 18.51% | 510.5 |
| final_holdout | approve | 79.95% | 4.23% | 41.89% | 630.5 |
| final_holdout | decline | 10.06% | 30.74% | 38.29% | 443.8 |
| final_holdout | manual_review | 9.99% | 16.01% | 19.82% | 510.7 |

## 官方 Test 预测概览

- 官方 test 行数：48,744
- 平均预测 PD：7.08%
- PD P50/P95：3.95% / 24.42%
- Score P50/P05：614.0 / 465.4

## 输出文件

- `outputs/tables/a_card_internal_test_metrics.csv`：内部有标签 test 指标。
- `outputs/tables/a_card_score_band_metrics.csv`：分数等级表现。
- `outputs/tables/a_card_strategy_action_metrics.csv`：approve / manual_review / decline 表现。
- `outputs/tables/a_card_strategy_scenarios.csv`：不同拒绝率/复核率情景对比。
- `outputs/tables/a_card_external_test_predictions.csv`：官方 test 预测 PD、分数、等级和策略动作。
- `outputs/tables/a_card_kaggle_submission.csv`：Kaggle submission 格式。
- `outputs/tables/a_card_scored_population.csv`：全量样本评分明细。

## 当前业务解读

- final_holdout AUC 为 0.7960，KS 为 0.4494。
- final_holdout Top 10% 高风险申请捕获 38.11% 坏样本。
- 当前策略阈值只是 baseline 策略，后续可以根据审批通过率、人工复核产能和误拒成本调整。
