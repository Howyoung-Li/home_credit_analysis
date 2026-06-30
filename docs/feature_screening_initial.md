# 候选特征筛选初版

本文件由 `src/04_screen_candidate_features.py` 自动生成。

## 方法

- 从申请级特征集市读取候选变量。
- 数值变量保留原始数值形态。
- 在 `application_train` 内部按标签分层切出 development / validation / final_holdout。
- 类别变量只用 development 拟合频率编码和 top 类别 one-hot。
- 先做 target-blind 质量筛选：缺失率、唯一值、development/validation PSI。
- 再只用 development 标签做 target-aware 单变量筛选：AUC power >= 0.515、KS >= 0.020、IV >= 0.005。
- 单变量门槛设置为宽松口径，目的是保留近期窗口和分组聚合变量，让 LightGBM 继续判断非线性与交互价值。
- 最后对单变量候选做高相关去重，避免重复变量堆叠。
- `final_holdout` 和官方 Kaggle test 不参与本阶段任何筛选。

## 样本切分

| dataset | screening_split | rows | bad_rate |
|---|---|---:|---:|
| test | external_unlabeled | 48744 |  |
| train | development | 184507 | 8.07% |
| train | final_holdout | 61502 | 8.07% |
| train | validation | 61502 | 8.07% |

## 产出概览

- 候选矩阵行数：356,255
- 候选特征数：674
- 质量筛选通过：672
- 单变量筛选通过：468
- 相关性去重后短名单：368

## 证据表清单

- `outputs/tables/feature_screening_report.csv`：全量候选变量筛选报告。
- `outputs/tables/feature_selection_evidence_table.csv`：面试证据总表，含筛选阶段、缺失率、PSI、AUC、KS、IV、相关性去重原因。
- `outputs/tables/feature_correlation_pairs_ge_0p80.csv`：development 样本中绝对相关系数 >= 0.80 的变量对。
- `outputs/tables/feature_correlation_prune_decisions.csv`：因相关性 >= 0.95 被去重剔除的变量及其对应保留变量。
- `outputs/tables/feature_screening_shortlist.csv`：最终统计短名单。

## 分组结果

| feature_group | candidate_count | quality_pass | univariate_pass | final_shortlist | median_auc_power | median_iv | median_psi_dev_val |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_application | 276 | 276 | 146 | 91 | 0.5075 | 0.0000 | 0.0000 |
| internal_previous_application | 83 | 81 | 71 | 65 | 0.5233 | 0.0149 | 0.0001 |
| external_credit_history | 86 | 86 | 73 | 64 | 0.5390 | 0.0288 | 0.0001 |
| internal_repayment_history | 60 | 60 | 54 | 52 | 0.5491 | 0.0297 | 0.0002 |
| internal_pos_cash_history | 55 | 55 | 48 | 39 | 0.5169 | 0.0043 | 0.0001 |
| internal_credit_card_history | 73 | 73 | 54 | 36 | 0.5116 | 0.0026 | 0.0000 |
| external_credit_monthly_status | 33 | 33 | 19 | 19 | 0.5104 | 0.0010 | 0.0000 |
| feature_store_quality | 2 | 2 | 2 | 1 | 0.5025 | 0.0058 | 0.0001 |
| history_coverage | 6 | 6 | 1 | 1 | 0.5072 | 0.0000 | 0.0000 |

## Top 30 统计短名单

| feature | original_feature | encoding | group | auc_power | ks | iv | psi_dev_val | missing |
|---|---|---|---|---:|---:|---:|---:|---:|
| `ext_source_mean` | `ext_source_mean` | numeric_original | current_application | 0.7166 | 0.3250 | 0.6095 | 0.0002 | 0.06% |
| `ext_source_min` | `ext_source_min` | numeric_original | current_application | 0.6912 | 0.2827 | 0.4679 | 0.0003 | 0.06% |
| `ext_source_max` | `ext_source_max` | numeric_original | current_application | 0.6858 | 0.2713 | 0.4410 | 0.0002 | 0.06% |
| `EXT_SOURCE_3` | `EXT_SOURCE_3` | numeric_original | current_application | 0.6818 | 0.2752 | 0.3386 | 0.0002 | 19.80% |
| `EXT_SOURCE_2` | `EXT_SOURCE_2` | numeric_original | current_application | 0.6544 | 0.2199 | 0.3005 | 0.0003 | 0.21% |
| `EXT_SOURCE_1` | `EXT_SOURCE_1` | numeric_original | current_application | 0.6638 | 0.2424 | 0.1452 | 0.0002 | 56.42% |
| `credit_card_recent_6m_utilization_max` | `credit_card_recent_6m_utilization_max` | numeric_original | internal_credit_card_history | 0.6529 | 0.2707 | 0.0796 | 0.0000 | 77.87% |
| `bureau_credit_card_debt_to_credit_max` | `bureau_credit_card_debt_to_credit_max` | numeric_original | external_credit_history | 0.6381 | 0.2276 | 0.1217 | 0.0001 | 51.42% |
| `credit_card_recent_12m_utilization_mean` | `credit_card_recent_12m_utilization_mean` | numeric_original | internal_credit_card_history | 0.6384 | 0.2425 | 0.0673 | 0.0001 | 76.77% |
| `credit_card_recent_24m_utilization_max` | `credit_card_recent_24m_utilization_max` | numeric_original | internal_credit_card_history | 0.6381 | 0.2388 | 0.0729 | 0.0001 | 74.34% |
| `bureau_credit_card_debt_to_credit_mean` | `bureau_credit_card_debt_to_credit_mean` | numeric_original | external_credit_history | 0.6285 | 0.2109 | 0.1052 | 0.0002 | 51.42% |
| `bureau_recent_24m_debt_to_credit_max` | `bureau_recent_24m_debt_to_credit_max` | numeric_original | external_credit_history | 0.6209 | 0.1912 | 0.1292 | 0.0002 | 35.94% |
| `credit_card_utilization_mean` | `credit_card_utilization_mean` | numeric_original | internal_credit_card_history | 0.6281 | 0.1994 | 0.0621 | 0.0001 | 72.06% |
| `credit_card_recent_24m_balance_mean` | `credit_card_recent_24m_balance_mean` | numeric_original | internal_credit_card_history | 0.6090 | 0.1995 | 0.0481 | 0.0000 | 71.77% |
| `bureau_credit_card_total_debt_to_credit_ratio` | `bureau_credit_card_total_debt_to_credit_ratio` | numeric_original | external_credit_history | 0.6107 | 0.1765 | 0.0845 | 0.0002 | 47.78% |
| `credit_card_recent_6m_balance_mean` | `credit_card_recent_6m_balance_mean` | numeric_original | internal_credit_card_history | 0.6042 | 0.2033 | 0.0419 | 0.0001 | 71.77% |
| `credit_card_drawings_mean` | `credit_card_drawings_mean` | numeric_original | internal_credit_card_history | 0.6069 | 0.1886 | 0.0444 | 0.0003 | 71.77% |
| `bureau_days_credit_mean` | `bureau_days_credit_mean` | numeric_original | external_credit_history | 0.6030 | 0.1565 | 0.1219 | 0.0002 | 14.29% |
| `credit_card_recent_12m_balance_max` | `credit_card_recent_12m_balance_max` | numeric_original | internal_credit_card_history | 0.6013 | 0.1953 | 0.0463 | 0.0002 | 71.77% |
| `bureau_recent_12m_debt_to_credit_max` | `bureau_recent_12m_debt_to_credit_max` | numeric_original | external_credit_history | 0.6075 | 0.1655 | 0.0932 | 0.0002 | 54.59% |
| `credit_card_utilization_max` | `credit_card_utilization_max` | numeric_original | internal_credit_card_history | 0.6071 | 0.1783 | 0.0424 | 0.0002 | 72.06% |
| `bureau_total_debt_to_credit_ratio` | `bureau_total_debt_to_credit_ratio` | numeric_original | external_credit_history | 0.5907 | 0.1414 | 0.1011 | 0.0002 | 14.65% |
| `bureau_active_loan_ratio` | `bureau_active_loan_ratio` | numeric_original | external_credit_history | 0.5836 | 0.1367 | 0.0892 | 0.0001 | 14.29% |
| `installment_recent_24m_late_days_mean` | `installment_recent_24m_late_days_mean` | numeric_original | internal_repayment_history | 0.5823 | 0.1412 | 0.0830 | 0.0003 | 18.06% |
| `employment_years` | `employment_years` | numeric_original | current_application | 0.5831 | 0.1323 | 0.1143 | 0.0001 | 18.04% |
| `installment_recent_24m_late_days_max` | `installment_recent_24m_late_days_max` | numeric_original | internal_repayment_history | 0.5814 | 0.1402 | 0.0807 | 0.0001 | 18.06% |
| `installment_recent_24m_late_ratio` | `installment_recent_24m_late_ratio` | numeric_original | internal_repayment_history | 0.5794 | 0.1415 | 0.0752 | 0.0003 | 18.06% |
| `DAYS_BIRTH` | `DAYS_BIRTH` | numeric_original | current_application | 0.5832 | 0.1225 | 0.0853 | 0.0001 | 0.00% |
| `previous_cash_refusal_rate` | `previous_cash_refusal_rate` | numeric_original | internal_previous_application | 0.5831 | 0.1479 | 0.0565 | 0.0003 | 51.41% |
| `installment_recent_12m_late_days_max` | `installment_recent_12m_late_days_max` | numeric_original | internal_repayment_history | 0.5783 | 0.1421 | 0.0659 | 0.0002 | 29.32% |

## 如何避免先射箭再画靶

- 业务解释不参与初始候选是否生成；候选由数据源、聚合口径和编码规则系统产生。
- 第一轮剔除规则提前固定，先看 development/validation 质量和稳定性，再看 development 单变量区分度。
- `final_holdout` 不参与筛选，留给后续模型最终评估、PSI 诊断和漂移变量 ablation。
- 官方 Kaggle test 是无标签外部样本，不参与任何特征选择或模型选择。
- 只有通过统计筛选的变量才进入后续业务 reason code 解释。
- 下一轮模型筛选会继续用交叉验证、LightGBM 特征重要性和 permutation/null importance 验证。
