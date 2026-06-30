# 原始数据盘点

本报告由 `src/00_profile_raw_data.py` 自动生成。

## 表级概览

| table | rows | columns | file_mb | key_columns | bad_rate |
|---|---:|---:|---:|---|---:|
| application_train | 307511 | 122 | 158.44 | SK_ID_CURR | 8.07% |
| application_test | 48744 | 121 | 25.34 | SK_ID_CURR |  |
| bureau | 1716428 | 17 | 162.14 | SK_ID_CURR, SK_ID_BUREAU |  |
| bureau_balance | 27299925 | 3 | 358.19 | SK_ID_BUREAU |  |
| previous_application | 1670214 | 37 | 386.21 | SK_ID_PREV, SK_ID_CURR |  |
| POS_CASH_balance | 10001358 | 8 | 374.51 | SK_ID_PREV, SK_ID_CURR |  |
| credit_card_balance | 3840312 | 23 | 404.91 | SK_ID_PREV, SK_ID_CURR |  |
| installments_payments | 13605401 | 8 | 689.62 | SK_ID_PREV, SK_ID_CURR |  |
| HomeCredit_columns_description | 219 | 5 | 0.04 |  |  |

## 初步业务解读

- `application_train` 是当前申请主表，`TARGET` 是贷后表现标签。
- 其他表均为历史行为或外部征信数据，需要聚合到 `SK_ID_CURR` 申请粒度。
- 后续建模前必须完成防泄露审计，确保特征在申请时点可见。
- 项目评估应同时报告模型指标和准入策略指标。
