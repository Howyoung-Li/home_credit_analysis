# 申请级特征集市初版

本文件由 `src/03_build_feature_store.py` 自动生成。

## 输出文件

- `data/processed/application_feature_store.parquet`：申请级宽表，356,255 行，545 列。
- `data/processed/application_feature_store_preview.csv`：前 1000 行预览。
- `data/processed/feature_manifest_initial.csv`：初始特征说明表。
- `outputs/tables/feature_store_group_profile.csv`：特征组覆盖率概览。

## 样本与标签

- 训练集申请数：307,511
- 训练集坏样本率：8.07%
- 测试集申请数：48,744

## 特征组

| feature_group | feature_count | avg_missing_rate |
|---|---:|---:|
| current_application | 144 | 21.16% |
| external_credit_history | 86 | 19.30% |
| external_credit_monthly_status | 33 | 34.41% |
| internal_previous_application | 83 | 22.89% |
| internal_repayment_history | 60 | 11.04% |
| internal_credit_card_history | 73 | 47.85% |
| internal_pos_cash_history | 55 | 13.48% |
| history_coverage | 6 | 0.00% |
| feature_store_quality | 2 | 0.00% |

## 业务使用方式

- 当前申请字段解释客户基本画像、申请金额、偿债压力、资料完整度和外部评分可得性。
- `bureau_*` 与 `bureau_balance_*` 对应外部征信历史，可用于判断多头借贷、历史逾期和活跃负债。
- `previous_*` 对应 Home Credit 历史申请，可解释历史拒贷率、通过率、渠道和产品偏好。
- `installment_*` 对应历史还款纪律，是贷前准入中最容易转成 reason code 的特征组。
- `credit_card_*` 和 `pos_*` 补充额度使用、月度余额、DPD 和剩余期数。

## 下一步

- 用该宽表训练一个可解释 baseline 模型。
- 输出 AUC、KS、Lift、Top 5%/10% 坏样本捕获率。
- 把模型分数转成 approve / manual review / decline 三段式准入策略。
