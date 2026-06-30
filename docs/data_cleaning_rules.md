# 申请主表清洗口径

本文件由 `src/02_clean_tables.py` 自动生成。

## 清洗规则

- 将 `DAYS_*` 字段中的特殊值 `365243` 视为缺失，并保留对应哨兵标记字段。
- 将 `XNA`、`Unknown` 统一视为未知/缺失类别，避免把无效类别当作稳定业务分群。
- 不修改 `data/raw/` 原始 CSV；清洗后的申请底座输出到 `data/interim/application_base.parquet`。
- `TARGET` 只作为训练标签保留，不参与特征衍生。

## 衍生业务字段

- `applicant_age_years`
- `employment_years`
- `registration_age_years`
- `id_publish_years`
- `last_phone_change_years`
- `credit_to_income_ratio`
- `annuity_to_income_ratio`
- `credit_to_annuity_ratio`
- `goods_to_credit_ratio`
- `income_per_family_member`
- `children_to_family_ratio`
- `employment_to_age_ratio`
- `ext_source_mean`
- `ext_source_min`
- `ext_source_max`
- `ext_source_std`
- `ext_source_available_count`
- `document_flag_count`
- `contact_channel_count`
- `bureau_request_total_count`
- `bureau_request_recent_count`
- `application_missing_field_count`
- `application_missing_field_ratio`

## 训练集标签概览

- 样本数：307,511
- 坏样本数：24,825
- 坏样本率：8.07%

## 业务解释

- 申请主表特征主要解释本次申请条件、客户基本偿债能力、信息完整度和外部评分可得性。
- `credit_to_income_ratio`、`annuity_to_income_ratio` 对应授信压力和月供压力。
- `application_missing_field_ratio` 可作为资料完整度或数据质量的代理变量。
- 后续历史表特征会补充征信、历史申请、还款纪律和额度使用情况。
