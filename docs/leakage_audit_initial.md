# 初始防泄露审计

本文件由 `src/01_build_data_dictionary.py` 生成。

## 核心原则

- `TARGET` 是贷后表现标签，只能用于训练监督信号，不能进入特征。
- `SK_ID_*` 只用于连接和聚合，不作为模型业务特征。
- `application_train/test` 中除标签和主键外，初步视为申请时点可见字段。
- 历史表字段初步视为历史可见，但必须在特征工程中确认它们发生在当前申请之前。
- 还款、余额、历史申请等表需要聚合到 `SK_ID_CURR` 申请粒度。

## 初始泄露风险汇总

| data_domain | leakage_risk_initial | columns |
|---|---|---:|
| current_application | high | 1 |
| current_application | low | 242 |
| external_credit_history | low | 3 |
| external_credit_history | medium | 17 |
| internal_credit_card_history | low | 2 |
| internal_credit_card_history | medium | 21 |
| internal_pos_cash_history | low | 2 |
| internal_pos_cash_history | medium | 6 |
| internal_previous_application | low | 2 |
| internal_previous_application | medium | 35 |
| internal_repayment_history | low | 2 |
| internal_repayment_history | medium | 6 |

## 下一步人工复核

- 逐项检查 `DAYS_*` 字段是否相对当前申请日，避免使用当前申请后的表现。
- 对历史还款、信用卡、POS余额表设置历史窗口，保留窗口口径。
- 对缺失率极高字段决定是否作为缺失信号、分箱变量或剔除。
- 对 `EXT_SOURCE_*` 作为外部评分单独解释，避免黑箱依赖。
