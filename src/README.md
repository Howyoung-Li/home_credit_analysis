# 脚本规划

后续建议按这个顺序创建脚本：

- `00_profile_raw_data.py`: 原始表盘点。
- `01_build_data_dictionary.py`: 字段字典、申请时点可用性与初始泄露风险标记。
- `02_clean_tables.py`: 清洗申请主表、处理特殊值、生成申请底座和业务比率字段。
- `03_build_feature_store.py`: 多表聚合成申请级特征集市，并输出初始特征说明。
- `04_screen_candidate_features.py`: 生成类别编码候选特征，并按缺失、稳定性、AUC/KS/IV、相关性去重筛选。
- `05_train_baseline.py`: 纯 numpy L2 Logit baseline，输出 AUC/KS/Lift/TopK、预测分数和标准化系数。
- `06_train_lgbm.py`: LightGBM 主模型，输出 AUC/KS/Lift/TopK、特征重要性和与 Logit 的同口径对比。
- `07_evaluate_strategy.py`: A 卡分数刻度、风险等级、approve/manual_review/decline 策略、内部 test 指标和官方 test 预测。
- `08_explain_monitor.py`: 策略阈值收益曲线、break-even 分析、SHAP 原因码与后续监控证据。
- `09_visualize_strategy.py`: 生成 SVG 图表、HTML dashboard 和可视化说明文档。
