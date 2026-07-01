# Home Credit 贷前信用风险项目

在线 dashboard：https://howyoung-li.github.io/home_credit_analysis/

GitHub 仓库：https://github.com/Howyoung-Li/home_credit_analysis

目标：从 Home Credit 多表原始数据出发，构建一个更接近真实风控工作的项目，而不是只拿现成宽表建模。

本项目可以参考 Kaggle 1st place solution 的多表聚合、特征工程和 LightGBM 思路，但项目主线不是刷榜，而是证明：

- 能理解贷前准入业务问题。
- 能把原始多表信贷数据整理成可审计的申请级风险数据资产。
- 能区分审批时点可用信息和贷后表现信息，避免信息泄露。
- 能把模型分数转化为 approve / manual review / decline 准入策略。
- 能解释模型为什么拦截这些申请，以及策略代价是什么。

## 项目主线

1. 业务问题定义：贷前准入、人工复核产能、误拒与漏放权衡。
2. 数据盘点与质量审计：每张表的粒度、主键、外键、缺失与异常。
3. 审批时点定义：哪些字段在申请时可见，哪些字段属于未来表现。
4. 多表聚合与申请级特征集市：客户历史授信、征信、还款、额度使用、历史拒绝/通过等。
5. 基准模型与主模型：Logit baseline + LightGBM，不追求复杂堆叠优先。
6. 准入策略：approve / manual review / decline 三段式策略。
7. 解释与监控：SHAP 理由码、分群表现、PSI 漂移监控、策略复盘。
8. 输出简历可用项目总结、业务报告和面试讲述稿。

## 目录

- `data/raw/`: 放原始 Kaggle CSV，不做修改。
- `data/interim/`: 放清洗后的中间表。
- `data/processed/`: 放最终建模宽表和特征字典。
- `notebooks/`: 放探索分析 notebook。
- `src/`: 放可复跑脚本。
- `docs/`: 放数据字典、防泄露清单、业务口径说明。
- `reports/`: 放最终项目报告。
- `outputs/models/`: 放模型文件。
- `outputs/figures/`: 放图表。
- `outputs/tables/`: 放指标表、策略表、特征重要性表。

## 可执行顺序

原始 CSV 已放入 `data/raw/` 后，先运行：

```bash
python src/00_profile_raw_data.py
python src/01_build_data_dictionary.py
python src/02_clean_tables.py
python src/03_build_feature_store.py
python src/04_screen_candidate_features.py
python src/05_train_baseline.py
python src/06_train_lgbm.py
python src/07_evaluate_strategy.py
python src/08_explain_monitor.py
python src/09_visualize_strategy.py
```

当前已规划/生成的核心输出：

- `outputs/tables/raw_table_profile.csv`
- `outputs/tables/raw_column_profile_sample.csv`
- `docs/raw_data_profile.md`
- `data/interim/application_base.parquet`
- `docs/data_cleaning_rules.md`
- `data/processed/application_feature_store.parquet`
- `data/processed/feature_manifest_initial.csv`
- `docs/feature_store_initial.md`
- `data/processed/candidate_feature_matrix.parquet`
- `outputs/tables/feature_screening_report.csv`
- `outputs/tables/feature_screening_shortlist.csv`
- `outputs/tables/feature_selection_evidence_table.csv`
- `outputs/tables/feature_correlation_pairs_ge_0p80.csv`
- `outputs/tables/feature_correlation_prune_decisions.csv`
- `docs/feature_screening_initial.md`
- `outputs/tables/baseline_logit_metrics.csv`
- `outputs/tables/baseline_logit_lift_table.csv`
- `outputs/tables/baseline_logit_coefficients.csv`
- `docs/baseline_model_initial.md`
- `outputs/tables/lgbm_metrics.csv`
- `outputs/tables/lgbm_lift_table.csv`
- `outputs/tables/lgbm_feature_importance.csv`
- `outputs/tables/lgbm_topk_precision_recall.csv`
- `outputs/tables/model_comparison_baseline_lgbm.csv`
- `docs/lgbm_model_initial.md`
- `outputs/tables/a_card_internal_test_metrics.csv`
- `outputs/tables/a_card_external_test_predictions.csv`
- `outputs/tables/a_card_kaggle_submission.csv`
- `docs/a_card_strategy_initial.md`
- `outputs/tables/strategy_threshold_curve.csv`
- `outputs/tables/strategy_profit_curve_summary.csv`
- `outputs/tables/strategy_cost_assumptions.csv`
- `outputs/tables/strategy_cost_sensitivity_curve.csv`
- `outputs/tables/strategy_cost_optimal_thresholds.csv`
- `outputs/tables/strategy_amount_cost_assumptions.csv`
- `outputs/tables/strategy_amount_weighted_cost_curve.csv`
- `outputs/tables/strategy_amount_weighted_optimal_thresholds.csv`
- `outputs/tables/shap_global_importance.csv`
- `outputs/tables/shap_reason_code_summary.csv`
- `docs/strategy_profit_shap_initial.md`
- `docs/interview_data_feature_deep_dive.md`
- `reports/strategy_visual_dashboard.html`
- `outputs/figures/01_model_validation_metrics.svg`
- `outputs/figures/02_strategy_action_profile.svg`
- `outputs/figures/03_lift_decile_curve.svg`
- `outputs/figures/04_topk_precision_recall.svg`
- `outputs/figures/04_amount_cost_heatmap.svg`
- `outputs/figures/05_amount_cost_scenarios.svg`
- `outputs/figures/06_shap_top_features.svg`
- `outputs/figures/07_decline_reason_codes.svg`
- `docs/visualization_initial.md`
