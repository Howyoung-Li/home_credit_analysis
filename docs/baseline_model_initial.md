# Baseline Logit 模型初版

本文件由 `src/05_train_baseline.py` 自动生成。

## 建模口径

- 使用 `feature_screening_shortlist.csv` 中的统计短名单变量。
- 只在 `development` 上拟合预处理参数和模型参数。
- 使用 `validation` 观察泛化表现。
- `final_holdout` 不参与训练、筛选或调参，只做最终只读评估。
- 类别变量已在 `04_screen_candidate_features.py` 中转成频率编码或 one-hot。
- 数值变量做 1%/99% 截尾、development 中位数填补、标准化；缺失率 >= 0.1% 的变量额外生成缺失指示器。
- Logit 使用类别平衡权重训练以提升排序；训练后只用 development 做截距校准，使平均预测 PD 对齐 development 坏样本率。

## 核心指标

| split | rows | bad_rate | AUC | KS | top_5pct_bad_capture | top_10pct_bad_capture | score_PSI_vs_dev |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 184507 | 8.07% | 0.7843 | 0.4283 | 22.30% | 36.96% | 0.0000 |
| validation | 61502 | 8.07% | 0.7772 | 0.4177 | 21.95% | 37.34% | 0.0002 |
| final_holdout | 61502 | 8.07% | 0.7857 | 0.4345 | 22.05% | 36.25% | 0.0001 |

## 输出文件

- `outputs/tables/baseline_logit_metrics.csv`：development / validation / final_holdout 指标。
- `outputs/tables/baseline_logit_lift_table.csv`：十分位 Lift 和累计坏样本捕获。
- `outputs/tables/baseline_logit_coefficients.csv`：标准化系数和风险方向。
- `outputs/tables/baseline_logit_predictions.csv`：三个有标签 split 的预测分数。
- `outputs/models/baseline_logit_model.npz`：模型参数和预处理参数。

## Top 标准化系数

| model_feature | direction | coef | group | univariate_auc_power |
|---|---|---:|---|---:|
| `EXT_SOURCE_3` | lower_risk | -0.1763 | current_application | 0.6818 |
| `EXT_SOURCE_2` | lower_risk | -0.1619 | current_application | 0.6544 |
| `catfreq__CODE_GENDER` | lower_risk | -0.1498 | current_application | 0.5461 |
| `employment_years` | lower_risk | -0.1407 | current_application | 0.5831 |
| `ext_source_mean` | lower_risk | -0.1262 | current_application | 0.7166 |
| `goods_to_credit_ratio` | lower_risk | -0.1179 | current_application | 0.5675 |
| `previous_annuity_mean` | lower_risk | -0.1075 | internal_previous_application | 0.5361 |
| `AMT_ANNUITY` | higher_risk | 0.0991 | current_application | 0.5017 |
| `ext_source_min` | lower_risk | -0.0966 | current_application | 0.6912 |
| `EXT_SOURCE_1` | lower_risk | -0.0960 | current_application | 0.6638 |
| `annuity_to_income_ratio` | higher_risk | 0.0957 | current_application | 0.5203 |
| `ext_source_max` | lower_risk | -0.0824 | current_application | 0.6858 |

## 当前解读

- validation AUC 为 0.7772，KS 为 0.4177。
- final_holdout AUC 为 0.7857，KS 为 0.4345。
- 这个 baseline 的价值不是最终性能，而是建立可复现的训练、验证、最终只读评估和风控指标口径。
- 下一步可以训练 LightGBM，并在模型冻结后做 final_holdout PSI 和高漂移变量 ablation。
