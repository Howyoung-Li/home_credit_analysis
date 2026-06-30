# Kaggle 方案如何借鉴，如何改造

参考链接：

https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution

## 可以借鉴的部分

- 多表数据都要用起来，而不是只用 `application_train`。
- 特征工程比单纯调参更重要。
- 将不同粒度的数据聚合到 `SK_ID_CURR` 申请级。
- LightGBM 适合作为强基线模型。
- 需要严格的训练/验证切分和稳定性评估。

## 本项目不照搬的部分

- 不以 leaderboard AUC 为唯一目标。
- 不优先做复杂 stacking/blending。
- 不堆大量难解释特征而忽略业务含义。
- 不把模型输出停留在概率分，而要转成贷前准入策略。

## 本项目的改造重点

1. 数据理解优先：解释每张表是什么业务来源、什么粒度、什么时点可见。
2. 防泄露优先：把当前申请后才知道的信息排除在特征外。
3. 策略优先：输出 approve / manual review / decline。
4. 解释优先：每个拒绝或复核建议要能给 reason code。
5. 监控优先：上线后要关注分数分布、特征 PSI、分群坏样本率。

## 面试表达

可以这样说：

> 我参考了 Kaggle 第一名方案中多表聚合和 LightGBM 建模的思路，但我没有把项目做成刷榜模型。我把重点放在贷前准入业务：先定义申请时点和历史观察窗口，再做多表数据治理、防泄露审计、申请级特征集市，最后把模型分数转成通过、人工复核和拒绝策略，并输出 SHAP reason code 和 PSI 监控框架。

