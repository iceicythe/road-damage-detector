# 实验产物

已完成的真实验证集实验：

- [n640 错误分析](n640-error-review.md)
- [置信度阈值对比](n640-thresholds.md)
- [训练数据复核](train-data-audit.md)
- [D10 重采样结果](n640-d10x2-results.md)
- [跨集合相似性审计](split-similarity.md)
- [本地巡检功能验收](inspection-acceptance.md)
- [ONNX 与留出测试验收](deployment-validation.md)

旧划分存在近重复候选，已有指标仅作为当前图片划分下的历史结果，不声称道路独立泛化性能。

已完成排除已知相似候选的留出测试和 PT/ONNX 部署验收；完整道路/序列独立测试仍待完成。模型权重、逐图预测和原始数据保留在被忽略的 models/、data/、runs/ 中。

