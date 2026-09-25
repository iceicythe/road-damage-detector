# ONNX 与留出测试验收

日期：2026-09-25。权重为 `runs/train/n640/weights/best.pt`，输入 640，FP32，batch=1。

## ONNX 精度对照

同一验证集和参数（`conf=0.001`、NMS IoU=0.7、`rect=False`、`half=False`）的 PT/ONNX 结果如下：

| 模型 | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| PT | 50.1894% | 45.3880% | 44.1874% | 19.3203% |
| ONNX | 50.1894% | 45.3880% | 44.1874% | 19.3203% |

原始指标差异低于 1e-6。结果位于 `runs/eval/n640_pt_deploy_val_v3/` 与 `runs/eval/n640_onnx_deploy_val_v2/`。

独立 ONNX Runtime CPU 管线不导入 PyTorch 或 Ultralytics，执行 letterbox、归一化、原始输出解析、按类别 NMS 和坐标还原。在固定随机种子的 64 张验证图片上，与官方 PT 输出逐框对比 64/64 通过，与 ONNX 输出逐框对比也是 64/64 通过；坐标误差阈值 0.1 像素，置信度误差阈值 1e-4。

## CPU 延迟

`runs/deploy/n640_cpu_deploy_v2/summary.json` 使用 64 张图片、5 次预热、2 轮交错测量、4 个线程。范围从已解码 BGR 图片开始，包含预处理、模型前向、NMS 和坐标还原，不含文件读取、权重加载、绘图、网页和导出。

| 引擎 | 端到端中位数 | P95 | 前向中位数 |
|---|---:|---:|---:|
| PyTorch CPU | 66.49 ms/图 | 76.19 ms/图 | 62.27 ms |
| ONNX Runtime CPU | 39.71 ms/图 | 44.72 ms/图 | 35.54 ms |

这是 i7-12650H、4 线程、FP32 的可复现实验，不代表 GPU、其他 CPU 或完整网页服务的延迟。

## 筛查后的留出测试

`prepare-holdout` 从原 test 排除与 train/val 命中的 65 张已知相似候选，冻结 1,672 张测试图片。固定 n640 权重、未使用本次成绩选模型，得到：

| 图片 | Precision | Recall | mAP50 | mAP50–95 |
|---:|---:|---:|---:|---:|
| 1,672 | 51.25% | 43.78% | 43.27% | 18.65% |

结果在 `runs/eval/n640_screened_test_v1/summary.json`，冻结规则在 `data/processed/rdd_screened_test_v1/evaluation_policy.json`。因为没有完整道路/序列 ID，这仍不能称为完全道路独立测试。

## 复现

```powershell
& "D:\CONDA\envs\pytorch314\python.exe" run.py deploy-check --samples 64 --repeats 2 --warmup 5 --threads 4 --name n640_cpu_deploy_v3
& "D:\CONDA\envs\pytorch314\python.exe" run.py prepare-holdout
& "D:\CONDA\envs\pytorch314\python.exe" run.py evaluate --weights runs/train/n640/weights/best.pt --data data/processed/rdd_screened_test_v1/dataset.yaml --split test --device 0 --name n640_screened_test_v2
```
