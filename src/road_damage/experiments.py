from pathlib import Path
import json

from .common import ROOT, NAMES, environment, local_path, sha256, write_json


def yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('请先安装训练依赖：python -m pip install -e ".[train]"') from exc
    return YOLO


def train_config(preset, data=None, device=None):
    import yaml
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    presets = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    if preset not in presets:
        raise ValueError(f"未知实验：{preset}，可用：{', '.join(presets)}")
    config.update(presets[preset])
    config["data"] = str(local_path(data or config["data"]))
    config["project"] = str(local_path(config["project"]))
    if device is not None:
        config["device"] = device
    return config


def data_identity(data):
    import yaml
    data = Path(data)
    if not data.is_file():
        raise ValueError(f"缺少数据配置：{data}。先运行 prepare。")
    result = {"dataset_yaml": sha256(data)}
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", data.parent))
    if not dataset_root.is_absolute():
        dataset_root = data.parent / dataset_root
    for split in ["train", "val", "test"]:
        entry = config.get(split)
        if isinstance(entry, str):
            path = Path(entry)
            path = path if path.is_absolute() else dataset_root / path
            if path.suffix == ".txt" and path.is_file():
                result[split + "_list"] = sha256(path)
    sampling = data.parent / "sampling.json"
    if sampling.is_file():
        result["sampling_metadata"] = sha256(sampling)
    policy = data.parent / 'evaluation_policy.json'
    if policy.is_file(): result['evaluation_policy'] = sha256(policy)
    for split in ["train", "val", "test"]:
        manifest = data.parent / "splits" / f"{split}.jsonl"
        if manifest.is_file():
            result[split + "_manifest"] = sha256(manifest)
    return result


def train(preset, data=None, device=None, dry_run=False, name=None):
    config = train_config(preset, data, device)
    if name:
        if '/' in name or '\\' in name or name in {'.', '..'}:
            raise ValueError('实验 name 只能是单层目录名。')
        config['name'] = name
    if dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    identity = data_identity(config["data"])
    destination = Path(config["project"]) / config["name"]
    if destination.exists():
        raise ValueError(f"实验目录已存在，请修改 preset 的 name，避免覆盖：{destination}")
    YOLO = yolo_class()
    model = YOLO(config["model"])
    destination.mkdir(parents=True)
    write_json(destination / "run_metadata.json", {"requested_config": config, "data": identity, "environment": environment()})
    try:
        model.train(**config, exist_ok=True)
    except Exception as exc:
        write_json(destination / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def evaluate(weights, data, split, imgsz, device, name):
    weights = local_path(weights)
    if not weights.is_file():
        raise ValueError(f"模型文件不存在：{weights}")
    identity = data_identity(local_path(data))
    policy_file = local_path(data).parent/'evaluation_policy.json'
    if policy_file.is_file():
        policy = json.loads(policy_file.read_text(encoding='utf-8'))
        if (sha256(weights) != policy['weights_sha256'] or split != policy['split'] or imgsz != policy['imgsz']
            or identity.get('test_manifest') != policy['test_manifest_sha256']
            or identity.get('test_list') != policy['test_list_sha256']):
            raise ValueError('权重、集合、输入尺寸或测试名单与冻结的留出测试方案不一致。')
    destination = ROOT / "runs" / "eval" / name
    if destination.exists():
        raise ValueError(f"评测目录已存在，请指定新 --name：{destination}")
    model = yolo_class()(str(weights), task="detect")
    metrics = model.val(data=str(local_path(data)), split=split, imgsz=imgsz, batch=1,
                        device=device, workers=0, project=str(destination.parent), name=name,
                        exist_ok=False, plots=True, save_json=True, rect=False,
                        conf=0.001, iou=0.7, half=False, max_det=300, augment=False)
    # Use the actual framework path instead of assuming naming behavior.
    save_dir = Path(metrics.save_dir)
    by_class = []
    for row, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(row)
        by_class.append({"id": int(class_id), "name": NAMES[int(class_id)],
                         "precision": float(p), "recall": float(r), "map50": float(ap50), "map50_95": float(ap)})
    write_json(save_dir / "summary.json", {"weights": str(weights), "weights_sha256": sha256(weights),
        "split": split, "imgsz": imgsz, "batch": 1, "device": device,
        "inference": {"rect": False, "conf": 0.001, "nms_iou": 0.7, "half": False, "max_det": 300},
        "metrics": {str(k): float(v) for k, v in metrics.results_dict.items()},
        "per_class": by_class, "framework_speed_ms": dict(metrics.speed),
        "speed_note": "框架验证阶段计时，不等于包含文件读取/渲染/导出的完整延迟；P/R 沿用框架选点口径。",
        "data": identity, "environment": environment()})
    print(f"评测已保存：{save_dir}")


def export(weights, imgsz, device):
    weights = local_path(weights)
    if not weights.is_file():
        raise ValueError(f"模型文件不存在：{weights}")
    if weights.with_suffix(".onnx").exists():
        raise ValueError("同名 ONNX 已存在，请换用另一个模型目录，避免覆盖。")
    model = yolo_class()(str(weights), task="detect")
    exported = Path(model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                                 half=False, simplify=False, nms=False, opset=17, device=device))
    write_json(exported.with_suffix(".export.json"), {"source": str(weights),
        "source_sha256": sha256(weights), "onnx_sha256": sha256(exported),
        "imgsz": imgsz, "batch": 1, "half": False, "dynamic": False, "opset": 17, "nms": False,
        "postprocess": "导出原始输出。可使用 Ultralytics 或 standalone ONNX Runtime 入口进行前后处理。",
        "environment": environment()})
    print(f"已导出：{exported}。请在同一验证集对比 PT 与 ONNX，未自动宣称精度/速度等价。")


def predict(weights, source, imgsz, device, conf, name):
    weights, source = local_path(weights), local_path(source)
    if not weights.is_file() or not source.exists():
        raise ValueError("模型或输入文件不存在。source 支持本地图片、视频或目录。")
    destination = ROOT / "runs" / "predict" / name
    if destination.exists():
        raise ValueError(f"输出已存在，请换一个 --name：{destination}")
    model = yolo_class()(str(weights), task="detect")
    for _ in model.predict(source=str(source), imgsz=imgsz, device=device, conf=conf,
                           stream=True, save=True, save_txt=True, save_conf=True,
                           project=str(destination.parent), name=name, exist_ok=False):
        pass
    print(f"预测输出：{destination}")

