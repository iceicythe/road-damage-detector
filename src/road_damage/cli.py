import argparse
import json

from .common import environment, local_path


def main():
    parser = argparse.ArgumentParser(description="道路病害检测：数据准备、训练、评测和部署入口")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="检查依赖版本；不下载模型")
    prepare = commands.add_parser("prepare", help="RDD2022 XML 检查、转换、划分及预览")
    prepare.add_argument("--raw", default="data/raw/RDD2022")
    prepare.add_argument("--out", default="data/processed/rdd_v1")
    prepare.add_argument("--coordinate-origin", choices=["voc1", "zero"], required=True,
                         help="人工确认：voc1=1 起点含端点；zero=0 起点 xyxy")
    prepare.add_argument("--groups", help="可选 CSV：image（相对 raw 的路径）,group；需覆盖所有标注图片")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--preview", type=int, default=24)
    train = commands.add_parser("train", help="运行训练 preset")
    train.add_argument("--preset", choices=["smoke", "n640", "n960", "s640", "n640_d10x2"], default="smoke")
    train.add_argument("--data")
    train.add_argument("--device")
    train.add_argument("--name", help="覆盖配置中的实验名，必须使用新名称避免覆盖已有结果")
    train.add_argument("--dry-run", action="store_true", help="仅显示配置；不导入 Ultralytics、不训练")
    review = commands.add_parser("review", help="验证集固定阈值错误统计、XML 核对及对照图")
    review.add_argument("--weights", default="runs/train/n640/weights/best.pt")
    review.add_argument("--data", default="data/processed/rdd_v1/dataset.yaml")
    review.add_argument("--device", default="0")
    review.add_argument("--name", required=True)
    review.add_argument("--imgsz", type=int, default=640)
    review.add_argument("--conf", type=float, default=0.25)
    review.add_argument("--match-iou", type=float, default=0.5)
    review.add_argument("--count", type=int, default=50)
    sweep = commands.add_parser("thresholds", help="从低阈值验证集缓存比较多个置信度阈值")
    sweep.add_argument("--cache", required=True)
    sweep.add_argument("--name", required=True)
    sweep.add_argument("--confs", nargs="+", type=float, default=[0.10, 0.25, 0.40])
    sweep.add_argument("--reference", help="可选的原 review 目录，用于验证匹配结果一致性")
    audit = commands.add_parser("train-audit", help="训练集类别/来源分布、XML 转换核对和定向样本复核")
    audit.add_argument("--name", required=True)
    audit.add_argument("--device", default="0")
    dup = commands.add_parser("duplicates", help="跨集合感知哈希近重复候选筛查")
    dup.add_argument("--name", required=True)
    commands.add_parser("prepare-sampling", help="生成仅训练集 D10 图像重复两次的采样配置")
    commands.add_parser('prepare-holdout', help='冻结排除已知跨集合相似候选的留出测试名单')
    standalone = commands.add_parser('onnx-predict', help='独立 ONNX Runtime CPU 单图检测，不导入 PyTorch/Ultralytics')
    standalone.add_argument('--weights', default='runs/train/n640/weights/best.onnx')
    standalone.add_argument('--source', required=True)
    standalone.add_argument('--name', required=True)
    standalone.add_argument('--conf', type=float, default=.25)
    standalone.add_argument('--threads', type=int, default=4)
    deploy = commands.add_parser('deploy-check', help='CPU PT/ONNX 输出一致性及完整推理链路耗时')
    deploy.add_argument('--weights', default='runs/train/n640/weights/best.pt')
    deploy.add_argument('--onnx', default='runs/train/n640/weights/best.onnx')
    deploy.add_argument('--name', required=True)
    deploy.add_argument('--samples', type=int, default=64)
    deploy.add_argument('--repeats', type=int, default=3)
    deploy.add_argument('--warmup', type=int, default=10)
    deploy.add_argument('--threads', type=int, default=4)
    evaluate = commands.add_parser("evaluate", help="验证或最终测试，默认 val")
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--data", default="data/processed/rdd_v1/dataset.yaml")
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--name", required=True)
    export = commands.add_parser("export", help="导出固定输入尺寸 FP32 ONNX")
    export.add_argument("--weights", required=True)
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--device", default="cpu")
    predict = commands.add_parser("predict", help="本地图片/视频推理")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--device", default="0")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--name", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            print(json.dumps(environment(), ensure_ascii=False, indent=2))
        elif args.command in {'onnx-predict','deploy-check'}:
            if args.command == 'onnx-predict':
                from .onnx_inference import onnx_predict as run
            else:
                from .deployment import deploy_check as run
            kwargs = vars(args).copy(); kwargs.pop('command'); run(**kwargs)
        elif args.command == "prepare-sampling":
            from .sampling import prepare_sampling
            prepare_sampling()
        elif args.command == 'prepare-holdout':
            from .holdout import prepare_holdout
            prepare_holdout()
        elif args.command == "duplicates":
            from .duplicates import scan_duplicates
            scan_duplicates(args.name)
        elif args.command == "prepare":
            from .data import prepare as run
            if args.preview < 0:
                raise ValueError("--preview 不能为负数。")
            run(local_path(args.raw), local_path(args.out), args.coordinate_origin, args.seed,
                local_path(args.groups) if args.groups else None, args.preview)
        elif args.command in {"review", "thresholds", "train-audit"}:
            if args.command == "review":
                from .review import review as run
            elif args.command == "thresholds":
                from .thresholds import thresholds as run
            else:
                from .train_audit import train_audit as run
            kwargs = vars(args).copy()
            kwargs.pop("command")
            run(**kwargs)
        else:
            from . import experiments
            kwargs = vars(args).copy()
            command = kwargs.pop("command")
            getattr(experiments, command)(**kwargs)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"错误：{exc}\n")

