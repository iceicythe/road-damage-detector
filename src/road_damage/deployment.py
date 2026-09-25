"""Reproducible CPU pipeline parity and latency measurements on validation images."""
import json
import os
import random
from time import perf_counter

from .common import ROOT, environment, local_path, sha256, write_json


def compare_boxes(reference, candidate, coordinate_atol=.1, score_atol=1e-4):
    import numpy as np
    if len(reference) != len(candidate):
        return {'passed':False, 'reason':'box_count', 'reference':len(reference), 'candidate':len(candidate)}
    unused = set(range(len(candidate))); errors = []; scores = []
    for row in reference:
        possible = [j for j in unused if candidate[j, 5] == row[5]]
        if not possible: return {'passed':False, 'reason':'class_count'}
        index = min(possible, key=lambda j:float(np.abs(candidate[j, :4]-row[:4]).max()))
        unused.remove(index)
        errors.append(float(np.abs(candidate[index, :4]-row[:4]).max()))
        scores.append(float(abs(candidate[index, 4]-row[4])))
    pixel_error = max(errors, default=0); score_error = max(scores, default=0)
    return {'passed':pixel_error <= coordinate_atol and score_error <= score_atol,
            'max_coordinate_error_px':pixel_error, 'max_score_error':score_error}


def deploy_check(weights, onnx, name, samples=64, repeats=3, warmup=10, threads=4):
    import cv2
    import numpy as np
    import torch
    from .experiments import yolo_class
    from .onnx_inference import OnnxDetector, preprocess, postprocess
    if min(samples, repeats, warmup, threads) < 1: raise ValueError('样本、重复、预热和线程数必须为正数。')
    weights, onnx = local_path(weights), local_path(onnx)
    if not weights.is_file() or not onnx.is_file(): raise ValueError('PT 或 ONNX 权重不存在。')
    output = ROOT/'runs/deploy'/name
    if output.exists(): raise ValueError('部署报告目录已存在，请换一个 --name。')
    root = ROOT/'data/processed/rdd_v1'; manifest = root/'splits/val.jsonl'
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    rows = random.Random(42).sample(rows, min(samples, len(rows)))
    images = [cv2.imdecode(np.fromfile(root/r['prepared_image'], dtype=np.uint8), cv2.IMREAD_COLOR) for r in rows]
    if any(im is None for im in images): raise ValueError('无法读取部分验证图片。')
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    model = yolo_class()(str(weights), task='detect')
    # Initialize the official predictor before fixing the measurement thread count.
    model.predict(images[0], device='cpu', imgsz=640, rect=False, half=False, conf=.25, iou=.7, verbose=False)
    net = model.predictor.model.model.eval()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    runtime = OnnxDetector(onnx, threads)
    metadata = {'environment':environment(), 'cpu':os.environ.get('PROCESSOR_IDENTIFIER'),
        'weights_sha256':sha256(weights), 'onnx_sha256':sha256(onnx), 'val_manifest_sha256':sha256(manifest),
        'settings':{'device':'cpu','torch_threads':threads,'ort_threads':threads,'interop_threads':1,
                    'opencv_threads':1,'imgsz':640,'batch':1,'half':False,'conf':.25,'nms_iou':.7,
                    'samples':len(images),'repeats':repeats,'warmup':warmup,'seed':42},
        'providers':runtime.session.get_providers(),
        'timing_scope':'预解码 BGR 图片到 NumPy 检测框：resize/letterbox/归一化、模型前向、NMS/坐标还原。排除加载权重、文件读取、绘图、网页与导出。',
        'samples':[r['image'] for r in rows]}
    write_json(output/'metadata.json', metadata)

    def pt_forward(tensor):
        result = net(torch.from_numpy(tensor))
        return (result[0] if isinstance(result, (list, tuple)) else result).detach().cpu().numpy()

    def timed(image, forward):
        start = perf_counter(); tensor, geometry = preprocess(image); pre = perf_counter()
        raw = forward(tensor); infer = perf_counter()
        boxes = postprocess(raw, geometry); end = perf_counter()
        return boxes, {'preprocess_ms':(pre-start)*1000, 'inference_ms':(infer-pre)*1000,
                       'postprocess_ms':(end-infer)*1000, 'pipeline_ms':(end-start)*1000}

    parity = []; timings = []
    with torch.inference_mode():
        for i in range(warmup):
            timed(images[i % len(images)], pt_forward); timed(images[i % len(images)], runtime.forward)
        for index, (row, image) in enumerate(zip(rows, images), 1):
            reference = model.predict(image, device='cpu', imgsz=640, rect=False, half=False,
                                      conf=.25, iou=.7, max_det=300, verbose=False)[0].boxes.data.numpy()
            pt_boxes, _ = timed(image, pt_forward); ort_boxes, _ = timed(image, runtime.forward)
            parity.append({'image':row['image'], 'pt_standalone_vs_official':compare_boxes(reference, pt_boxes),
                           'onnx_vs_official_pt':compare_boxes(reference, ort_boxes)})
            if index % 16 == 0: print(f'Parity {index}/{len(images)}', flush=True)
        write_json(output/'parity.json', parity)
        # Interleave engines with deterministic randomized order to limit order bias.
        generator = random.Random(43)
        for repeat in range(repeats):
            for index, image in enumerate(images):
                engines = [('pt', pt_forward), ('onnx', runtime.forward)]; generator.shuffle(engines)
                for engine, forward in engines:
                    _, elapsed = timed(image, forward)
                    timings.append({'engine':engine,'repeat':repeat,'image':rows[index]['image'],**elapsed})
            print(f'Timing pass {repeat+1}/{repeats}', flush=True)
    write_json(output/'timings.json', timings)
    summary = {'settings':metadata['settings'], 'timing_scope':metadata['timing_scope'],
               'parity_tolerance':{'coordinate_atol_px':.1, 'score_atol':1e-4},
               'parity_samples':len(parity), 'pt_standalone_passed':sum(r['pt_standalone_vs_official']['passed'] for r in parity),
               'onnx_passed':sum(r['onnx_vs_official_pt']['passed'] for r in parity), 'engines':{}}
    for engine in ['pt','onnx']:
        summary['engines'][engine] = {}
        for key in ['preprocess_ms','inference_ms','postprocess_ms','pipeline_ms']:
            values = [t[key] for t in timings if t['engine']==engine]
            summary['engines'][engine][key] = {'median':float(np.median(values)), 'p95':float(np.percentile(values,95)),
                                               'mean':float(np.mean(values)), 'n':len(values)}
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
