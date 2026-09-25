"""YOLO11 detection on CPU using OpenCV, NumPy and ONNX Runtime only."""
import ast
from pathlib import Path

import cv2
import numpy as np

from .common import NAMES, ROOT, local_path, sha256, write_json


def preprocess(image, size=640):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('需要 HWC 格式的三通道 BGR 图片。')
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = (round(width * ratio), round(height * ratio))
    dw, dh = (size - resized[0]) / 2, (size - resized[1]) / 2
    left, top = round(dw - .1), round(dh - .1)
    if resized != (width, height):
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, round(dh + .1), left, round(dw + .1),
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, (ratio, left, top, width, height)


def nms(boxes, scores, classes, threshold=.7, max_det=300):
    """Class-aware greedy suppression, returning original row indexes."""
    order = np.argsort(-scores, kind='stable')[:30000]
    keep = []
    while len(order) and len(keep) < max_det:
        index = order[0]; keep.append(int(index)); order = order[1:]
        if not len(order): break
        lt = np.maximum(boxes[index, :2], boxes[order, :2])
        rb = np.minimum(boxes[index, 2:], boxes[order, 2:])
        inter = np.maximum(rb - lt, 0).prod(axis=1)
        area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
        iou = inter / np.maximum(area[index] + area[order] - inter, 1e-7)
        order = order[(classes[order] != classes[index]) | (iou <= threshold)]
    return np.array(keep, dtype=np.int64)


def postprocess(raw, geometry, conf=.25, iou=.7, max_det=300):
    if not 0 < conf < 1 or not 0 < iou <= 1 or max_det < 1:
        raise ValueError('conf、iou 或 max_det 无效。')
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 8:
        raise ValueError(f'需要四类 YOLO11 原始输出 (1,8,N)，实际为 {raw.shape}。')
    rows = raw[0].T
    if not np.isfinite(rows).all(): raise ValueError('模型输出包含非有限值。')
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    selected = scores > conf
    rows, scores, classes = rows[selected], scores[selected], classes[selected]
    if not len(rows): return np.empty((0, 6), dtype=np.float32)
    boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2,
                            rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    keep = nms(boxes, scores, classes, iou, max_det)
    boxes, scores, classes = boxes[keep].copy(), scores[keep], classes[keep]
    ratio, left, top, width, height = geometry
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / ratio, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / ratio, 0, height)
    return np.column_stack((boxes, scores, classes)).astype(np.float32)


class OnnxDetector:
    def __init__(self, weights, threads=4):
        import onnxruntime as ort
        if threads < 1: raise ValueError('线程数必须为正数。')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(weights), sess_options=options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('仅支持单输入、单输出的 YOLO11 检测模型。')
        inp = inputs[0]
        if inp.type != 'tensor(float)' or inp.shape != [1, 3, 640, 640]:
            raise ValueError(f'仅支持固定 1×3×640×640 FP32 输入，实际为 {inp.type} {inp.shape}。')
        metadata = self.session.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(metadata.get('names', '{}'))
            if isinstance(names, dict): names = [names[k] for k in sorted(names)]
        except (ValueError, SyntaxError, TypeError): names = None
        if names != NAMES: raise ValueError('ONNX 类别元数据与四类道路病害定义不一致。')
        self.input_name = inp.name
        self.output_name = outputs[0].name

    def forward(self, tensor):
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

    def predict(self, image, conf=.25, iou=.7):
        tensor, geometry = preprocess(image)
        return postprocess(self.forward(tensor), geometry, conf, iou)


def onnx_predict(weights, source, name, conf=.25, threads=4):
    weights, source = local_path(weights), local_path(source)
    if not weights.is_file() or not source.is_file(): raise ValueError('请提供模型和一张图片的有效路径。')
    output = ROOT/'runs/predict'/name
    if output.exists(): raise ValueError('输出目录已存在，请换一个 --name。')
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError('无法读取图片。')
    model = OnnxDetector(weights, threads)
    boxes = model.predict(image, conf)
    output.mkdir(parents=True)
    for x1, y1, x2, y2, score, category in boxes:
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 220, 255), 2)
        cv2.putText(image, f'{NAMES[int(category)]} {score:.2f}', (round(x1), max(15, round(y1)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok: raise RuntimeError('输出图片编码失败。')
    encoded.tofile(output/'prediction.jpg')
    write_json(output/'predictions.json', {'source':str(source), 'source_sha256':sha256(source),
        'weights_sha256':sha256(weights), 'provider':model.session.get_providers(),
        'conf':conf, 'iou':.7, 'imgsz':640, 'box_format':'x1,y1,x2,y2,confidence,class_id',
        'boxes':boxes.tolist(), 'coordinate_note':'坐标基于 OpenCV 解码后的图片方向。'})
    print(f'独立 ONNX 检测完成：{output}')
