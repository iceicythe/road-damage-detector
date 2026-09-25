"""Validation-only error review; fixed-threshold diagnostics, not an AP evaluator."""
from pathlib import Path
import html
import json
import random
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from .common import CLASSES, NAMES, ROOT, environment, local_path, sha256, write_json

CODES = list(CLASSES)


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match(gt, predictions, threshold=0.5):
    """Confidence-ordered, same-class, one-to-one matching. No AP claim."""
    matched_gt, matched_pred, pairs = set(), set(), []
    for p in sorted(range(len(predictions)), key=lambda i: predictions[i]['conf'], reverse=True):
        candidates = [(overlap(predictions[p]['box'], g['box']), i) for i, g in enumerate(gt)
                      if i not in matched_gt and g['class'] == predictions[p]['class']]
        if candidates:
            score, g = max(candidates)
            if score >= threshold:
                matched_gt.add(g)
                matched_pred.add(p)
                pairs.append([g, p, score])
    return {'matches': pairs, 'fn': [i for i in range(len(gt)) if i not in matched_gt],
            'fp': [i for i in range(len(predictions)) if i not in matched_pred]}


def read_labels(path, width, height):
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        c, x, y, w, h = map(float, line.split())
        if c != int(c) or int(c) not in range(4):
            raise ValueError(f'Unexpected class: {path}')
        result.append({'class': int(c), 'box': [(x-w/2)*width, (y-h/2)*height,
                                               (x+w/2)*width, (y+h/2)*height]})
    return result


def audit_annotation(record, gt, raw, origin):
    """Independent inverse-conversion comparison against original XML."""
    xml = raw / record['annotation']
    root = ET.parse(xml).getroot()
    expected = []
    for obj in root.findall('object'):
        box = [float(obj.findtext('bndbox/' + k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
        if origin == 'voc1':
            box[0] -= 1
            box[1] -= 1
        expected.append({'class': CLASSES[obj.findtext('name').strip()], 'box': box})
    issues = []
    if sha256(xml) != record['annotation_sha256']:
        issues.append('raw_annotation_changed')
    if (int(root.findtext('size/width')), int(root.findtext('size/height'))) != (record['width'], record['height']):
        issues.append('xml_size_mismatch')
    if len(expected) != len(gt):
        issues.append('box_count_mismatch')
    elif any(a['class'] != b['class'] or any(abs(x-y) > 0.05 for x,y in zip(a['box'], b['box']))
             for a,b in zip(expected, gt)):
        issues.append('class_or_coordinate_mismatch')
    return issues


def diagnose(gt, predictions, matched, threshold):
    # These are hints for human inspection, not proven causes. A FN can have both hints.
    localization, confusion = [], []
    for i in matched['fn']:
        for j in matched['fp']:
            score = overlap(gt[i]['box'], predictions[j]['box'])
            if gt[i]['class'] == predictions[j]['class'] and 0.1 <= score < threshold:
                localization.append([i,j,score])
            elif gt[i]['class'] != predictions[j]['class'] and score >= threshold:
                confusion.append([i,j,score])
    return {'localization_candidates': localization, 'class_confusion_candidates': confusion}


def select_cases(records, count, seed):
    rng = random.Random(seed)
    selected = {}
    def take(pool, n, reason):
        pool = [r for r in pool if r['image'] not in selected]
        rng.shuffle(pool)
        for r in pool[:n]:
            selected[r['image']] = (r, reason)
    take(records, min(20, count), '随机抽样')
    take([r for r in records if any(r['gt'][i]['class'] == 1 for i in r['fn'])],
         min(10, count-len(selected)), '横向裂缝漏检')
    take([r for r in records if r['localization_candidates']], min(10, count-len(selected)), '定位偏差候选')
    take([r for r in records if not r['gt'] and r['fp']], min(10, count-len(selected)), '无标注图片上的检出')
    take(records, count-len(selected), '补充随机抽样')
    return list(selected.values())


def comparison(record, path):
    with Image.open(record['image']) as im:
        original = im.convert('RGB')
    original.thumbnail((720, 720))
    w,h = original.size
    sx,sy = w/record['width'], h/record['height']
    canvas = Image.new('RGB', (w*2, h+32), 'white')
    canvas.paste(original, (0,32)); canvas.paste(original, (w,32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8,8), 'GROUND TRUTH | red = missed', fill='black')
    draw.text((w+8,8), 'PREDICTIONS | red = unmatched', fill='black')
    for side, boxes, unmatched in [(0,record['gt'],record['fn']), (1,record['predictions'],record['fp'])]:
        for i,b in enumerate(boxes):
            x1,y1,x2,y2 = b['box']
            box = [x1*sx+side*w, y1*sy+32, x2*sx+side*w, y2*sy+32]
            color = '#e53935' if i in unmatched else '#00a050'
            draw.rectangle(box, outline=color, width=3)
            label = CODES[b['class']] + (f" {b['conf']:.2f}" if side else '')
            tx,ty = max(side*w,box[0]), max(32,box[1]-14)
            draw.rectangle((tx,ty,tx+len(label)*7+4,ty+14), fill=color)
            draw.text((tx+2,ty+1),label,fill='white')
    canvas.save(path, quality=90)


def review(weights, data, device, name, imgsz=640, conf=0.25, match_iou=0.5, count=50):
    from .experiments import data_identity, yolo_class
    if not 0 < conf < 1 or not 0 < match_iou <= 1 or count < 1:
        raise ValueError('需要 0<conf<1、0<match-iou<=1、count>=1。')
    weights, data = local_path(weights), local_path(data)
    if not weights.is_file():
        raise ValueError(f'模型不存在：{weights}')
    root = data.parent
    identity = data_identity(data)
    audit = json.loads((root/'audit.json').read_text(encoding='utf-8'))
    raw = Path(audit['raw_root'])
    manifest = [json.loads(s) for s in (root/'splits/val.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if not manifest or any(r['split'] != 'val' for r in manifest):
        raise ValueError('仅支持非空验证集清单。')
    output = ROOT/'runs/review'/name
    if output.exists():
        raise ValueError(f'目录已存在，请更换 --name：{output}')
    model = yolo_class()(str(weights))
    if list(model.names.values()) != NAMES:
        raise ValueError('模型类别顺序不匹配。')
    output.mkdir(parents=True)
    (output/'images').mkdir()
    records, conversion_issues = [], []
    lookup = {str((root/r['prepared_image']).resolve()): r for r in manifest}
    sources = output/'sources.txt'
    sources.write_text('\n'.join(lookup)+'\n', encoding='utf-8')
    predictions = model.predict(source=str(sources), stream=True, imgsz=imgsz, batch=8,
                                device=device, conf=conf, iou=0.7, max_det=300, rect=False,
                                verbose=False, save=False)
    with (output/'predictions.jsonl').open('w', encoding='utf-8') as handle:
        for index,result in enumerate(predictions,1):
            path = str(Path(result.path).resolve())
            r = lookup[path]
            gt = read_labels(root/'labels/val'/ (Path(path).stem+'.txt'), r['width'],r['height'])
            issues = audit_annotation(r,gt,raw,audit['coordinate_origin'])
            if tuple(result.orig_shape) != (r['height'],r['width']):
                issues.append('actual_image_size_mismatch')
            if issues:
                conversion_issues.append({'image':r['image'],'issues':issues})
            p = [{'class':int(b[5]), 'box':b[:4], 'conf':b[4]} for b in result.boxes.data.cpu().tolist()]
            matched = match(gt,p,match_iou)
            record = {'image':path,'source_image':r['image'],'source':r['source'],
                      'width':r['width'],'height':r['height'],'gt':gt,'predictions':p,
                      **matched, **diagnose(gt,p,matched,match_iou)}
            records.append(record)
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            if index % 200 == 0:
                handle.flush()
                print(f'Review inference: {index}/{len(manifest)}',flush=True)
    if len(records) != len(manifest):
        raise RuntimeError('推理结果数量与清单不一致。')
    by_class = []
    for c,code in enumerate(CODES):
        tp = sum(sum(r['gt'][i]['class']==c for i,_,_ in r['matches']) for r in records)
        fn = sum(sum(r['gt'][i]['class']==c for i in r['fn']) for r in records)
        fp = sum(sum(r['predictions'][i]['class']==c for i in r['fp']) for r in records)
        by_class.append({'class':code,'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
                         'recall':tp/(tp+fn) if tp+fn else None})
    summary = {'weights':str(weights),'weights_sha256':sha256(weights),'data':identity,
               'environment':environment(),'images':len(records),'imgsz':imgsz,'conf':conf,
               'match_iou':match_iou,'nms_iou':0.7,'batch':8,'rect':False,'max_det':300,
               'coordinate_origin':audit['coordinate_origin'],'annotation_issues':conversion_issues,
               'per_class':by_class,'images_with_fn':sum(bool(r['fn']) for r in records),
               'images_with_fp':sum(bool(r['fp']) for r in records),
               'unlabeled_images':sum(not r['gt'] for r in records),
               'unlabeled_images_with_predictions':sum(not r['gt'] and bool(r['fp']) for r in records),
               'fn_with_localization_hint':sum(len({x[0] for x in r['localization_candidates']}) for r in records),
               'fn_with_class_confusion_hint':sum(len({x[0] for x in r['class_confusion_candidates']}) for r in records),
               'note':'固定阈值、按置信度同类贪心一对一匹配；不等同于 mAP 或框架的最佳 F1 选点。定位/错分类仅为非互斥候选原因。无标注不保证没有可见病害。'}
    write_json(output/'summary.json',summary)
    selected = select_cases(records,min(count,len(records)),42)
    cards, checklist = [], []
    for i,(r,reason) in enumerate(selected,1):
        filename = f'{i:02d}_{Path(r["image"]).stem}.jpg'
        comparison(r,output/'images'/filename)
        caption = f'{i:02d} {r["source_image"]} | {reason} | TP={len(r["matches"])} FN={len(r["fn"])} FP={len(r["fp"])}'
        cards.append(f'<article><h3>{html.escape(caption)}</h3><a href="images/{filename}"><img loading="lazy" src="images/{filename}"></a></article>')
        checklist.append({'id':i,'image':r['source_image'],'selection':reason,'comparison':'images/'+filename,
                          'human_review':'pending','cause':None,'notes':''})
    write_json(output/'review_checklist.json',checklist)
    table = ''.join(f"<tr><td>{r['class']}</td><td>{r['tp']}</td><td>{r['fp']}</td><td>{r['fn']}</td><td>{r['precision']:.1%}</td><td>{r['recall']:.1%}</td></tr>" for r in by_class)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>道路病害验证集错误复核</title>
<style>body{{font:16px system-ui;background:#f4f6f8;color:#182432;margin:30px auto;max-width:1480px;padding:0 20px}}article,header{{background:white;padding:20px;margin:20px 0;border-radius:12px}}img{{width:100%;height:auto}}td,th{{padding:8px 22px;border-bottom:1px solid #ddd}}p{{line-height:1.7}}h3{{font-size:16px}}</style>
<header><h1>道路病害：验证集错误复核</h1><p>全量统计：{len(records)} 张。conf={conf}，匹配 IoU={match_iou}，NMS IoU=0.7。左侧原始标注，右侧模型预测；绿色表示匹配，红色表示未匹配。D00 纵向裂缝 / D10 横向裂缝 / D20 网状裂缝 / D40 坑洞。</p>
<table><tr><th>类别</th><th>匹配 TP</th><th>未匹配预测 FP</th><th>漏检 FN</th><th>Precision</th><th>Recall</th></tr>{table}</table>
<p>XML 与 YOLO 反算坐标检查：{len(conversion_issues)} 张存在异常。此检查不证明原始人工标注完整或类别正确，沿用数据准备的坐标起点假设。</p>
<p>以下 {len(selected)} 张是随机样本和定向错误样本的混合，用于人工复核，不用于估计错误占比。无标注图上的检出既可能是误报，也可能是漏标，需要查看原图。统计不等同于 mAP。</p></header>{''.join(cards)}</html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f'Review report: {output / "index.html"}',flush=True)
