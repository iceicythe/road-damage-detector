"""Compare operating thresholds on one cached validation inference run."""
from pathlib import Path
import html
import json
import random

from .common import ROOT, local_path, sha256, write_json
from .review import CODES, comparison, match


def score(records, conf, match_iou):
    counts = [[0, 0, 0] for _ in CODES]  # TP, FP, FN
    rows = []
    for r in records:
        predictions = [p for p in r['predictions'] if p['conf'] >= conf]
        matches = match(r['gt'], predictions, match_iou)
        for i, _, _ in matches['matches']:
            counts[r['gt'][i]['class']][0] += 1
        for i in matches['fp']:
            counts[predictions[i]['class']][1] += 1
        for i in matches['fn']:
            counts[r['gt'][i]['class']][2] += 1
        rows.append({**r, 'predictions': predictions, **matches})

    def metrics(tp, fp, fn):
        return {'tp': tp, 'fp': fp, 'fn': fn,
                'precision': tp/(tp+fp) if tp+fp else 0.0,
                'recall': tp/(tp+fn) if tp+fn else 0.0,
                'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}

    return {'conf': conf, 'overall_micro': metrics(*(sum(c[i] for c in counts) for i in range(3))),
            'per_class': [{'class': code, **metrics(*c)} for code, c in zip(CODES, counts)],
            'unlabeled_images_with_predictions': sum(not r['gt'] and bool(r['predictions']) for r in rows)}, rows


def thresholds(cache, name, confs=(0.1, 0.25, 0.4), reference=None):
    cache = local_path(cache)
    metadata = json.loads((cache/'summary.json').read_text(encoding='utf-8'))
    confs = sorted(set(confs))
    if not confs or any(not metadata['conf'] <= t < 1 for t in confs):
        raise ValueError('比较阈值必须 >= 缓存推理阈值且 <1；低于缓存阈值需重新推理。')
    records = [json.loads(s) for s in (cache/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records) != metadata['images'] or len({r['image'] for r in records}) != len(records):
        raise ValueError('缓存不完整或有重复图片。')
    results, views = [], {}
    for conf in confs:
        summary, rows = score(records, conf, metadata['match_iou'])
        results.append(summary)
        views[conf] = rows
    check = None
    if reference:
        refpath = local_path(reference)
        ref = json.loads((refpath/'summary.json').read_text(encoding='utf-8'))
        for key in ['weights_sha256', 'data', 'images', 'imgsz', 'batch', 'rect', 'nms_iou', 'match_iou', 'max_det']:
            if metadata[key] != ref[key]:
                raise ValueError(f'参考运行配置不一致：{key}')
        selected = next((r for r in results if r['conf'] == ref['conf']), None)
        if selected is None:
            raise ValueError('比较阈值未包含参考阈值。')
        old = [json.loads(s) for s in (refpath/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        by_image = {r['image']: r for r in old}
        changed = []
        for row in views[ref['conf']]:
            prior = by_image.get(row['image'])
            if prior is None or row['gt'] != prior['gt'] or any(row[k] != prior[k] for k in ['fn', 'fp', 'matches']):
                changed.append(row['source_image'])
        check = {'reference': str(refpath), 'changed_matching_images': len(changed), 'examples': changed[:10]}
        if changed:
            raise ValueError(f'重新推理与原阈值匹配结果不同：{len(changed)} 张；需先调查，不能直接合并对比。')
    output = ROOT/'runs/review'/name
    if output.exists():
        raise ValueError(f'目录已存在：{output}')
    output.mkdir(parents=True)
    (output/'images').mkdir()
    report = {'cache': str(cache), 'predictions_sha256': sha256(cache/'predictions.jsonl'),
              'inference': metadata, 'reference_check': check, 'thresholds': results,
              'note': '同一批 NMS 后预测按置信度筛选并重新匹配。micro P/R/F1；不是 mAP，不改变模型权重。FP 相对于原始标注统计。阈值仅在验证集探索。'}
    write_json(output/'thresholds.json', report)
    # Record recovered GT identities; distinguish newly recovered targets from net counts.
    base = 0.25 if 0.25 in views else confs[-1]
    low = confs[0]
    changes = []
    for a,b in zip(views[low], views[base]):
        recovered = sorted(set(b['fn'])-set(a['fn']))
        changes.append({'image': a['source_image'], 'recovered_gt': recovered,
                        'recovered_classes': [CODES[a['gt'][i]['class']] for i in recovered],
                        'fp_increase': len(a['fp'])-len(b['fp'])})
    write_json(output/'changes.json', {'low_conf': low, 'reference_conf': base, 'images': changes})
    lines = ['# n640 置信度阈值对比', '',
             f"验证集 {len(records)} 张；imgsz={metadata['imgsz']}，匹配 IoU={metadata['match_iou']}，NMS IoU={metadata['nms_iou']}。",
             '', report['note'], '', '## 全类别 micro 指标', '',
             '| 阈值 | TP | FP | FN | Precision | Recall | F1 | 空标注图上有预测的图片数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m = r['overall_micro']
        lines.append(f"| {r['conf']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {r['unlabeled_images_with_predictions']} |")
    lines += ['', '## 各类别', '', '| 类别 | 阈值 | TP | FP | FN | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in range(4):
        for r in results:
            m = r['per_class'][c]
            lines.append(f"| {m['class']} | {r['conf']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |")
    lines += ['', '## 可复现检查', '', f'原 0.25 运行匹配结果核对：{check}',
              '', f'缓存达到 max_det 的图片数：{sum(len(r["predictions"]) >= metadata["max_det"] for r in records)}。',
              '', '结论只适用于这些阈值和该验证集。未做测试集推理，未根据预测修改标注。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Six examples, chosen reproducibly; three recovered D10 and three extra-FP images.
    rng = random.Random(42)
    rescued = [i for i,r in enumerate(changes) if 'D10' in r['recovered_classes']]
    rng.shuffle(rescued)
    examples = rescued[:3]
    extra = [i for i,r in enumerate(changes) if r['fp_increase'] > 0 and i not in examples]
    rng.shuffle(extra)
    examples += extra[:3]
    cards = []
    for i in examples:
        options = []
        for t in confs:
            filename = f'{i:04d}_conf{t:.2f}.jpg'
            comparison(views[t][i], output/'images'/filename)
            options.append(f'<figure><figcaption>conf={t:.2f}</figcaption><img loading="lazy" src="images/{filename}"></figure>')
        cards.append(f'<article><h2>{html.escape(records[i]["source_image"])}</h2><p>从 {base} 降至 {low}：找回 {len(changes[i]["recovered_gt"])} 个目标，新增 {changes[i]["fp_increase"]} 个未匹配预测。</p>{"".join(options)}</article>')
    table = ''.join(f'<tr><td>{r["conf"]:.2f}</td><td>{r["overall_micro"]["precision"]:.1%}</td><td>{r["overall_micro"]["recall"]:.1%}</td><td>{r["overall_micro"]["fp"]}</td></tr>' for r in results)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>置信度阈值对比</title>
<style>body{{font:16px system-ui;background:#f4f6f8;max-width:1450px;margin:30px auto;padding:20px}}article,header{{background:white;padding:20px;margin:20px 0;border-radius:12px}}img{{width:100%}}figure{{margin:15px 0}}td,th{{padding:10px 25px}}p{{line-height:1.7}}</style>
<header><h1>n640：降低阈值的收益与代价</h1><p>{html.escape(report['note'])} 左侧为标注，右侧为预测；绿框匹配、红框未匹配。以下 6 张为定向样例，不能用来估计总体比例。</p><table><tr><th>阈值</th><th>Precision</th><th>Recall</th><th>FP</th></tr>{table}</table></header>{''.join(cards)}</html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({'thresholds':results,'reference_check':check,'report':str(output/'report.md')},ensure_ascii=False,indent=2))
