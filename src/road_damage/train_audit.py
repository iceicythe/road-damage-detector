"""Train-only stratified audit. In-sample predictions rank cases, not model quality."""
from collections import Counter
from pathlib import Path
import html
import json
import random

from PIL import Image, ImageDraw

from .common import ROOT, CLASSES, NAMES, local_path, sha256, write_json
from .review import read_labels, audit_annotation, match, comparison


def train_audit(name, device='0'):
    from .experiments import yolo_class, data_identity
    root = ROOT/'data/processed/rdd_v1'
    weights = ROOT/'runs/train/n640/weights/best.pt'
    output = ROOT/'runs/review'/name
    if output.exists():
        raise ValueError(f'目录已存在：{output}')
    rows = [json.loads(s) for s in (root/'splits/train.jsonl').read_text(encoding='utf-8').splitlines()]
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('只允许训练集')
    settings = json.loads((root/'audit.json').read_text(encoding='utf-8'))
    raw = Path(settings['raw_root'])
    stats = {}
    candidates, reasons = {}, {}
    rng = random.Random(42)
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == source]
        empty = [r for r in group if not r['labels']]
        d10 = [r for r in group if any(b[0] == 1 for b in r['labels'])]
        boxes = Counter(int(b[0]) for r in group for b in r['labels'])
        def short_side(r):
            scale = 640/max(r['width'],r['height'])
            return min(min(b[3]*r['width'],b[4]*r['height'])*scale for b in r['labels'] if b[0]==1)
        thin = sorted(d10,key=short_side)[:4]
        random_d10 = rng.sample([r for r in d10 if r not in thin], min(4,max(0,len(d10)-4)))
        sampled_bg = rng.sample(empty,min(200,len(empty)))
        for r in thin+random_d10+sampled_bg:
            path = str((root/r['prepared_image']).resolve())
            candidates[path] = r
            if r in thin:
                reasons[path] = f'{source} D10 最细框候选'
            elif r in random_d10:
                reasons[path] = f'{source} D10 随机'
        all_short = sorted(min(b[3]*r['width'],b[4]*r['height'])*640/max(r['width'],r['height'])
                           for r in d10 for b in r['labels'] if b[0]==1)
        stats[source] = {'images':len(group),'empty_images':len(empty),
                         'empty_fraction':len(empty)/len(group),'d10_images':len(d10),
                         'boxes':{code:boxes[c] for code,c in CLASSES.items()},
                         'd10_short_side_at_640':{'minimum':min(all_short),
                             'median':all_short[len(all_short)//2], 'below_8px':sum(v<8 for v in all_short)}}
    output.mkdir(parents=True)
    (output/'images').mkdir()
    (output/'sheets').mkdir()
    issues = []
    for i,r in enumerate(rows,1):
        path = root/'labels/train'/(Path(r['prepared_image']).stem+'.txt')
        gt = read_labels(path,r['width'],r['height'])
        found = audit_annotation(r,gt,raw,settings['coordinate_origin'])
        if found:
            issues.append({'image':r['image'],'issues':found})
        if i%2000==0:
            print(f'Annotation audit {i}/{len(rows)}',flush=True)
    sourcefile = output/'sources.txt'
    sourcefile.write_text('\n'.join(candidates)+'\n',encoding='utf-8')
    model = yolo_class()(str(weights))
    if list(model.names.values()) != NAMES:
        raise ValueError('类别不匹配')
    results = []
    for pred in model.predict(source=str(sourcefile),stream=True,imgsz=640,batch=8,rect=False,
                              device=device,conf=.25,iou=.7,max_det=300,verbose=False):
        key = str(Path(pred.path).resolve())
        r = candidates[key]
        gt = read_labels(root/'labels/train'/(Path(key).stem+'.txt'),r['width'],r['height'])
        p = [{'class':int(b[5]),'box':b[:4],'conf':b[4]} for b in pred.boxes.data.cpu().tolist()]
        results.append({'image':key,'source_image':r['image'],'source':r['source'],
                        'width':r['width'],'height':r['height'],'gt':gt,'predictions':p,
                        **match(gt,p), 'selection':reasons.get(key)})
    if len(results) != len(candidates):
        raise RuntimeError('推理数量不完整')
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    selected = [r for r in results if r['selection']]
    for source in stats:
        bg = [r for r in results if r['source']==source and not r['gt']]
        ranked = sorted([r for r in bg if r['predictions']],key=lambda r:max(p['conf'] for p in r['predictions']),reverse=True)[:4]
        random_bg = rng.sample([r for r in bg if r not in ranked],min(4,len(bg)-len(ranked)))
        for r in ranked:
            r['selection'] = f'{source} 空标注高置信检出'
        for r in random_bg:
            r['selection'] = f'{source} 空标注随机（排除已选高置信样本）'
        selected.extend(ranked+random_bg)
        stats[source]['sampled_empty_images'] = len(bg)
        stats[source]['sampled_empty_with_predictions'] = sum(bool(r['predictions']) for r in bg)
    notes = []
    cards = []
    panels = []
    for i,r in enumerate(selected,1):
        stem = f'{i:02d}_{Path(r["image"]).stem}'
        comparison(r,output/'images'/(stem+'.jpg'))
        caption = f'{i:02d} {r["source_image"]} | {r["selection"]}'
        cards.append(f'<article><h3>{html.escape(caption)}</h3><img loading="lazy" src="images/{stem}.jpg"></article>')
        notes.append({'id':i,'image':r['source_image'],'selection':r['selection'],
                      'comparison':'images/'+stem+'.jpg','assistant_observation':None,'label_change':False})
        # Single panel atlas for inspection. Green GT and magenta prediction distinguish origins.
        with Image.open(r['image']) as im:
            im = im.convert('RGB'); im.thumbnail((640,640))
            panel = Image.new('RGB',(640,680),'white'); panel.paste(im,(0,40))
        d = ImageDraw.Draw(panel)
        d.text((6,5),f'{i:02d} {Path(r["source_image"]).name}  GT=green PRED=magenta',fill='black')
        d.text((6,22),f'GT {len(r["gt"])} / PRED {len(r["predictions"])}',fill='black')
        sx,sy = im.width/r['width'], im.height/r['height']
        for boxes,color,tag in [(r['gt'],'#00c020','GT'),(r['predictions'],'#f000b0','P')]:
            for b in boxes:
                x1,y1,x2,y2 = b['box']; box=[x1*sx,y1*sy+40,x2*sx,y2*sy+40]
                d.rectangle(box,outline=color,width=2)
                text = f'{tag} {list(CLASSES)[b["class"]]}' + (f' {b["conf"]:.2f}' if tag=='P' else '')
                d.text((box[0],max(40,box[1]-12)),text,fill=color,stroke_width=1,stroke_fill='white')
        panels.append(panel)
    for start in range(0,len(panels),4):
        sheet = Image.new('RGB',(1280,1360),'#ddd')
        for j,panel in enumerate(panels[start:start+4]):
            sheet.paste(panel,((j%2)*640,(j//2)*680))
        sheet.save(output/'sheets'/f'{start+1:02d}-{min(start+4,len(panels)):02d}.jpg',quality=95)
    write_json(output/'review_notes.json',notes)
    write_json(output/'summary.json',{'sources':stats,'annotation_images_checked':len(rows),
               'annotation_issues':issues,'sample_inference_images':len(results),'selected_images':len(selected),
               'weights_sha256':sha256(weights),'data':data_identity(root/'dataset.yaml'),
               'seed':42,'conf':.25,'imgsz':640,'batch':8,'rect':False,'nms_iou':.7,'max_det':300,
               'note':'训练集内预测仅用于筛选复核样本，不代表泛化性能；空标注不保证无病害；未修改任何标签。'})
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>训练集复核</title>
<style>body{font:16px system-ui;background:#f4f6f8;max-width:1450px;margin:30px auto;padding:20px}article,header{background:white;padding:20px;margin:20px 0;border-radius:12px}img{width:100%}p{line-height:1.7}</style>
<header><h1>训练集：横向裂缝与空标注图片复核</h1><p>左侧标注，右侧预测；绿色为匹配，红色为未匹配。训练集内预测仅用于筛选，不是泛化评估。空标注不保证没有可见病害。定向选样不能用来估计总体错误比例，所有标签保持不变。</p></header>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print(f'Annotation issues: {len(issues)}; selected: {len(selected)}; {output}',flush=True)
