"""Local inspection batches with immutable predictions and versioned human edits."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import csv
import hashlib
import io
import json
import math
import os
import uuid
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps
from .common import ROOT, NAMES, sha256

STORE = ROOT/'runs/inspections'
LABELS = ['D00 纵向裂缝','D10 横向裂缝','D20 网状裂缝','D40 坑洞']
COLORS = ['#2e86de','#e67e22','#8e44ad','#e74c3c']


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(temp,path)
    finally:
        if temp.exists(): temp.unlink()


def load_batch(directory):
    return json.loads((Path(directory)/'batch.json').read_text(encoding='utf-8'))


def create_batch(uploads, title, weights, device='cpu', conf=.25, store=STORE):
    if not uploads or len(uploads)>100:
        raise ValueError('每批请上传 1～100 张图片。')
    if not 0 < conf < 1: raise ValueError('置信度必须在 0 和 1 之间。')
    weights = Path(weights).resolve()
    if not weights.is_file(): raise ValueError('找不到模型文件。')
    decoded=[]
    for filename,content in uploads:
        if len(content)>20*1024*1024: raise ValueError(f'{filename} 超过单图 20MB 限制。')
        try:
            with Image.open(io.BytesIO(content)) as im:
                if im.width*im.height>30_000_000: raise ValueError('图片像素数超过 3000 万。')
                decoded.append((Path(filename.replace('\\','/')).name, hashlib.sha256(content).hexdigest(),
                                ImageOps.exif_transpose(im).convert('RGB')))
        except (OSError,Image.DecompressionBombError) as exc:
            raise ValueError(f'无法读取图片：{filename}') from exc
    batch_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    directory=Path(store)/batch_id
    (directory/'images').mkdir(parents=True)
    items=[]
    for i,(filename,digest,im) in enumerate(decoded,1):
        image_id=f'{i:04d}_{digest[:12]}'
        relative=f'images/{image_id}.png'; im.save(directory/relative)
        items.append({'id':image_id,'filename':filename,'upload_sha256':digest,'image':relative,
                      'width':im.width,'height':im.height,'detection_status':'pending','error':None,
                      'original_predictions':[],'reviewed_boxes':None,'review_status':'pending',
                      'revision':0,'history':[]})
    batch={'schema_version':1,'id':batch_id,'title':title.strip() or '道路巡检',
           'created_at':now(),'model':{'weights':str(weights),'sha256':sha256(weights),
           'names':NAMES,'device':device,'conf':conf,'imgsz':640,'nms_iou':.7},'items':items}
    atomic_json(directory/'batch.json',batch)
    return directory


def process_batch(directory, model, progress=None):
    directory=Path(directory); batch=load_batch(directory)
    if list(model.names.values()) != NAMES: raise ValueError('请选择四类道路病害模型。')
    if sha256(Path(batch['model']['weights'])) != batch['model']['sha256']:
        raise ValueError('创建批次后模型文件发生变化，请使用原权重。')
    for index,item in enumerate(batch['items'],1):
        if item['detection_status']!='complete':
            try:
                result=model.predict(str(directory/item['image']),imgsz=640,
                    conf=batch['model']['conf'],iou=.7,max_det=300,rect=False,
                    device=batch['model']['device'],verbose=False,save=False)[0]
                item['original_predictions']=[{'id':f'p{i+1:03d}','class_id':int(b[5]),
                    'box':[float(v) for v in b[:4]],'confidence':float(b[4])}
                    for i,b in enumerate(result.boxes.data.cpu().tolist())]
                item['detection_status']='complete'; item['error']=None; item['detected_at']=now()
            except Exception as exc:
                item['detection_status']='error'; item['error']=str(exc)
            atomic_json(directory/'batch.json',batch)
        if progress: progress(index,len(batch['items']))
    return batch


def current_boxes(item):
    return item['reviewed_boxes'] if item['reviewed_boxes'] is not None else item['original_predictions']


def review_sort_key(item, mode):
    """Return a deterministic key for a human-review queue without changing predictions."""
    boxes = item.get('original_predictions', [])
    if mode == '低置信度优先':
        # Empty images are kept after images with actual low-confidence candidates.
        return (0 if boxes else 1, min((b.get('confidence', 1.0) for b in boxes), default=1.0), item['id'])
    if mode == '候选框多优先':
        return (-len(boxes), item['id'])
    if mode == '空图抽查':
        return (0 if not boxes else 1, item['id'])
    return (0, item['id'])


def editor_rows(item):
    return [{'编号':b['id'],'类别':LABELS[b['class_id']],
             'x1':b['box'][0],'y1':b['box'][1],'x2':b['box'][2],'y2':b['box'][3]}
            for b in current_boxes(item)]


def validate_edits(rows,item):
    predictions={b['id']:b for b in item['original_predictions']}
    existing={b['id']:b for b in current_boxes(item)}
    output=[]; seen=set()
    for i,row in enumerate(rows,1):
        category=row.get('类别')
        if not isinstance(category,str) or category not in LABELS:
            raise ValueError(f'第 {i} 行请选择病害类别。')
        try: box=[float(row[k]) for k in ['x1','y1','x2','y2']]
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError(f'第 {i} 行请完整填写四个坐标。') from exc
        x1,y1,x2,y2=box
        if not all(math.isfinite(v) for v in box) or not (0<=x1<x2<=item['width'] and 0<=y1<y2<=item['height']):
            raise ValueError(f'第 {i} 行框越界或大小无效：需要 0≤x1<x2≤{item["width"]}，0≤y1<y2≤{item["height"]}。')
        ident=row.get('编号')
        if not isinstance(ident,str) or ident not in existing: ident='m'+uuid.uuid4().hex[:10]
        if ident in seen: raise ValueError('框编号重复，请删除重复行。')
        seen.add(ident)
        original=predictions.get(ident)
        unchanged=original is not None and original['class_id']==LABELS.index(category) and original['box']==box
        output.append({'id':ident,'class_id':LABELS.index(category),'box':box,
                       'confidence':original['confidence'] if unchanged else None,
                       'origin':'model' if unchanged else 'human',
                       'original_prediction_id':ident if original else None})
    return output


def save_review(directory, image_id, rows, note='', expected_revision=None):
    directory=Path(directory); batch=load_batch(directory)
    item=next(x for x in batch['items'] if x['id']==image_id)
    if item['detection_status']!='complete': raise ValueError('请先完成此图检测。')
    if expected_revision is not None and item['revision']!=expected_revision:
        raise ValueError('此图已被更新，请刷新后再保存。')
    boxes=validate_edits(rows,item)
    item['revision']+=1; item['reviewed_boxes']=boxes; item['review_status']='reviewed'
    item['history'].append({'revision':item['revision'],'at':now(),'note':note.strip(),'boxes':copy.deepcopy(boxes)})
    atomic_json(directory/'batch.json',batch)
    return item


def annotated(image, boxes, grid=False):
    im=image.convert('RGB').copy(); draw=ImageDraw.Draw(im)
    fontfile=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(fontfile),max(12,im.width//50)) if fontfile.exists() else ImageFont.load_default()
    if grid:
        step=max(50,round(max(im.size)/8/50)*50)
        for x in range(0,im.width,step):
            draw.line((x,0,x,im.height),fill='#aaaaaa',width=1); draw.text((x+2,2),str(x),font=font,fill='yellow',stroke_width=1,stroke_fill='black')
        for y in range(step,im.height,step):
            draw.line((0,y,im.width,y),fill='#aaaaaa',width=1); draw.text((2,y),str(y),font=font,fill='yellow',stroke_width=1,stroke_fill='black')
    for b in boxes:
        color=COLORS[b['class_id']]; x1,y1,x2,y2=b['box']
        draw.rectangle((x1,y1,x2,y2),outline=color,width=max(2,im.width//300))
        label=b['id']+' '+LABELS[b['class_id']]+(f" {b['confidence']:.2f}" if b.get('confidence') is not None else ' 人工')
        y=max(0,y1-font.size-4); rect=draw.textbbox((x1,y),label,font=font)
        draw.rectangle(rect,fill=color); draw.text((x1,y),label,font=font,fill='white')
    return im


def safe_cell(value):
    text=str(value)
    return "'"+text if text.lstrip().startswith(('=','+','-','@')) else text


def export_batch(directory):
    directory=Path(directory); batch=load_batch(directory); output=io.BytesIO()
    summary=io.StringIO(newline=''); writer=csv.writer(summary)
    writer.writerow(['图片编号','原文件名','检测状态','复核状态','版本','模型框数','当前框数','D00','D10','D20','D40'])
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('records.json',json.dumps(batch,ensure_ascii=False,indent=2))
        for item in batch['items']:
            boxes=current_boxes(item); counts=[sum(b['class_id']==c for b in boxes) for c in range(4)]
            writer.writerow([item['id'],safe_cell(item['filename']),item['detection_status'],item['review_status'],
                             item['revision'],len(item['original_predictions']),len(boxes),*counts])
            archive.write(directory/item['image'],'originals/'+item['id']+'.png')
            if item['detection_status']=='complete':
                with Image.open(directory/item['image']) as im:
                    for label,values in [('model',item['original_predictions']),('current',boxes)]:
                        buf=io.BytesIO(); annotated(im,values).save(buf,format='JPEG',quality=90)
                        archive.writestr(f'{label}/{item["id"]}.jpg',buf.getvalue())
                if item['review_status']=='reviewed':
                    lines=[]
                    for b in boxes:
                        x1,y1,x2,y2=b['box']; w,h=item['width'],item['height']
                        lines.append(f'{b["class_id"]} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                    archive.writestr(f'reviewed_labels/{item["id"]}.txt','\n'.join(lines)+('\n' if lines else ''))
        archive.writestr('summary.csv',summary.getvalue().encode('utf-8-sig'))
        archive.writestr('README.txt','框数按图片累计，不是跨图片去重后的独立病害数量。\nrecords.json 保留模型原始输出和全部复核版本。\ncurrent/ 为已保存结果：待复核图使用原预测，须结合 summary.csv 的状态。\nreviewed_labels/ 仅包含已复核图，空文件表示人工确认当前为零个目标。坐标基于 originals/ 中经 EXIF 方向校正的 PNG。\n人工新增或修改框不赋予模型置信度。数据不自动加入训练集。\n类别：0=D00纵向裂缝，1=D10横向裂缝，2=D20网状裂缝，3=D40坑洞。\n')
    return output.getvalue()
