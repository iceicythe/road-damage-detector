"""Cross-split visual similarity screening; candidates are not confirmed leakage."""
from collections import Counter
import html
import json

from .common import ROOT, write_json, sha256


def scan_duplicates(name):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    root = ROOT/'data/processed/rdd_v1'
    output = ROOT/'runs/review'/name
    if output.exists():
        raise ValueError(f'目录已存在：{output}')
    output.mkdir(parents=True)
    rows = []
    for split in ['train','val','test']:
        rows.extend(json.loads(s) for s in (root/'splits'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines())
    phashes, dhashes = [], []
    def bits(values):
        return int(''.join('1' if x else '0' for x in values.flatten()),2)
    for i,r in enumerate(rows,1):
        gray = cv2.imdecode(np.fromfile(root/r['prepared_image'],dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'无法读取：{r["prepared_image"]}')
        dct = cv2.dct(cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].flatten()[1:]
        phashes.append(bits(dct > np.median(dct)))
        small = cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
        dhashes.append(bits(small[:,1:] > small[:,:-1]))
        if i%2000==0:
            print(f'Image hashes {i}/{len(rows)}',flush=True)
    (output/'hashes.jsonl').write_text(''.join(json.dumps({'image':r['image'],'split':r['split'],
        'sha256':r['sha256'],'phash':str(p),'dhash':str(d)},ensure_ascii=False)+'\n'
        for r,p,d in zip(rows,phashes,dhashes)),encoding='utf-8')
    phashes = np.array(phashes,dtype=np.uint64); dhashes = np.array(dhashes,dtype=np.uint64)
    splits = np.array([r['split'] for r in rows]); positions=np.arange(len(rows))
    pairs=[]
    for start in range(0,len(rows),256):
        stop=min(len(rows),start+256)
        distances=np.bitwise_count(phashes[start:stop,None]^phashes[None,:])
        x,y=np.where((distances<=6)&(positions[None,:]<positions[start:stop,None])&(splits[start:stop,None]!=splits[None,:]))
        for a,b in zip(x,y):
            i,j=int(start+a),int(b)
            dd=int((int(dhashes[i])^int(dhashes[j])).bit_count())
            if dd<=8:
                pairs.append({'a':rows[i]['image'],'b':rows[j]['image'],'a_split':rows[i]['split'],
                    'b_split':rows[j]['split'],'phash_distance':int(distances[a,b]),'dhash_distance':dd,
                    'exact_hash_equal':rows[i]['sha256']==rows[j]['sha256'],
                    'a_prepared':rows[i]['prepared_image'],'b_prepared':rows[j]['prepared_image']})
    pairs.sort(key=lambda p:(p['phash_distance']+p['dhash_distance'],p['a'],p['b']))
    write_json(output/'candidates.json',pairs)
    by_pair=Counter('/'.join(sorted([p['a_split'],p['b_split']])) for p in pairs)
    summary={'images':len(rows),'phash_bits':63,'phash_threshold':6,'dhash_bits':64,'dhash_threshold':8,
             'cross_split_pairs':len(pairs),'by_split_pair':dict(by_pair),
             'candidate_images':len({p[k] for p in pairs for k in ['a','b']}),
             'exact_duplicate_pairs':sum(p['exact_hash_equal'] for p in pairs),
             'split_manifest_sha256':{s:sha256(root/'splits'/f'{s}.jsonl') for s in ['train','val','test']},
             'note':'自动检查三集合图像相似性，不使用标签或模型输出。仅展示 train/val 候选；不展示测试图。感知哈希不证明同一道路、不保证召回所有近重复；不自动重划分。'}
    write_json(output/'summary.json',summary)
    (output/'images').mkdir()
    cards=[]
    shown=[p for p in pairs if {p['a_split'],p['b_split']}=={'train','val'}][:20]
    for i,p in enumerate(shown,1):
        canvas=Image.new('RGB',(1280,680),'white'); draw=ImageDraw.Draw(canvas)
        for side,key in enumerate(['a','b']):
            with Image.open(root/p[key+'_prepared']) as im:
                im=im.convert('RGB'); im.thumbnail((640,640)); canvas.paste(im,(side*640,40))
            draw.text((side*640+5,10),f'{p[key+"_split"]}: {p[key]}',fill='black')
        filename=f'{i:02d}.jpg'; canvas.save(output/'images'/filename,quality=90)
        cards.append(f'<article><h3>{i}: pHash={p["phash_distance"]}, dHash={p["dhash_distance"]}</h3><img src="images/{filename}"></article>')
    page='<!doctype html><meta charset="utf-8"><title>跨集合近重复候选</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;background:#f4f6f8}article{padding:20px;background:white;margin:20px 0}img{width:100%}p{line-height:1.7}</style><h1>跨集合近重复候选</h1><p>'+html.escape(summary['note'])+'</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
