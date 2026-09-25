"""Prepare a train-only repeated-image list; original images and labels stay intact."""
from collections import Counter
import json
import math

import yaml

from .common import ROOT, NAMES, write_json
from .experiments import data_identity


def d10_repeat_rows(rows):
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('采样列表只允许非空训练集，不能加入验证/测试图像。')
    return [r for r in rows for _ in range(2 if any(b[0] == 1 for b in r['labels']) else 1)]


def prepare_sampling():
    root = ROOT/'data/processed/rdd_v1'
    output = ROOT/'data/processed/rdd_d10x2'
    if output.exists():
        raise ValueError(f'采样配置已存在：{output}')
    rows = [json.loads(s) for s in (root/'splits/train.jsonl').read_text(encoding='utf-8').splitlines()]
    repeated = d10_repeat_rows(rows)
    # The original source manifests were already split. Verify exact-hash isolation again.
    heldout = {json.loads(s)['sha256'] for split in ['val','test']
               for s in (root/'splits'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()}
    if any(r['sha256'] in heldout for r in repeated):
        raise ValueError('发现训练图像与验证/测试集的精确哈希交集。')
    paths = [(root/r['prepared_image']).resolve() for r in repeated]
    if not all(p.is_file() for p in paths):
        raise ValueError('训练图片缺失')
    output.mkdir(parents=True)
    train_list = output/'train.txt'
    train_list.write_text(''.join(p.as_posix()+'\n' for p in paths),encoding='utf-8')
    (output/'dataset.yaml').write_text(yaml.safe_dump({'path':str(root.resolve()),
        'train':str(train_list.resolve()),'val':'images/val','test':'images/test',
        'names':dict(enumerate(NAMES))},sort_keys=False),encoding='utf-8')
    boxes = Counter(int(b[0]) for r in repeated for b in r['labels'])
    budget_epochs = round(100*math.ceil(len(rows)/8)/math.ceil(len(repeated)/8))
    write_json(output/'sampling.json',{'source_data':data_identity(root/'dataset.yaml'),
        'unique_images':len(rows),'entries_per_epoch':len(repeated),
        'repeated_d10_images':len(repeated)-len(rows),'weighted_boxes':dict(boxes),
        'suggested_epochs_for_approx_equal_max_batches':budget_epochs,
        'baseline_max_batches':100*math.ceil(len(rows)/8),
        'variant_max_batches':budget_epochs*math.ceil(len(repeated)/8),
        'note':'含 D10 的整张训练图出现两次；同图其他类别也会被重复。保留所有背景图。没有新增独立样本，不改变两国 D10 比例。验证测试路径不变；不保证近重复道路场景隔离。'})
    print((output/'sampling.json').read_text(encoding='utf-8'))
