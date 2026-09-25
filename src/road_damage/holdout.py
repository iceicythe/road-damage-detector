"""Freeze a test subset excluding known cross-split similarity candidates."""
import json
import shutil

from .common import ROOT, NAMES, sha256, write_json


def prepare_holdout():
    import yaml
    original = ROOT/'data/processed/rdd_v1'
    audit = ROOT/'runs/review/split_similarity_v1'
    output = ROOT/'data/processed/rdd_screened_test_v1'
    weights = ROOT/'runs/train/n640/weights/best.pt'
    if output.exists(): raise ValueError('留出测试已冻结，请使用已有配置；不会覆盖名单。')
    summary = json.loads((audit/'summary.json').read_text(encoding='utf-8'))
    for split in ['train','val','test']:
        if sha256(original/'splits'/f'{split}.jsonl') != summary['split_manifest_sha256'][split]:
            raise ValueError('相似性审计与当前集合清单不一致，不能使用过期筛查结果。')
    pairs = json.loads((audit/'candidates.json').read_text(encoding='utf-8'))
    if len(pairs) != summary['cross_split_pairs']: raise ValueError('候选数与审计摘要不符。')
    rows = [json.loads(line) for line in (original/'splits/test.jsonl').read_text(encoding='utf-8').splitlines()]
    exclusions = {}
    for pair in pairs:
        for side, other in [('a','b'),('b','a')]:
            if pair[side+'_split']=='test' and pair[other+'_split'] in {'train','val'}:
                exclusions.setdefault(pair[side], []).append({'image':pair[other], 'split':pair[other+'_split'],
                    'phash_distance':pair['phash_distance'], 'dhash_distance':pair['dhash_distance']})
    if not set(exclusions).issubset({r['image'] for r in rows}): raise ValueError('候选图片不属于当前测试清单。')
    included = [r for r in rows if r['image'] not in exclusions]
    if not included: raise ValueError('筛查后没有测试图片。')
    (output/'splits').mkdir(parents=True)
    for split in ['train','val']:
        shutil.copyfile(original/'splits'/f'{split}.jsonl', output/'splits'/f'{split}.jsonl')
    (output/'splits/test.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in included),encoding='utf-8')
    (output/'test.txt').write_text(''.join(str(original/r['prepared_image'])+'\n' for r in included),encoding='utf-8')
    config = {'path':str(original),'train':'images/train','val':'images/val','test':str(output/'test.txt'),
              'names':dict(enumerate(NAMES))}
    (output/'dataset.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    policy = {'weights':str(weights), 'weights_sha256':sha256(weights), 'split':'test','imgsz':640,
              'model_selection':'n640 基线，依据此前验证实验选定；未使用本次测试成绩选择模型。',
              'selection_rule':'排除 pHash<=6 且 dHash<=8 的 test 对 train/val 已知候选；规则不使用标签或模型预测。',
              'source_manifest_sha256':summary['split_manifest_sha256'],
              'candidate_file_sha256':sha256(audit/'candidates.json'),
              'test_manifest_sha256':sha256(output/'splits/test.jsonl'),
              'test_list_sha256':sha256(output/'test.txt'),
              'original_test_images':len(rows),'excluded_images':len(exclusions),'included_images':len(included),
              'limitations':'仅隔离当前哈希规则发现的相似候选，未按完整道路/序列标识分组；不能保证道路独立。原始标注未重新人工验收。'}
    write_json(output/'evaluation_policy.json',policy)
    write_json(output/'excluded.json',exclusions)
    print(json.dumps(policy,ensure_ascii=False,indent=2))
