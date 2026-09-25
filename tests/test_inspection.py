import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image
from road_damage.common import NAMES
from road_damage.inspection import (create_batch,load_batch,process_batch,save_review,
                                    editor_rows,export_batch,LABELS,validate_edits,review_sort_key)


@pytest.fixture
def completed(tmp_path):
    content=io.BytesIO(); Image.new('RGB',(100,80),'white').save(content,format='PNG')
    weight=tmp_path/'fake.pt'; weight.write_bytes(b'test model')
    folder=create_batch([('../same.png',content.getvalue()),('same.png',content.getvalue())],
                        'test',weight,store=tmp_path/'batches')
    class Data:
        def cpu(self): return self
        def tolist(self): return [[10.,10.,30.,30.,.8,1.]]
    class Model:
        names=dict(enumerate(NAMES))
        def predict(self,*args,**kwargs): return [SimpleNamespace(boxes=SimpleNamespace(data=Data()))]
    process_batch(folder,Model())
    return folder


def test_edits_preserve_raw_predictions_and_history(completed):
    item=load_batch(completed)['items'][0]
    rows=editor_rows(item); rows[0]['类别']=LABELS[2]
    saved=save_review(completed,item['id'],rows,'class corrected',0)
    assert saved['original_predictions'][0]['class_id']==1
    assert saved['reviewed_boxes'][0]['class_id']==2
    assert saved['reviewed_boxes'][0]['confidence'] is None
    assert saved['history'][0]['note']=='class corrected'
    with pytest.raises(ValueError,match='更新'):
        save_review(completed,item['id'],[],expected_revision=0)
    save_review(completed,item['id'],[],'removed',1)
    reloaded=load_batch(completed)['items'][0]
    assert reloaded['reviewed_boxes']==[]
    assert len(reloaded['history'])==2 and len(reloaded['original_predictions'])==1


def test_exports_only_reviewed_labels_and_keeps_unreviewed_status(completed):
    items=load_batch(completed)['items']
    assert items[0]['image']!=items[1]['image']
    save_review(completed,items[0]['id'],[],expected_revision=0)
    with zipfile.ZipFile(io.BytesIO(export_batch(completed))) as archive:
        labels=[n for n in archive.namelist() if n.startswith('reviewed_labels/')]
        assert labels==[f'reviewed_labels/{items[0]["id"]}.txt']
        assert archive.read(labels[0])==b''
        assert json.loads(archive.read('records.json'))['items'][1]['review_status']=='pending'
        assert not any('..' in n for n in archive.namelist())


def test_add_box_and_reject_bad_coordinates(completed):
    item=load_batch(completed)['items'][0]
    row={'类别':LABELS[0],'x1':0,'y1':0,'x2':100,'y2':80}
    boxes=validate_edits([row],item)
    assert boxes[0]['origin']=='human' and boxes[0]['confidence'] is None
    for value in [float('nan'),101,-1]:
        with pytest.raises(ValueError): validate_edits([{**row,'x2':value}],item)


def test_review_queue_sorting_is_deterministic_and_does_not_change_boxes(completed):
    items=load_batch(completed)['items']
    items[0]['original_predictions'][0]['confidence']=.15
    items[1]['original_predictions'][0]['confidence']=.85
    assert review_sort_key(items[0],'低置信度优先') < review_sort_key(items[1],'低置信度优先')
    assert review_sort_key(items[0],'候选框多优先') == review_sort_key(items[0],'候选框多优先')
    empty=dict(items[0],id='empty',original_predictions=[])
    assert review_sort_key(empty,'空图抽查') < review_sort_key(items[0],'空图抽查')
