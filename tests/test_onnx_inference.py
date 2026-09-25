import numpy as np
import pytest

from road_damage.onnx_inference import preprocess, postprocess, nms
from road_damage.deployment import compare_boxes


def test_letterbox_color_normalization_and_inverse_coordinates():
    image = np.zeros((101, 200, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    tensor, geometry = preprocess(image)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 320, 320], [1, 0, 0])
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([114]*3)/255)
    ratio, left, top, _, _ = geometry
    raw = np.zeros((1, 8, 1), dtype=np.float32)
    raw[0, :4, 0] = [100*ratio+left, 50.5*ratio+top, 200*ratio, 101*ratio]
    raw[0, 5, 0] = .8
    boxes = postprocess(raw, geometry)
    np.testing.assert_allclose(boxes[0, :4], [0, 0, 200, 101], atol=1e-4)
    assert boxes[0, 5] == 1


def test_nms_suppresses_same_class_only_and_empty_predictions():
    boxes = np.array([[0,0,10,10],[0,0,10,10],[0,0,10,10]], dtype=np.float32)
    assert nms(boxes, np.array([.9,.8,.7]), np.array([0,0,1])).tolist() == [0,2]
    assert postprocess(np.zeros((1,8,10)), (1,0,0,10,10)).shape == (0,6)
    with pytest.raises(ValueError): postprocess(np.zeros((1,84,10)), (1,0,0,10,10))


def test_parity_compares_boxes_independent_of_order_and_detects_differences():
    boxes = np.array([[0,0,10,10,.8,0],[20,20,30,30,.7,1]], dtype=np.float32)
    assert compare_boxes(boxes, boxes[::-1])['passed']
    changed = boxes.copy(); changed[0,0] = 1
    assert not compare_boxes(boxes, changed)['passed']
    assert not compare_boxes(boxes, boxes[:1])['passed']
