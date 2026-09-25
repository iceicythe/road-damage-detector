from road_damage.review import match, diagnose


def test_matching_does_not_double_count_or_match_wrong_class():
    gt = [{'class': 0, 'box': [0, 0, 10, 10]}]
    predictions = [{'class': 1, 'box': [0, 0, 10, 10], 'conf': .99},
                   {'class': 0, 'box': [0, 0, 10, 10], 'conf': .8},
                   {'class': 0, 'box': [0, 0, 10, 10], 'conf': .7}]
    result = match(gt, predictions)
    assert result == {'matches': [[0, 1, 1.0]], 'fn': [], 'fp': [0, 2]}


def test_localization_hint_keeps_both_fn_and_fp():
    gt = [{'class': 1, 'box': [0, 0, 10, 10]}]
    predictions = [{'class': 1, 'box': [5, 0, 15, 10], 'conf': .8}]
    result = match(gt, predictions)
    assert result['fn'] == [0] and result['fp'] == [0]
    hints = diagnose(gt, predictions, result, .5)
    assert len(hints['localization_candidates']) == 1
    assert not hints['class_confusion_candidates']


def test_empty_background_and_completely_missed_image():
    assert match([], []) == {'matches': [], 'fn': [], 'fp': []}
    assert match([{'class': 1, 'box': [0, 0, 10, 10]}], [])['fn'] == [0]


def test_threshold_sweep_recovers_low_conf_target_without_counting_duplicate():
    from road_damage.thresholds import score
    records = [{'gt': [{'class': 1, 'box': [0, 0, 10, 10]}],
                'predictions': [{'class': 1, 'box': [0, 0, 10, 10], 'conf': .2},
                                {'class': 1, 'box': [0, 0, 10, 10], 'conf': .15},
                                {'class': 0, 'box': [20, 20, 30, 30], 'conf': .4}]}]
    low, _ = score(records, .1, .5)
    high, _ = score(records, .25, .5)
    assert low['overall_micro']['tp'] == 1
    assert low['overall_micro']['fp'] == 2
    assert low['overall_micro']['fn'] == 0
    assert high['overall_micro']['tp'] == 0
    assert high['overall_micro']['fp'] == 1
    assert high['overall_micro']['fn'] == 1
