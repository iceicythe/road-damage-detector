import pytest

from road_damage.sampling import d10_repeat_rows


def test_sampling_repeats_whole_d10_image_and_retains_background():
    background = {'split':'train','labels':[]}
    mixed = {'split':'train','labels':[[1,.5,.5,.1,.1],[3,.5,.5,.1,.1]]}
    other = {'split':'train','labels':[[0,.5,.5,.1,.1]]}
    result = d10_repeat_rows([background,mixed,other])
    assert result == [background,mixed,mixed,other]
    assert mixed['labels'][1][0] == 3


def test_sampling_rejects_heldout_images():
    with pytest.raises(ValueError):
        d10_repeat_rows([{'split':'val','labels':[[1,.5,.5,.1,.1]]}])
