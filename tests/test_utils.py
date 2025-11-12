import numpy as np
import pytest

from eks.utils import crop_frames


def test_crop_frames_no_crop_none():
    """If s_frames is None, return y unchanged."""
    y = np.arange(20)
    out = crop_frames(y, None)
    assert np.shares_memory(out, y) or np.array_equal(out, y)
    assert out.shape == y.shape


def test_crop_frames_no_crop_none_none():
    """If s_frames == [(None, None)], return y unchanged."""
    y = np.arange(20)
    out = crop_frames(y, [(None, None)])
    assert np.shares_memory(out, y) or np.array_equal(out, y)
    assert out.shape == y.shape


def test_crop_frames_single_span():
    """Basic single-span crop with 1-based (inclusive) bounds."""
    y = np.arange(10)  # [0..9]
    # (2,5) -> indices 2,3,4
    out = crop_frames(y, [(2, 5)])
    np.testing.assert_array_equal(out, np.array([2, 3, 4]))


def test_crop_frames_open_ended_spans():
    """Open-ended spans using None for start or end."""
    y = np.arange(10)
    # (None,3) -> [0,1,2]
    # (7,None) -> [7,8,9]
    out = crop_frames(y, [(None, 3), (7, None)])
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 6, 7, 8, 9]))


def test_crop_frames_invalid_tuple_shape():
    """Each element must be a 2-tuple (start, end)."""
    y = np.arange(10)
    with pytest.raises(ValueError):
        crop_frames(y, [(1, 3, 5)])  # 3-tuple is invalid


def test_crop_frames_out_of_bounds():
    """Out-of-bounds ranges raise ValueError."""
    y = np.arange(10)
    with pytest.raises(ValueError):
        crop_frames(y, [(1, 20)])

    # start beyond end (invalid after conversion)
    with pytest.raises(ValueError):
        crop_frames(y, [(6, 5)])


def test_crop_frames_overlap_raises():
    """Overlapping intervals are rejected."""
    y = np.arange(20)
    with pytest.raises(ValueError):
        crop_frames(y, [(2, 6), (5, 10)])
