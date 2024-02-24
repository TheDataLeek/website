import politicalboundaries
import pytest
import numpy as np


class TestMask(object):
    @pytest.fixture
    def mask(self):
        newmask = politicalboundaries.Mask(height=3, width=3)
        newmask.mask = np.array([
            [1, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
        ])
        return newmask

    def test_size(self, mask):
        assert mask.size == 5

    def test_parse(self, mask):
        old_mask_vals = np.copy(mask.mask)
        mask.parse_list([
            [1, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
        ])
        assert np.array_equal(old_mask_vals, mask.mask)

        mask.parse_list([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        mask.parse_list([[1, 0, 0],
                         [0, 0, 0]])

    def test_valid(self, mask):
        assert mask.is_valid is False

        mask.parse_list([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        assert mask.is_valid is True
        mask.parse_list([[1, 0, 0],
                         [0, 0, 0]])
        assert mask.is_valid is True
        mask.parse_list([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 1]])
        assert mask.is_valid is False
        mask.parse_list([[1, 0, 0],
                         [1, 0, 1],
                         [0, 0, 1]])
        assert mask.is_valid is False
        mask.parse_list([[1, 0, 0],
                         [1, 1, 1],
                         [0, 0, 1]])
        assert mask.is_valid is True
        mask.parse_list([[1, 1, 0],
                         [0, 1, 1]])
        assert mask.is_valid is True

        mask.parse_list([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]])
        assert mask.is_valid is False

    def test_parse_locations(self, mask):
        mask.parse_locations(3, 3, [[0, 0], [1, 1], [1, 0]])
        assert (
            mask == np.array([[1, 0, 0],
                              [1, 1, 0],
                              [0, 0, 0]]))

    def test_make_valid1(self, mask):
        mask.make_valid()
        assert (
            mask == np.array([[0, 0, 1],
                              [0, 0, 1],
                              [0, 0, 1]]) or
            mask == np.array([[1, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]])
        )

    def test_make_valid_2(self, mask):
        mask.parse_list([[1, 0, 1],
                         [0, 0, 0],
                         [1, 0, 1]])
        mask.make_valid()
        assert (
            mask == np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]) or
            mask == np.array([[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]) or
            mask == np.array([[0, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]]) or
            mask == np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]))

    def test_location(self, mask):
        assert mask.location == [(0, 0),
                                 (0, 2),
                                 (1, 0),
                                 (1, 2),
                                 (2, 2)]

    def test_get_labels(self, mask):
        curlab, labels = mask.get_labels()
        assert curlab == 3
        assert np.array_equal(labels, np.array([[1, 0, 2],
                                                [1, 0, 2],
                                                [0, 0, 2]]))
