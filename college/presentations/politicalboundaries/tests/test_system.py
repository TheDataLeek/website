import politicalboundaries
import pytest
import numpy as np


class TestSystem(object):
    @pytest.fixture
    def system(self):
        return politicalboundaries.System('./smallState.txt')

    def test_file_parser(self, system):
        filesystem = np.array([
                        [0,1,0,1,0,1,1,1],
                        [0,0,1,0,1,1,1,1],
                        [0,0,0,1,1,1,1,1],
                        [0,0,1,1,1,1,0,1],
                        [1,1,0,0,0,1,1,1],
                        [1,0,0,0,0,0,1,1],
                        [1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,1,0]])
        assert (system.matrix == filesystem).all()

    def test_width(self, system):
        assert system.width == 8

    def test_height(self, system):
        assert system.height == 8

    def test_stats(self, system):
        size, percents = system.stats
        assert size == 64
        assert percents['D'] == 0.5
        assert percents['R'] == 0.5
