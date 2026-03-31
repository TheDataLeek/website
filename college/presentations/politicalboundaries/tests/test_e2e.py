import politicalboundaries
import pytest
import numpy as np


class TestEndToEnd(object):
    @pytest.fixture
    def system(self):
        return politicalboundaries.System('./smallState.txt')

    def test_simulated_annealing(self, system):
        politicalboundaries.simulated_annealing(system, 8, 10, False, False)
        politicalboundaries.simulated_annealing(system, 4, 10, False, False)

    def test_genetic(self, system):
        politicalboundaries.genetic_algorithm(system, 8, 10, False, False)
        politicalboundaries.genetic_algorithm(system, 4, 10, False, False)
