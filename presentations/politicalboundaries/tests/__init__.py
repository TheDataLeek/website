import politicalboundaries
import pytest
import numpy as np


class PseudoSystem(object):
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height

from .test_e2e import *
from .test_mask import *
from .test_solution import *
from .test_system import *