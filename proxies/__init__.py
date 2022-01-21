
import os
_CF = os.path.realpath(__file__)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(_CF)))

from . import register

class Proxy(object):
    def __init__(self, task_type='extraction'):
        super(Proxy, self).__init__()

        self.task_type = task_type
        assert( self.task_type in ('extraction', 'matching') )

        self.model = None

    def initialize(self):
        pass

    def __call__(self, inputs):
        pass

from .KP2D import proxy
from .SuperGlue import proxy
from .feature_extraction import proxy