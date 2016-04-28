# -*- coding: utf-8 -*-
'''
This file is part of the Python jushacore package, an open source tool
for exploration, analysis and visualization of data.

'''
__version__ = '0.1.13'
__date__ = 'November 10, 2015'
import os
import sys
sys.path.append(os.getcwd())
if sys.hexversion < 0x02060000:
  raise ImportError('Python jushacore requires at least Python version 2.6.')
del sys

def cmappertoolserror(name):
    def f(*args, **kwargs):
        raise ImportError("please install the 'cmappertools' module "
        "for the '{}' function".format(name))
    return f

from jushacore._jushacore import *
from jushacore.draw_jusha_output import *
from jushacore.scale_graph import *

from jushacore import tools
from jushacore import metric
from jushacore import filters
from jushacore import cover
from jushacore import cutoff
