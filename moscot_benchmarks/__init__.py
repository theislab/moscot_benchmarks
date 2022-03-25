from time import perf_counter
from typing import Callable
from functools import wraps, partial
import os
import logging

from sacred import Experiment
from memory_profiler import memory_usage
import seml
import pandas as pd

import numpy as np

ex = Experiment()
seml.setup_logger(ex)
