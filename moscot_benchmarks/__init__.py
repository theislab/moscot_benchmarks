import numpy as np
from typing import Callable
from functools import wraps, partial
from memory_profiler import memory_usage
from time import perf_counter
from sacred import Experiment
import seml
import os
import logging
import pandas as pd

ex = Experiment()
seml.setup_logger(ex)