import numpy as np
from typing import Callable
from functools import wraps, partial
from memory_profiler import memory_usage
from time import perf_counter
import pandas as pd
import time

# DO NOT IMPORT ANY JAX FUNCTIONS HERE

def benchmark_memory(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(*args, **kwargs):
        return usage((func, args, kwargs))

    mp = False
    usage = partial(memory_usage, interval=0.01, include_children=mp, multiprocess=mp, retval=True)

    return wrapper

def benchmark_time(func: Callable) -> Callable:

    @wraps(func) 
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # needs python3.3 or higher
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return end-start, result
    
    return wrapper