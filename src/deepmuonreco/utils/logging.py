import time
from contextlib import contextmanager

@contextmanager
def elapsed_timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
