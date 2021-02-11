try:
    import cupy as cp
    from cupy import get_array_module, asnumpy, fuse
except:
    import numpy as np

    cp = np

    def get_array_module(a):
        return np

    def asnumpy(a):
        return a

    def fuse():
        return lambda x: x
