# Shim to load pickles produced under NumPy 2.x while running NumPy 1.x
import sys, types, numpy as _np
if 'numpy._core' not in sys.modules:
    m = types.ModuleType('numpy._core')
    m.__dict__.update(_np.__dict__)
    sys.modules['numpy._core'] = m
