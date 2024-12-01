#!/usr/bin/env python
import fileinput
import numpy as np

a = np.loadtxt(fileinput.input(), np.int32).T
a.sort(axis=1)
print(np.sum(np.abs(a[0] - a[1])))
left = np.searchsorted(a[1], a[0], "left")
right = np.searchsorted(a[1], a[0], "right")
print(np.sum((right - left) * a[0]))
