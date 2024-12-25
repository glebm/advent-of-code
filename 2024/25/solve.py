#!/usr/bin/env python
import fileinput
import numpy as np

locks = []
keys = []
state = 0
for line in fileinput.input():
    line = line.removesuffix("\n")
    if state == 0:
        if line == "#####":
            state = 1
            locks.append(np.zeros(5, dtype=np.int32))
        else:
            state = 2
            keys.append(np.zeros(5, dtype=np.int32))
    elif not line:
        state = 0
    else:
        a = locks[-1] if state == 1 else keys[-1]
        for i, c in enumerate(line):
            a[i] += c == "#"

print(sum(all(lock <= 6 - key) for lock in locks for key in keys))
