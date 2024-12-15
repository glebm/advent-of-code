#!/usr/bin/env python
import fileinput
import re
from functools import reduce
from operator import mul
import numpy as np
import os


def init_debug1(w, h):
    return (
        [["."] * (w // 2) + [" "] + ["."] * (w // 2) for _ in range(h // 2)]
        + [[" "] * w]
        + [["."] * (w // 2) + [" "] + ["."] * (w // 2) for _ in range(h // 2)]
    )


def init_debug2(w, h):
    return [["."] * w for _ in range(h)]


def print_debug(d):
    print("\n".join("".join(s) for s in d))


def advance(x, v, t, w):
    return (x + v * t) % w


def solve1(m, w, h, t):
    d = init_debug1(w, h)
    q = [0, 0, 0, 0]
    for x, y, vx, vy in m:
        nx = advance(x, vx, t, w)
        ny = advance(y, vy, t, h)
        if nx == (w // 2) or ny == (h // 2):
            continue
        d[ny][nx] = str(int(d[ny][nx]) + 1) if d[ny][nx] != "." else "1"
        quad = (int(ny > (h // 2)) << 1) | int(nx > (w // 2))
        q[quad] += 1
    return reduce(mul, q)


def consecutive_nonzero_length(xs):
    a = xs > 0
    loc_change = np.empty_like(a, dtype=bool)
    loc_change[0] = True
    np.not_equal(a[:-1], a[1:], out=loc_change[1:])
    run_starts = np.nonzero(a & loc_change)[0]
    if len(run_starts) == 0:
        return 0
    loc_run_ends = ~a & loc_change
    loc_run_ends[0] = False
    run_ends = np.nonzero(loc_run_ends)[0]
    if a[-1]:
        run_ends = np.append(run_ends, len(a))
    run_lengths = run_ends - run_starts
    return max(run_lengths)


def solve2(m, w, h):
    counts = np.zeros((h, w), dtype=np.int32)
    pos = [(x, y) for x, y, _, _ in m]
    for x, y in pos:
        counts[y, x] += 1

    for step in range(500000):
        for i, (_, _, vx, vy) in enumerate(m):
            x, y = pos[i]
            counts[y, x] -= 1
            nx = advance(x, vx, 1, w)
            ny = advance(y, vy, 1, h)
            pos[i] = (nx, ny)
            counts[ny, nx] += 1

        # Look for a wide consecutive run (the bottom of a tree)
        for y in range(h):
            if consecutive_nonzero_length(counts[y]) > 10:
                d = init_debug2(w, h)
                for x, y in pos:
                    d[y][x] = "*"
                print_debug(d)
                return step + 1

    return -1


m = [list(map(int, re.findall(r"-?\d+", line))) for line in fileinput.input()]
newpos = []
w = 101
h = 103

print(solve1(m, w, h, t=100))
print(solve2(m, w, h))
