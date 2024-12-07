#!/usr/bin/env python
import fileinput
import operator
from typing import List


def eval1(xs, mask):
    r = xs[0]
    for i in range(len(xs) - 1):
        if (mask & (1 << i)) == 0:
            r += xs[i + 1]
        else:
            r *= xs[i + 1]
    return r


def solve1(xs, r):
    mask = 0
    while mask != (1 << len(xs) - 1):
        if eval1(xs, mask) == r:
            return True
        mask += 1
    return False


OpEval = [operator.add, operator.mul, lambda a, b: int(str(a) + str(b))]


def eval2(xs, mask):
    r = xs[0]
    for i in range(len(xs) - 1):
        r = OpEval[mask[i]](r, xs[i + 1])
    return r


def nextmask2(xs: List[int]):
    for i, x in enumerate(xs):
        if x == 2:
            xs[i] = 0
        else:
            xs[i] = x + 1
            return True
    return False


def solve2(xs, r):
    mask = [0] * (len(xs) - 1)
    while True:
        if eval2(xs, mask) == r:
            return True
        if not nextmask2(mask):
            return False


result = 0
result2 = 0
for line in fileinput.input():
    lhs, rhs = line.split(": ")
    xs = list(map(int, rhs.split(" ")))
    r = int(lhs)
    if solve1(xs, r):
        result += r
    if solve2(xs, r):
        result2 += r

print(result)
print(result2)
