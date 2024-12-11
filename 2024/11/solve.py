#!/usr/bin/env python
import fileinput
import functools


@functools.cache
def solve(x, iter):
    if iter == 0:
        return 1
    if x == 0:
        return solve(1, iter - 1)
    s = str(x)
    if len(s) % 2 == 0:
        l = int(s[: len(s) // 2])
        r = int(s[len(s) // 2 :])
        return solve(l, iter - 1) + solve(r, iter - 1)
    return solve(x * 2024, iter - 1)


m = list(map(int, fileinput.input().__next__().removesuffix("\n").split(" ")))
print(sum(solve(x, 25) for x in m))
print(sum(solve(x, 75) for x in m))
