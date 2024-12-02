#!/usr/bin/env python
import fileinput
import operator
import itertools
import dataclasses
from typing import Iterable


def prevs(r):
    return itertools.islice(r, 1, None)


def check(r):
    return not (
        any(map(operator.le, r, prevs(r))) and any(map(operator.ge, r, prevs(r)))
    ) and all(map(lambda a, b: abs(a - b) <= 3, r, prevs(r)))


@dataclasses.dataclass
class ExcludeI:
    r: Iterable
    i: int

    def __iter__(self):
        return itertools.chain(
            itertools.islice(self.r, self.i), itertools.islice(self.r, self.i + 1, None)
        )


safe = 0
safe2 = 0
for line in fileinput.input():
    nums = list(map(int, line.split()))
    if check(nums):
        safe += 1
    else:
        safe2 += any(check(ExcludeI(nums, i)) for i in range(0, len(nums)))
print(safe)
print(safe + safe2)
