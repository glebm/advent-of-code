#!/usr/bin/env python
import fileinput
import functools


def check(order, pages, is_part1):
    def _compare(a, b):
        if (a, b) in order:
            return -1
        if (b, a) in order:
            return 1
        return 0

    sorted_pages = sorted(pages, key=functools.cmp_to_key(_compare))
    if is_part1:
        return pages[len(pages) // 2] if sorted_pages == pages else 0
    else:
        return sorted_pages[len(pages) // 2] if sorted_pages != pages else 0


order = set()
result = 0
result2 = 0
with fileinput.input() as f:
    for line in f:
        if line == "\n":
            break
        order.add(tuple(map(int, line.split("|"))))
    for line in f:
        xs = list(map(int, line.split(",")))
        result += check(order, xs, is_part1=True)
        result2 += check(order, xs, is_part1=False)
print(result)
print(result2)
