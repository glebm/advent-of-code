#!/usr/bin/env python
import fileinput
import collections
import numpy as np


def in_bounds(p, w, h):
    return p[0] >= 0 and p[0] < w and p[1] >= 0 and p[1] < h


def get_antinodes(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    hyp = p2 - p1
    r1 = p1 - hyp
    r2 = p2 + hyp
    return [(r1[0], r1[1]), (r2[0], r2[1])]


def get_antinodes2(p1, p2, w, h):
    p1 = np.array(p1)
    p2 = np.array(p2)
    hyp = p2 - p1
    result = []
    r1 = p1 - hyp
    r2 = p2 + hyp
    while in_bounds(r1, w, h):
        result.append((r1[0], r1[1]))
        r1 -= hyp
    while in_bounds(r2, w, h):
        result.append((r2[0], r2[1]))
        r2 += hyp
    return result


m = [list(line.removesuffix("\n")) for line in fileinput.input()]

antennas = collections.defaultdict(set)
for y, row in enumerate(m):
    for x, c in enumerate(row):
        if c == ".":
            continue
        antennas[c].add((x, y))

antinodes = set()
antinodes2 = set()
for xs in antennas.values():
    ls = list(xs)
    if len(ls) > 1:
        for a in ls:
            antinodes2.add(a)
    for i in range(len(ls)):
        for j in range(i):
            for a in get_antinodes(ls[i], ls[j]):
                if in_bounds(a, len(m[0]), len(m)):
                    antinodes.add(a)
            for a in get_antinodes2(ls[i], ls[j], len(m[0]), len(m)):
                antinodes2.add(a)

print(len(antinodes))
print(len(antinodes2))
