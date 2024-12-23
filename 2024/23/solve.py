#!/usr/bin/env python
import fileinput
import itertools
import collections


def solve1(edges):
    es = collections.defaultdict(set)
    for a, b in edges:
        es[a].add(b)
        es[b].add(a)

    result = 0
    r2 = 0
    r3 = 0
    for u, vs in es.items():
        if u[0] != "t":
            continue
        for v1, v2 in itertools.product(vs, repeat=2):
            if v1 >= v2:
                continue
            if v2 in es[v1]:
                result += 1
                r2 += v1[0] == "t"
                r3 += v2[0] == "t"
    return result - r2 // 2 - r3 // 2


def bron_kerbosch(
    es: dict[str, set[str]],
    r: set[str],
    p: set[str],
    x: set[str],
    result: list[set[str]],
):
    if not p and not x:
        result.append(r)
        return
    while p:
        v = p.pop()
        bron_kerbosch(
            es, r.union(set((v,))), p.intersection(es[v]), x.intersection(es[v]), result
        )
        x.add(v)


def solve2(edges):
    vs = set()
    es = collections.defaultdict(set)
    for a, b in edges:
        es[a].add(b)
        es[b].add(a)
        vs.add(a)
        vs.add(b)

    cliques = []
    bron_kerbosch(es, set(), vs, set(), cliques)
    return ",".join(sorted(max(cliques, key=len)))


edges = [tuple(line.removesuffix("\n").split("-")) for line in fileinput.input()]

print(solve1(edges))
print(solve2(edges))
