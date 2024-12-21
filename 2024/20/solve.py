#!/usr/bin/env python
import fileinput
import collections
import itertools

_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def in_bounds(m, u):
    return u[0] >= 0 and u[0] < len(m[0]) and u[1] >= 0 and u[1] < len(m)


def shortest_path(m: list[list[str]], start: tuple[int, int], end: tuple[int, int]):
    queue = collections.deque([start])
    dist = [[0] * len(m[0]) for _ in range(len(m))]
    prev = [[None] * len(m[0]) for _ in range(len(m))]
    while queue:
        u = queue.popleft()
        for d in _DIRS:
            v = (u[0] + d[0], u[1] + d[1])
            if m[v[1]][v[0]] == "#" or dist[v[1]][v[0]] != 0:
                continue
            dist[v[1]][v[0]] = dist[u[1]][u[0]] + 1
            prev[v[1]][v[0]] = u
            if v == end:
                path = []
                while v != start:
                    v = prev[v[1]][v[0]]
                    path.append(v)
                path.reverse()
                return dist, path
            queue.append(v)


def solve(m, end, dist, path, max_cheat):
    cheat_ds = [
        (dx, dy)
        for dx, dy in itertools.product(range(-max_cheat, max_cheat + 1), repeat=2)
        if abs(dx) + abs(dy) > 0 and abs(dx) + abs(dy) <= max_cheat
    ]

    dist_end = dist[end[1]][end[0]]
    result = 0
    for i, p in enumerate(path):
        for d in cheat_ds:
            c = (p[0] + d[0], p[1] + d[1])
            if in_bounds(m, c) and dist[c[1]][c[0]] != 0:
                cp_len = i + (dist_end - dist[c[1]][c[0]]) + abs(d[0]) + abs(d[1])
                if len(path) - cp_len >= 100:
                    result += 1
    return result


m = []
for y, line in enumerate(fileinput.input()):
    if line == "\n":
        break
    m.append(list(line.removesuffix("\n")))
    try:
        start = (line.index("S"), y)
    except ValueError:
        pass
    try:
        end = (line.index("E"), y)
    except ValueError:
        pass

dist, path = shortest_path(m, start, end)
for max_cheat_len in (2, 20):
    print(solve(m, end, dist, path, max_cheat_len))
