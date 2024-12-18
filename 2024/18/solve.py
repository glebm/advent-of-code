#!/usr/bin/env python
import fileinput


def solve(obstacles, w, h):
    frontier = [(0, 0)]
    dist = [[0] * w for _ in range(h)]
    while frontier:
        new_frontier = []
        for u in frontier:
            if u == (w - 1, h - 1):
                return dist[u[1]][u[0]]
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                v = (u[0] + d[0], u[1] + d[1])
                if (
                    (v[0] >= 0 and v[1] >= 0 and v[0] < w and v[1] < h)
                    and dist[v[1]][v[0]] == 0
                    and v not in obstacles
                ):
                    new_frontier.append(v)
                    dist[v[1]][v[0]] = dist[u[1]][u[0]] + 1
        frontier = new_frontier


obstacles = [tuple(map(int, line.split(","))) for line in fileinput.input()]

example = False
if example:
    print(solve(set(obstacles[:12]), 7, 7))
else:
    print(solve(set(obstacles[:1024]), 71, 71))

for i in range(len(obstacles)):
    if not (
        solve(set(obstacles[:i]), 7, 7)
        if example
        else solve(set(obstacles[:i]), 71, 71)
    ):
        print(f"{obstacles[i - 1][0]},{obstacles[i - 1][1]}")
        break
