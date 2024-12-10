#!/usr/bin/env python
import fileinput


def solve(m, x, y, visited, part1):
    if part1:
        visited.add((x, y))
    if m[y][x] == 9:
        return 1
    score = 0
    w = len(m[0])
    h = len(m)
    next_val = m[y][x] + 1
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx = x + dx
        ny = y + dy
        if part1 and (nx, ny) in visited:
            continue
        if nx < 0 or nx >= w or ny < 0 or ny >= h or m[ny][nx] != next_val:
            continue
        score += solve(m, nx, ny, visited, part1)
    return score


m = [list(map(int, line.removesuffix("\n"))) for line in fileinput.input()]
for part1 in [True, False]:
    print(
        sum(
            solve(m, x, y, set(), part1)
            for y, row in enumerate(m)
            for x, c in enumerate(row)
            if c == 0
        )
    )
