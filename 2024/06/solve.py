#!/usr/bin/env python
import fileinput
import dataclasses
from typing import List, Tuple


@dataclasses.dataclass(unsafe_hash=True)
class Point:
    x: int
    y: int


def initial_pos(m: List[List[str]]) -> Point:
    for y, row in enumerate(m):
        for x, c in enumerate(row):
            if c == "^":
                return Point(x, y)
    raise ("not found")


def rot90(dir: Tuple[int, int]):
    return (-dir[1], dir[0])


def solve1(m: List[List[str]]):
    pos = initial_pos(m)
    dir = (0, -1)
    m[pos.y][pos.x] = "X"
    while pos.x != 0 and pos.x + 1 != len(m[0]) and pos.y != 0 and pos.y + 1 != len(m):
        while (next_pos := Point(pos.x + dir[0], pos.y + dir[1])) and m[next_pos.y][
            next_pos.x
        ] == "#":
            dir = rot90(dir)
        pos = next_pos
        m[pos.y][pos.x] = "X"
    return sum(1 for row in m for c in row if c == "X")


def is_loop(m: List[List[str]], pos: Point, obstacle_pos: Point):
    dir = (0, -1)
    m[pos.y][pos.x] = "X"
    m[obstacle_pos.y][obstacle_pos.x] = "#"
    visited = set()
    visited.add((pos, dir))
    while pos.x != 0 and pos.x + 1 != len(m[0]) and pos.y != 0 and pos.y + 1 != len(m):
        while (next_pos := Point(pos.x + dir[0], pos.y + dir[1])) and m[next_pos.y][
            next_pos.x
        ] == "#":
            dir = rot90(dir)
        pos = next_pos
        if (pos, dir) in visited:
            m[obstacle_pos.y][obstacle_pos.x] = "X"
            return True
        visited.add((pos, dir))

    m[obstacle_pos.y][obstacle_pos.x] = "."
    return False


def solve2(m):
    pos = initial_pos(m)
    return sum(
        1
        for y, row in enumerate(m)
        for x, c in enumerate(row)
        if pos != Point(x, y) and c != "#" and is_loop(m, pos, Point(x, y))
    )


m = [list(line.removesuffix("\n")) for line in fileinput.input()]
print(solve1(m))

m = [list(line.removesuffix("\n")) for line in fileinput.input()]
print(solve2(m))
