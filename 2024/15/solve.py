#!/usr/bin/env python
import fileinput
import copy
import time
import numpy as np
import collections


def vec2(x, y):
    return np.array([x, y], dtype=np.int32)


def getv(m, p):
    return m[p[1]][p[0]]


def setv(m, p, v):
    m[p[1]][p[0]] = v


_MOVES = {"<": vec2(-1, 0), ">": vec2(1, 0), "^": vec2(0, -1), "v": vec2(0, 1)}


def solve(m: list[list[str]], robot: np.ndarray, moves: list[str]):
    for move in moves:
        dir = _MOVES[move]
        newpos = robot + dir
        if getv(m, newpos) == "#":
            continue
        if getv(m, newpos) == ".":
            setv(m, robot, ".")
            setv(m, newpos, "@")
            robot = newpos
            continue
        newo = newpos + dir
        while getv(m, newo) == "O":
            newo += dir
        if getv(m, newo) == "#":
            continue
        setv(m, robot, ".")
        setv(m, newpos, "@")
        setv(m, newo, "O")
        robot = newpos

    return sum(
        y * 100 + x for y, row in enumerate(m) for x, c in enumerate(row) if c == "O"
    )


def makewide(m):
    result = []
    for line in m:
        r = []
        for c in line:
            if c == "@":
                r.append("@")
                r.append(".")
            elif c == "O":
                r.append("[")
                r.append("]")
            else:
                r.append(c)
                r.append(c)
        result.append(r)
    return result


def is_box(m, p):
    c = getv(m, p)
    return c == "[" or c == "]"


def box_offset(m, p):
    return vec2(1, 0) if getv(m, p) == "[" else vec2(-1, 0)


def can_move(m, p, dir):
    offset = box_offset(m, p)
    return getv(m, p + dir) != "#" and getv(m, p + dir + offset) != "#"


def canonical_box(m, p):
    return p if getv(m, p) == "[" else p + vec2(-1, 0)


def wide_move(m, p, dir):
    offset = box_offset(m, p)
    c1 = getv(m, p)
    c2 = getv(m, p + offset)
    setv(m, p, ".")
    setv(m, p + offset, ".")
    setv(m, p + dir, c1)
    setv(m, p + dir + offset, c2)


def solve2(m: list[list[str]], robot: np.ndarray, moves: list[str]):
    m = makewide(m)
    robot[0] *= 2
    for move in moves:
        # time.sleep(0.3)
        # print("\033c\033[3J", end='')
        # print("\n".join("".join(s) for s in m))
        # print(move)
        dir = _MOVES[move]
        newpos = robot + dir
        if getv(m, newpos) == "#":
            continue
        if getv(m, newpos) == ".":
            setv(m, robot, ".")
            setv(m, newpos, "@")
            robot = newpos
            continue

        boxes = []
        queue = collections.deque([canonical_box(m, newpos)])
        visited = set()
        while queue:
            box = queue.popleft()
            if tuple(box) in visited:
                continue
            if not can_move(m, box, dir):
                boxes.clear()
                break
            visited.add(tuple(box))
            boxes.append(box)

            offset = box_offset(m, box)
            if is_box(m, box + dir):
                queue.append(canonical_box(m, box + dir))
            if is_box(m, box + offset + dir):
                queue.append(canonical_box(m, box + offset + dir))

        if not boxes:
            continue
        while boxes:
            wide_move(m, boxes.pop(), dir)

        setv(m, robot, ".")
        setv(m, newpos, "@")
        robot = newpos
    # print("\n".join("".join(s) for s in m))

    return sum(
        y * 100 + x for y, row in enumerate(m) for x, c in enumerate(row) if c == "["
    )


input = fileinput.input()
m = []
robot = None
for y, line in enumerate(input):
    if line == "\n":
        break
    m.append(list(line.removesuffix("\n")))
    try:
        x = line.index("@")
        robot = vec2(x, y)
    except ValueError:
        pass
moves = []
for line in input:
    moves += list(line.removesuffix("\n"))

print(solve(copy.deepcopy(m), robot, moves))
print(solve2(m, robot, moves))
