#!/usr/bin/env python
import fileinput


def rot(a):
    return list(map("".join, zip(*a[::-1])))


def check(a, b, c):
    return a == "M" and b == "A" and c == "S"


def check2(a, b):
    return (a == "M" and b == "S") or (a == "S" and b == "M")


m = [line.removesuffix("\n") for line in fileinput.input()]
result = 0
for a in [m, rot(m), rot(rot(m)), rot(rot(rot(m)))]:
    for y, row in enumerate(a):
        end = 0
        while (x := row.find("X", end)) != -1:
            if x + 3 < len(row):
                result += check(a[y][x + 1], a[y][x + 2], a[y][x + 3])
            if y + 3 < len(a) and x + 3 < len(row):
                result += check(a[y + 1][x + 1], a[y + 2][x + 2], a[y + 3][x + 3])
            end = x + 1
print(result)

result2 = 0
a = m
for y, row in enumerate(a):
    end = 0
    while (x := row.find("A", end)) != -1:
        if x > 0 and x + 1 < len(row) and y > 0 and y + 1 < len(row):
            result2 += check2(a[y - 1][x - 1], a[y + 1][x + 1]) and check2(
                a[y + 1][x - 1], a[y - 1][x + 1]
            )
        end = x + 1
print(result2)
