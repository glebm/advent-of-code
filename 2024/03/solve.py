#!/usr/bin/env python
import fileinput
import re


_MUL = re.compile(r"mul\((\d{1,3}),(\d{1,3})\)")
_TOGGLE = re.compile(r"do(?:n't)?\(\)")


def process_mul(s):
    return sum(int(m[1]) * int(m[2]) for m in re.finditer(_MUL, s))


result = 0
result2 = 0
enabled = True
for line in fileinput.input():
    result += process_mul(line)

    end = 0
    for m in re.finditer(_TOGGLE, line):
        if enabled:
            result2 += process_mul(line[end : m.start()])
        enabled = m[0] == "do()"
        end = m.end()
    if enabled:
        result2 += process_mul(line[end:])

print(result)
print(result2)
