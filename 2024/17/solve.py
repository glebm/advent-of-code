#!/usr/bin/env python
import dataclasses
import itertools
import pathlib
import re
import sys


@dataclasses.dataclass
class State:
    regs: list[int, int, int]
    ip: int
    out: list[int]


def combo(s, op):
    return op if op <= 3 else s.regs[op - 4]


def adv(s, op):
    s.regs[0] = s.regs[0] // (1 << op)
    s.ip += 2


def bxl(s, op):
    s.regs[1] = s.regs[1] ^ op
    s.ip += 2


def bst(s, op):
    s.regs[1] = op % 8
    s.ip += 2


def jnz(s, op):
    if s.regs[0] != 0:
        s.ip = op
    else:
        s.ip += 2


def bxc(s, op):
    s.regs[1] = s.regs[1] ^ s.regs[2]
    s.ip += 2


def out(s, op):
    s.out.append(op % 8)
    s.ip += 2


def bdv(s, op):
    s.regs[1] = s.regs[0] // (1 << op)
    s.ip += 2


def cdv(s, op):
    s.regs[2] = s.regs[0] // (1 << op)
    s.ip += 2


_INSTRUCTIONS = [adv, bxl, bst, jnz, bxc, out, bdv, cdv]
_COMBO_OP = [True, False, True, False, False, True, True, True]
_NAMES = ["adv", "bxl", "bst", "jnz", "bxc", "out", "bdv", "cdv"]


def solve(regs, program):
    s = State(list(regs), 0, [])
    while s.ip < len(program):
        inst, op = program[s.ip], program[s.ip + 1]
        _INSTRUCTIONS[inst](s, combo(s, op) if _COMBO_OP[inst] else op)
    return s.out


def inst_to_asm(inst, op):
    if _COMBO_OP[inst] and op > 3:
        return f"{_NAMES[inst]} %{chr(ord('A') + (op - 4))}"
    return f"{_NAMES[inst]} ${op}"


def to_asm(program):
    result = []
    for i in range(0, len(program), 2):
        result.append(inst_to_asm(program[i], program[i + 1]))
    return "\n".join(result)


input = pathlib.Path(sys.argv[1]).read_text().split("\n\n")
regs = list(map(int, re.findall(r"\d+", input[0])))
program = list(map(int, re.findall(r"\d+", input[1])))

print(",".join(list(map(str, solve(regs, program)))))

# print(to_asm(program))

# If you inspect the program above, you will notice that:
# 1. It's a single loop until `a` becomes 0.
# 2. `a` is divided by 8 each iteration (a = a // 8).
# 3. `b` and `c` are only used as temporaries within the loop.
#
# We can find the candidate values of a in the last iteration and limit
# the values of a in the preceding iterations based on that.
#
# In the last iteration, `a` must be `<8` and `out[0] == program[-1]`.
# We can then expand the candidates for subsequent iterations.
candidates = [0]
for step in range(len(program) - 1, -1, -1):
    candidates = [
        a * 8 + i
        for i in range(8)
        for a in candidates
        if solve([a * 8 + i, 0, 0], program)[0] == program[step]
    ]

print(min(candidates))
