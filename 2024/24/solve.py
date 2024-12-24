#!/usr/bin/env python
import fileinput
import re
import collections
import dataclasses
from typing import Callable

_GATE_FNS = {
    "AND": lambda x, y: x & y,
    "OR": lambda x, y: x | y,
    "XOR": lambda x, y: x ^ y,
}


@dataclasses.dataclass(frozen=True)
class Gate:
    a: str
    b: str
    op: Callable[[int, int], int]
    op_name: str


kx = lambda idx: f"x{idx:02}"
ky = lambda idx: f"y{idx:02}"


def eval(regs: dict[str, int], gates: dict[str, Gate], out: str, gate: Gate):
    if out in regs:
        return regs[out]
    regs[out] = -1  # cycle detection
    a = regs[gate.a] if gate.a in regs else eval(regs, gates, gate.a, gates[gate.a])
    if a == -1:
        return -1
    b = regs[gate.b] if gate.b in regs else eval(regs, gates, gate.b, gates[gate.b])
    if b == -1:
        return b
    result = gate.op(a, b)
    regs[out] = result
    return result


def solve(regs: dict[str, int], gates: dict[str, Gate]):
    result = 0
    for out, gate in gates.items():
        if out.startswith("z"):
            bit = eval(regs, gates, out, gate)
            if bit == -1:
                return -1
            result |= bit << int(out[1:])
    return result


def check(r: dict[str, int], swp: dict[str, Gate], i: int):
    return (
        solve({**r, kx(i): 1}, swp) == 1 << i and solve({**r, ky(i): 1}, swp) == 1 << i
    )


def solve2(gates: dict[str, Gate]):
    num_bits = max(
        int(out.removeprefix("z")) for out in gates.keys() if out.startswith("z")
    )

    inputs = collections.defaultdict(list)
    gate_types: dict[str, str] = {}
    for out, gate in gates.items():
        inputs[out].append(gate.a)
        inputs[out].append(gate.b)
        gate_types[out] = gate.op_name
    for i in range(num_bits):
        gate_types[kx(i)] = "IN"
        gate_types[ky(i)] = "IN"
    for v in inputs.values():
        v.sort(key=lambda x: gate_types[x])

    # intermediate z outputs should have an OR and an XOR input
    bad_gates = []
    for i in range(2, num_bits):
        z = f"z{i:02}"
        if gate_types[z] != "XOR" or (
            input_types := tuple(gate_types[g] for g in inputs[z])
        ) != ("OR", "XOR"):
            # print(z, input_types, "->", gate_types[z])
            bad_gates.append(z)

    r = {f"{k}{b:02}": 0 for b in range(num_bits) for k in ("x", "y")}

    # Find the right swap for each sink gate by validating a few additions.
    # print("bad gates: ", bad_gates)
    result = []
    for bad_gate in bad_gates:
        i = int(bad_gate[1:])
        if check(r, gates, i):
            # print("already ok", bad_gate)
            continue
        found = False
        for candidate in [bad_gate, *inputs[bad_gate]]:
            for out, gate in gates.items():
                swp = {**gates, candidate: gate, out: gates[candidate]}
                if check(r, swp, i):
                    rt = {**r}
                    for j in range(i):
                        rt[kx(j)] = rt[ky(j)] = 1
                    if solve(rt, swp) == (1 << (i + 1)) - 2:
                        gates = swp
                        result.append(candidate)
                        result.append(out)
                        # print(f"{bad_gate}: {candidate} <-> {out}")
                        found = True
                        break
            if found:
                break
        if not found:
            return "no candidate for {bad_gate}"

    return ",".join(sorted(result))


regs = {}
gates = {}
input = fileinput.input()
for line in input:
    if line == "\n":
        break
    k, v = line.split(": ")
    regs[k] = int(v)
gate_re = re.compile(r"^(\w+) (AND|X?OR) (\w+) -> (\w+)\n$")
for line in input:
    m = gate_re.match(line)
    gates[m[4]] = Gate(m[1], m[3], _GATE_FNS[m[2]], m[2])
print(solve(regs, gates))
print(solve2(gates))
