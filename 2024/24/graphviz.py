#!/usr/bin/env python
import fileinput
import re
import subprocess

input = fileinput.input()
for line in input:
    if line == "\n":
        break

_SHAPE = {
    "AND": "box",
    "OR": "diamond",
    "XOR": "triangle",
}

gate_re = re.compile(r"^(\w+) (AND|X?OR) (\w+) -> (\w+)\n$")
with open("input.dot", "w") as f:
    f.write("digraph {\n")
    xs = []
    ys = []
    zs = []
    for line in input:
        a, op, b, out = gate_re.match(line).groups()
        for node in (a, b):
            if node.startswith("x"):
                xs.append(node)
            elif node.startswith("y"):
                ys.append(node)
        if out.startswith("z"):
            zs.append(out)
        f.write(f'  {out} [shape="{_SHAPE[op]}" label="{op}\\n{out}"]\n')
        f.write(f"  {a} -> {out}\n")
        f.write(f"  {b} -> {out}\n")
    zs.sort()
    for x in xs:
        f.write(f"  {x} [style=filled fillcolor=blue rank=source]\n")
    for y in ys:
        f.write(f"  {y} [style=filled fillcolor=green rank=source]\n")
    for z in zs:
        f.write(f"  {z} [style=filled fillcolor=red rank=sink]\n")
    f.write("}")

subprocess.call(["dot", "-Kdot", "-Gnewrank=true", "-Tpng", "-oinput.png", "input.dot"])
