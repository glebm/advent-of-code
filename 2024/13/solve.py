#!/usr/bin/env python
import fileinput
import re
from ortools.sat.python import cp_model

# https://developers.google.com/optimization/cp/cp_solver
def solve(a, b, p):
    model = cp_model.CpModel()
    na = model.new_int_var(0, 10000000000000, "a")
    nb = model.new_int_var(0, 10000000000000, "b")
    model.add(a[0] * na + b[0] * nb == p[0])
    model.add(a[1] * na + b[1] * nb == p[1])
    objective = 3 * na + nb
    model.minimize(objective)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return solver.value(objective)
    return 0


def parse(s):
    return tuple(map(int, re.search(r"(\d+).*?(\d+)", s).groups()))


result = 0
result2 = 0
input = fileinput.input()
while True:
    try:
        a = parse(input.__next__())
        b = parse(input.__next__())
        p = parse(input.__next__())
        result += solve(a, b, p)
        result2 += solve(a, b, (p[0] + 10000000000000, p[1] + 10000000000000))
        input.__next__()
    except StopIteration:
        break
print(result)
print(result2)
