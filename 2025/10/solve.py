#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp
import dataclasses
import functools
from ortools.sat.python import cp_model


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Machine:
    target: int  # a bitmask
    buttons: jax.Array  # bitmasks
    joltage: jax.Array


@jax.jit
def part1(m: Machine):
    def check(mask):
        result = jnp.bitwise_xor.reduce(
            jnp.where(mask & (1 << jnp.arange(len(m.buttons))), m.buttons, 0)
        )
        return jnp.where(m.target == result, jnp.bitwise_count(mask), 127)

    clicks = jax.vmap(check)(jnp.arange(2 ** len(m.buttons)))
    return jnp.min(clicks).astype(jnp.int32)


# https://developers.google.com/optimization/cp/cp_solver
def part2_ortools(joltage, buttons):
    model = cp_model.CpModel()
    btn_syms = [model.new_int_var(0, 1000, f"b{i}") for i in range(len(buttons))]
    btns_mask = (buttons[:, None] & (1 << jnp.arange(len(joltage)))).astype(bool)
    for i in range(len(joltage)):
        model.add(
            functools.reduce(
                lambda a, b: a + b,
                [btn_syms[j] for j in range(len(buttons)) if btns_mask[j, i]],
            )
            == int(joltage[i])
        )
    objective = functools.reduce(lambda a, b: a + b, btn_syms)
    model.minimize(objective)
    solver = cp_model.CpSolver()
    solver.solve(model)
    return solver.value(objective)


def solve(machines: list[Machine]) -> None:
    result1 = 0
    result2 = 0
    for m in machines:
        result1 += part1(m)
        result2 += part2_ortools(m.joltage, m.buttons)
    print(result1, result2, sep="\n")


def parse_machine(machine_str: str) -> Machine:
    parts = machine_str.split(" ")
    return Machine(
        target=int(
            "".join(parts[0][1:-1].translate(str.maketrans(".#", "01"))[::-1]), 2
        ),
        buttons=jnp.array(
            [
                jnp.bitwise_or.reduce(
                    1 << jnp.array([int(x) for x in p[1:-1].split(",")])
                )
                for p in parts[1:-1]
            ]
        ),
        joltage=jnp.array([int(p) for p in parts[-1][1:-2].split(",")]),
    )


solve([parse_machine(s) for s in fileinput.input()])
