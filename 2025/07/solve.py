#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def solve(m):
    beams = m[0]
    num_splits = 0
    for i in range(1, len(m)):
        splitters = m[i, :] != 0
        split_points = beams * splitters
        num_splits += jnp.count_nonzero(split_points)
        left_splits = jnp.roll(split_points, -1)
        right_splits = jnp.roll(split_points, 1)
        beams = beams * ~splitters + left_splits + right_splits
    return (num_splits, beams.sum())


print(
    *solve(jnp.array([[int(c != ".") for c in s[:-1]] for s in fileinput.input()])),
    sep="\n"
)
