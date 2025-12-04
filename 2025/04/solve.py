#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp

m = jnp.array(
    [[1 if c == "@" else 0 for c in s] for s in fileinput.input()], dtype=jnp.int32
)
kernel = jnp.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)


def find_accessible():
    neighbour_count = jax.scipy.signal.convolve(m, kernel, "same")
    return (neighbour_count < 4) & (m == 1)


accesible = find_accessible()
num_accessible = accesible.sum()
print(num_accessible)

removed = 0
while num_accessible != 0:
    m -= accesible
    removed += num_accessible
    accesible = find_accessible()
    num_accessible = accesible.sum()

print(removed)
