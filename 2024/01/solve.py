#!/usr/bin/env python
import fileinput
import jax.numpy as jnp

a = jnp.array([list(map(int, s.split())) for s in fileinput.input()], dtype=jnp.int32)
a = a.transpose().sort(axis=1)
print(jnp.sum(jnp.abs(a[0] - a[1])))
left = jnp.searchsorted(a[1], a[0], side="left")
right = jnp.searchsorted(a[1], a[0], side="right")
print(jnp.sum((right - left) * a[0]))
