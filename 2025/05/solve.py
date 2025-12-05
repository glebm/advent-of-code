#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp
import itertools

jax.config.update("jax_enable_x64", True)

input = fileinput.input()
rs = jnp.array(
    [
        [int((p := s.split("-"))[0]), int(p[1]) + 1]
        for s in itertools.takewhile(lambda s: s != "\n", input)
    ],
    dtype=jnp.int64,
)
xs = jnp.array(list(map(int, input)), dtype=jnp.int64)
print(jax.vmap(lambda x: ((x >= rs[:, 0]) & (x < rs[:, 1])).any())(xs).sum())

sorted = jnp.sort(rs, axis=0)
prev_end = jnp.insert(sorted[:-1, 1], 0, 0, axis=0)
print((sorted[:, 1] - jnp.maximum(prev_end, sorted[:, 0])).sum())
