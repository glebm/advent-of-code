#!/usr/bin/env python
import fileinput
import jax.numpy as jnp
import jax.lax as lax

a = jnp.array(
    [(int(s[1:]) * (1 if s[0] == "R" else -1)) for s in fileinput.input()],
    dtype=jnp.int32,
)


def modulo_add(s, a):
    return (s + a) % 100


s = jnp.frompyfunc(modulo_add, nin=2, nout=1, identity=0).accumulate(
    jnp.insert(a, 0, 50)
)
print(len(s) - jnp.count_nonzero(s))


def count_pass_zero(row):
    s, a = row
    return jnp.where(a < 0, (lax.rem(s - 100, 100) + a) // -100, (s + a) // 100)


r = jnp.apply_along_axis(count_pass_zero, 0, jnp.vstack([s[:-1], a]))

print(r.sum())
