#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp
import jax.lax as lax
import itertools
import numpy as np

jax.config.update("jax_enable_x64", True)

_BATCH_SIZE = 2**26


def num_digits(x):
    return jnp.floor(jnp.log10(x)).astype(jnp.int64) + 1


def is_repeated_twice(x):
    q, r = jnp.divmod(x, jnp.pow(10, num_digits(x) // 2))
    return q == r


def is_repeated_n(x, rep_len):
    p10 = jnp.pow(10, rep_len)
    rep = x % p10

    def cond_fun(r):
        return (r[0] > 0) & r[1]

    def body_fun(r):
        q, r = jnp.divmod(r[0], p10)
        return (q, r == rep)

    return lax.while_loop(cond_fun, body_fun, init_val=(x, True))[1]


def is_repeated_any(x):
    d = num_digits(x)

    def body_fun(rep_len, r):
        return r + jnp.where(d % rep_len == 0, is_repeated_n(x, rep_len), 0)

    return lax.fori_loop(1, d // 2 + 1, body_fun, init_val=0) != 0


ranges = jnp.array(
    [list(map(int, r.split("-"))) for s in fileinput.input() for r in s.split(",")],
    dtype=jnp.int64,
)

for f in (is_repeated_twice, is_repeated_any):
    i = 0
    sum = 0
    padded_batch = np.empty(_BATCH_SIZE, dtype=np.int64)
    for batch in itertools.batched(
        itertools.chain(*[range(a, b + 1) for (a, b) in ranges]), _BATCH_SIZE
    ):
        padded_batch[: len(batch)] = batch
        padded_batch[len(batch):] = 0
        sum += jax.vmap(lambda i: jnp.where(f(i), i, 0))(padded_batch).sum()

    print(sum)
