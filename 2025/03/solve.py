#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)


def part1(bank):
    d1 = jnp.argmax(bank[:-1])
    d2 = lax.fori_loop(
        d1 + 2,
        len(bank),
        body_fun=lambda i, r: jnp.where(bank[i] > bank[r], i, r),
        init_val=d1 + 1,
    )
    return bank[d1] * 10 + bank[d2]


def part2(bank):
    prev_d = jnp.argmax(bank[:-11]).astype(jnp.int64)
    r = bank[prev_d].astype(jnp.int64)
    for i in range(1, 12):
        d = lax.fori_loop(
            prev_d + 2,
            len(bank) - (11 - i),
            body_fun=lambda i, r: jnp.where(bank[i] > bank[r], i, r),
            init_val=prev_d + 1,
        )
        r = r * 10 + bank[d]
        prev_d = d
    return r


banks = jnp.array(
    [list(map(int, s.removesuffix("\n"))) for s in fileinput.input()], dtype=jnp.int32
)

print(jax.vmap(part1)(banks).sum())
print(jax.vmap(part2)(banks).sum())
