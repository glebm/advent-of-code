#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp
import itertools

jax.config.update("jax_enable_x64", True)

_ADD = 0
_MUL = 1
_SPACE = -1


@jax.jit
def part1(nums, ops):
    return jax.vmap(
        lambda x: jnp.where(x[-1] == _ADD, x[:-1].sum(), x[:-1].prod()), in_axes=1
    )(jnp.vstack((nums, ops))).sum()


lines = list(fileinput.input())
ops = jnp.array([[_MUL if s == "*" else _ADD for s in lines[-1].split()]])
part1_nums = jnp.array([list(map(int, s.split())) for s in lines[:-1]])
print(part1(part1_nums, ops))


def col_to_num(col):
    is_space = col == _SPACE
    pow10 = jnp.power(
        jnp.full_like(col, 10), jnp.arange(len(col))[::-1] - jnp.count_nonzero(is_space)
    )
    return jnp.dot(
        jnp.where(is_space, 0, col),
        jnp.roll(pow10, jnp.nonzero(col != _SPACE, size=1)[0]),
    )


m = jnp.array([[_SPACE if c == " " else int(c) for c in s[:-1]] for s in lines[:-1]])
nums = jax.vmap(col_to_num, in_axes=1)(m)
lists = [list(v) for k, v in itertools.groupby(nums.tolist(), lambda x: x != 0) if k]
pad_len = max(map(len, lists))
padded_num_lists = jnp.array(
    [v + ([op] * (pad_len - len(v))) for (v, op) in zip(lists, ops[0])]
)
print(part1(padded_num_lists.transpose(), ops))
