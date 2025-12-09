#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def in_bounds(a, b, x):
    return (x >= jnp.minimum(a, b)) & (x <= jnp.maximum(a, b))


def east_ray_crossings(p, s):
    x, y = p
    x1, y1, x2, y2 = s.T
    return in_bounds(y1, y2, y) & (x < jnp.minimum(x1, x2))


def is_on_edge(p, s):
    x, y = p
    x1, y1, x2, y2 = s.T
    return jnp.where(
        y1 == y2,
        (y == y1) & in_bounds(x1, x2, x),
        (x == x1) & in_bounds(y1, y2, y),
    )


@jax.jit
def solve(ps: jax.Array):
    pairwise_deltas = ps - ps[:, None]
    areas = jnp.prod(jnp.abs(pairwise_deltas) + 1, axis=-1)
    _, flat_indices = jax.lax.sort_key_val(
        -jnp.triu(areas, -1).flatten(), jnp.arange(areas.size)
    )
    pairs = jnp.asarray(jnp.unravel_index(flat_indices, areas.shape)).T
    edges = jnp.hstack((jnp.roll(ps, 1, 0), ps))

    def is_point_in_shape(p):
        crossings = east_ray_crossings(p, edges)
        on_edge = is_on_edge(p, edges)
        return jnp.any(on_edge) | (crossings.sum() % 2 != 0)

    def is_invalid(a, b):
        x_range = (jnp.minimum(a[0], b[0]), jnp.maximum(a[0], b[0]) + 1)
        y_range = (jnp.minimum(a[1], b[1]), jnp.maximum(a[1], b[1]) + 1)

        def check_range(begin, range_len, c, is_x):
            return jax.lax.while_loop(
                lambda it: (it[0] < begin + range_len) & it[1],
                lambda it: (
                    it[0] + 1,
                    is_point_in_shape(jnp.array([it[0], c] if is_x else [c, it[0]])),
                ),
                (begin, True),
            )[1]

        return ~jnp.all(
            jnp.array(
                [
                    check_range(x_range[0], x_range[1] - x_range[0], a[1], True),
                    check_range(x_range[0], x_range[1] - x_range[0], b[1], True),
                    check_range(y_range[0], y_range[1] - y_range[0], a[0], False),
                    check_range(y_range[0], y_range[1] - y_range[0], b[0], False),
                ]
            )
        )

    invalid = jax.vmap(lambda pair: is_invalid(*ps[pair]))(pairs)
    best_pair = jax.lax.while_loop(
        lambda i: invalid[i] & (i < pairs.shape[0]), lambda i: i + 1, 0
    )

    return (areas[*pairs[0]], areas[*pairs[best_pair]])


points = jnp.array([list(map(int, s.split(","))) for s in fileinput.input()])
print(*solve(points), sep="\n")
