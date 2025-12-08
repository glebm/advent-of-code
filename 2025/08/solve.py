#!/usr/bin/env python
import fileinput
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@jax.jit
def solve(points: jax.Array):
    n = points.shape[0]

    pairwise_deltas = points - points[:, None]  # broadcasting trick
    dists = jnp.square(pairwise_deltas).sum(axis=-1)

    # Mask out diagonal and lower triangle to avoid self-edge and duplicates:
    dists = jnp.triu(dists, -1) + (jnp.max(dists) + 1) * jnp.tri(n, dtype=jnp.int64)

    # Sort the distances and get the (N, 2) edge list:
    _, flat_indices = jax.lax.sort_key_val(dists.flatten(), jnp.arange(n * n))
    edges = jnp.asarray(jnp.unravel_index(flat_indices, dists.shape)).T

    def add_edge(it):
        """Adds a single edge and returns new labels and next edge index."""
        labels, edge_idx = it
        u = labels[edges[edge_idx][0]]
        v = labels[edges[edge_idx][1]]
        new_labels = jnp.where(labels == jnp.maximum(u, v), jnp.minimum(u, v), labels)
        return new_labels, edge_idx + 1

    # Add edges for part 1:
    it = (jnp.arange(n), 0)  # (labels, number of edges added)
    it = jax.lax.while_loop(lambda l: l[1] != (10 if n < 50 else 1000), add_edge, it)
    ans1 = jax.lax.top_k(jnp.bincount(it[0], length=n), 3)[0].prod()

    # Continue adding edges for part 2:
    it = jax.lax.while_loop(lambda l: jnp.any(l[0] != 0), add_edge, it)
    edge = edges[it[1] - 1]
    part2 = points[edge[0]][0] * points[edge[1]][0]

    return (ans1, part2)


points = jnp.array([list(map(int, s.split(","))) for s in fileinput.input()])
print(*solve(points), sep="\n")
