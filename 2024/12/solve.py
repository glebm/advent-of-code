#!/usr/bin/env python
import fileinput
import itertools


def oob(x, y, w, h):
    return x < 0 or x >= w or y < 0 or y >= h


def debug_char(cur, vertical):
    if cur != " ":
        return "🭽"
    return "▏" if vertical else "▔"


def debug_edges(edges, w, h):
    print()
    m = [[" "] * (w + 1) for i in range(h + 1)]
    for edge in edges:
        a, b = edge
        x, y = a
        m[y][x] = debug_char(m[y][x], x == b[0])
    print("\n".join("".join(xs) for xs in m))


def visit(m, start, seen):
    w = len(m[0])
    h = len(m)
    stack = [start]
    c = m[start[1]][start[0]]
    seen[start[1]][start[0]] = True
    area = 0
    perimeter = 0
    # fence edges, a pair of vertices always from from top/left to bottom/right
    edges = set()
    while stack:
        x, y = stack.pop()
        area += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            nc = "." if oob(nx, ny, w, h) else m[ny][nx]
            if nc == c:
                if not seen[ny][nx]:
                    seen[ny][nx] = True
                    stack.append((nx, ny))
            else:
                perimeter += 1
                if dx < 0 or dy < 0:
                    edges.add(((x, y), (x - dy, y - dx)))
                else:
                    edges.add(((x + dx, y + dy), (x + 1, y + 1)))

    # if c == 'A':
    #     debug_edges(edges, w, h)
    return area, perimeter, count_sides(edges)


def count_sides(edges):
    num_lines = len(edges)
    orig_edges = set(edges)
    while edges:
        edge = edges.pop()
        for dx, dy in (
            [(-1, 0), (1, 0)] if edge[0][0] != edge[1][0] else [(0, -1), (0, 1)]
        ):
            na, nb = edge
            # Grow the edge in this direction while we can:
            while True:
                na2 = (na[0] + dx, na[1] + dy)
                nb2 = (nb[0] + dx, nb[1] + dy)
                if (na2, nb2) not in edges:
                    break

                # Intersections are not allowed, check for a cross:
                cx, cy = na if (dx < 0 or dy < 0) else nb
                adx = abs(dx)
                ady = abs(dy)
                if ((cx, cy), (cx + ady, cy + adx)) in orig_edges and (
                    (cx - ady, cy - adx),
                    (cx, cy),
                ) in orig_edges:
                    break

                edges.remove((na2, nb2))
                na, nb = na2, nb2
                num_lines -= 1
            # if na != edge[0] or nb != edge[1]:
            #     debug_edges(edges, 10, 10)
    return num_lines


m = [list(list(line.removesuffix("\n"))) for line in fileinput.input()]
seen = [[False] * len(m[0]) for i in range(len(m))]
result = 0
result2 = 0
for y, x in itertools.product(range(len(m)), range(len(m[0]))):
    if not seen[y][x]:
        area, perimeter, sides = visit(m, (x, y), seen)
        # print(area, perimeter, sides)
        result += area * perimeter
        result2 += area * sides
print(result)
print(result2)
