#!/usr/bin/env python
import fileinput
import dataclasses
import collections
import heapq


@dataclasses.dataclass(unsafe_hash=True, frozen=True)
class Node:
    pos: tuple[int, int]
    dir: tuple[int, int]


@dataclasses.dataclass(unsafe_hash=True, order=True, frozen=True)
class NodeWithCost:
    node: Node = dataclasses.field(compare=False)
    cost: int

    def __iter__(self):
        return iter((self.node, self.cost))


def neighbours(m: list[list[str]], node: Node):
    newpos = (node.pos[0] + node.dir[0], node.pos[1] + node.dir[1])
    if m[newpos[1]][newpos[0]] != "#":
        yield NodeWithCost(Node(newpos, node.dir), 1)
    yield NodeWithCost(Node(node.pos, (-node.dir[1], node.dir[0])), 1000)
    yield NodeWithCost(Node(node.pos, (node.dir[1], -node.dir[0])), 1000)


_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def solve(m: list[list[str]], start: Node, end: tuple[int, int]):
    queue = [NodeWithCost(start, 0)]
    best = {}
    done = set()
    pred = collections.defaultdict(list)
    while queue:
        u, cu = heapq.heappop(queue)
        if u in done:
            continue
        for v, cv in neighbours(m, u):
            if v in done:
                continue
            if v not in best or best[v] >= cu + cv:
                if v in best and best[v] > cu + cv:
                    pred[v].clear()
                pred[v].append(u)
                best[v] = cu + cv
                heapq.heappush(queue, NodeWithCost(v, cu + cv))
        done.add(u)

    bestScore = min(best[Node(end, dir)] for dir in _DIRS if Node(end, dir) in best)
    bestNodes = []
    frontier = [
        Node(end, dir)
        for dir in _DIRS
        if Node(end, dir) in pred and best[Node(end, dir)] == bestScore
    ]
    while frontier:
        v = frontier.pop()
        bestNodes.append(v)
        if v == start:
            continue
        for u in pred[v]:
            frontier.append(u)

    return (
        bestScore,
        len(set(node.pos for node in bestNodes)),
    )


m = []
for y, line in enumerate(fileinput.input()):
    if line == "\n":
        break
    m.append(list(line.removesuffix("\n")))
    try:
        start = (line.index("S"), y)
    except ValueError:
        pass
    try:
        end = (line.index("E"), y)
    except ValueError:
        pass

part1, part2 = solve(m, Node(start, (1, 0)), end)
print(part1)
print(part2)
