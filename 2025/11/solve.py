#!/usr/bin/env python
import fileinput
import collections


def count(e: dict[str, list[str]], start: str) -> int:
    visited = set()
    stack = [(start, False)]
    topsorted = list()
    while stack:
        u, finishing = stack.pop()
        if finishing:
            topsorted.append(u)
            continue
        else:
            if u in visited:
                continue
            stack.append((u, True))
        visited.add(u)
        for v in e.get(u, []):
            stack.append((v, False))

    counts = collections.defaultdict(int)
    counts[start] = 1
    for u in reversed(topsorted):
        for v in e.get(u, []):
            counts[v] += counts[u]
    return counts


def solve(e: dict[str, list[str]]) -> None:
    print(count(e, "you")["out"])

    from_dac = count(e, "dac")
    from_fft = count(e, "fft")
    from_svr = count(e, "svr")
    print(
        from_svr["dac"] * from_dac["fft"] * from_fft["out"]
        if "fft" in from_dac
        else from_svr["fft"] * from_fft["dac"] * from_dac["out"]
    )


solve({(t := s.split(":"))[0]: t[1].strip().split(" ") for s in fileinput.input()})
