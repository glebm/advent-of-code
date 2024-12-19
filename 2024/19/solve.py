#!/usr/bin/env python
import fileinput


class Solver:
    def __init__(self, patterns, aggregator):
        self.patterns = patterns
        self.cache = {"": True}
        self.aggregator = aggregator

    def solve(self, design: str):
        if design not in self.cache:
            self.cache[design] = self.aggregator(
                self.solve(design.removeprefix(p))
                for p in self.patterns
                if design.startswith(p)
            )
        return self.cache[design]


input = fileinput.input()
patterns = input.__next__().removesuffix("\n").split(", ")
input.__next__()
designs = [line.removesuffix("\n") for line in input]

solver = Solver(patterns, any)
print(sum(solver.solve(d) for d in designs))

solver2 = Solver(patterns, sum)
print(sum(solver2.solve(d) for d in designs))
