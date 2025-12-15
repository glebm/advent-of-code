#!/usr/bin/env python
import fileinput
import numpy as np
import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class Region:
    w: int
    h: int
    shapes: list[int]


def parse_region(s: str) -> Region:
    size, shapes = s.split(": ")
    w, h = map(int, size.split("x"))
    return Region(w, h, list(map(int, shapes.split(" "))))


def parse_shape(rows: list[str]) -> np.array:
    return np.array(
        [list(map(int, row.translate(str.maketrans(".#", "01")))) for row in rows]
    )


input_parts = "".join(fileinput.input()).split("\n\n")
shapes = np.array([parse_shape(shape.split("\n")[1:]) for shape in input_parts[:-1]])
regions = [
    parse_region(region) for region in input_parts[-1].removesuffix("\n").split("\n")
]

result = 0
for region in regions:
    required_area = 0
    for shape_idx, shape_count in enumerate(region.shapes):
        shape = shapes[shape_idx]
        required_area += shape.shape[0] * shape.shape[1] * shape_count
    result += int(required_area <= region.w * region.h)
print(result)
