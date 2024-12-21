#!/usr/bin/env python
import fileinput

_DEBUG = False

# fmt: off
_NUMERIC_LAYOUT = {
    "7": (0, 0), "8": (1, 0), "9": (2, 0),
    "4": (0, 1), "5": (1, 1), "6": (2, 1),
    "1": (0, 2), "2": (1, 2), "3": (2, 2),
                 "0": (1, 3), "A": (2, 3),
}

_ARROW_LAYOUT = {
                 "^": (1, 0), "A": (2, 0),
    "<": (0, 1), "v": (1, 1), ">": (2, 1),
}
# fmt: on


def route_to(btn, layout: dict[str, tuple[int, int]], pos: tuple[int, int]):
    newpos = layout[btn]
    dx, dy = (newpos[0] - pos[0], newpos[1] - pos[1])
    b0, b1 = ("<" if dx < 0 else ">"), ("^" if dy < 0 else "v")
    return newpos, (b0, abs(dx)), (b1, abs(dy))


def is_legal(pos, newpos, numeric, xfirst):
    if numeric:
        if xfirst:
            return not (newpos[0] == 0 and pos[1] == 3)
        else:
            return not (newpos[1] == 3 and pos[0] == 0)
    else:
        if xfirst:
            return not (newpos[0] == 0 and pos[1] == 0)
        else:
            return not (newpos[1] == 0 and pos[0] == 0)


def press_one(
    positions: list[tuple[int, int]],
    i: int,
    btn: str,
    cache: dict[str, int],
    debug: bool,
) -> int:
    if debug:
        if i == len(positions):
            print("  " * i, btn)
        else:
            print("  " * i, "@", positions[i], btn)

    if i == len(positions):
        return 1
    cache_key = ";".join(f"{x},{y}" for x, y in positions[i:]) + f":{btn}"
    if cache_key in cache:
        result, new_positions = cache[cache_key]
        positions[i:] = new_positions
        return result

    newpos, btn_x, btn_y = route_to(
        btn, (_NUMERIC_LAYOUT if i == 0 else _ARROW_LAYOUT), positions[i]
    )
    btns_xfirst = [btn_x, btn_y, ("A", 1)]
    btns_yfirst = [btn_y, btn_x, ("A", 1)]

    if not is_legal(positions[i], newpos, numeric=(i == 0), xfirst=True):
        result = press(btns_yfirst, positions, i + 1, cache, debug)
    elif not is_legal(positions[i], newpos, numeric=(i == 0), xfirst=False):
        result = press(btns_xfirst, positions, i + 1, cache, debug)
    else:
        xfirst = press(
            btns_xfirst, list(positions), i + 1, ({} if debug else cache), False
        )
        yfirst = press(
            btns_yfirst, list(positions), i + 1, ({} if debug else cache), False
        )
        result = press(
            (btns_xfirst if xfirst < yfirst else btns_yfirst),
            positions,
            i + 1,
            cache,
            debug,
        )
    positions[i] = newpos
    if not debug:
        cache[cache_key] = (result, list(positions[i:]))
    return result


def press(
    buttons_with_counts: list[tuple[str, int]],
    positions: list[tuple[int, int]],
    i: int,
    cache: dict[str, int],
    debug: bool,
) -> int:
    return sum(
        press_one(positions, i, btn, cache, debug)
        for btn, n in buttons_with_counts
        for _ in range(n)
    )


def solve(code, n):
    result = press([(c, 1) for c in code], [(2, 3)] + [(2, 0)] * n, 0, {}, _DEBUG)
    if _DEBUG:
        print(code, result)
    return result * int(code[:-1])


codes = [line.removesuffix("\n") for line in fileinput.input()]
print(sum(solve(code, 2) for code in codes))
if not _DEBUG:
    print(sum(solve(code, 25) for code in codes))
