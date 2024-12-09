#!/usr/bin/env python
import fileinput
import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class BlockSeq:
    pos: int
    len: int
    id: int


def defrag(blocks: List[BlockSeq]):
    gapIdx = 0
    while True:
        while gapIdx + 1 < len(blocks):
            if blocks[gapIdx].pos + blocks[gapIdx].len < blocks[gapIdx + 1].pos:
                break
            gapIdx += 1
        if gapIdx + 1 == len(blocks):
            return
        gapBegin = blocks[gapIdx].pos + blocks[gapIdx].len
        gapEnd = blocks[gapIdx + 1].pos
        gapLen = gapEnd - gapBegin
        if blocks[-1].len <= gapLen:
            blocks.insert(gapIdx + 1, BlockSeq(gapBegin, blocks[-1].len, blocks[-1].id))
            blocks.pop()
        else:
            blocks.insert(gapIdx + 1, BlockSeq(gapBegin, gapLen, blocks[-1].id))
            blocks[-1] = BlockSeq(
                blocks[-1].pos, blocks[-1].len - gapLen, blocks[-1].id
            )


def checksum(blocks):
    return sum(
        block.id * (block.pos + (block.pos + block.len - 1)) * block.len // 2
        for block in blocks
    )


def solve(blocks: List[BlockSeq]):
    defrag(blocks)
    return checksum(blocks)


def defrag2(blocks: List[BlockSeq]):
    i = len(blocks) - 1
    while True:
        block = blocks[i]
        moved = False
        for j in range(0, i):
            gapBegin = blocks[j].pos + blocks[j].len
            gapEnd = blocks[j + 1].pos
            if block.len > gapEnd - gapBegin:
                continue
            del blocks[i]
            blocks.insert(j + 1, BlockSeq(gapBegin, block.len, block.id))
            moved = True
            break
        if i == 0:
            break
        if not moved:
            i -= 1


def solve2(blocks: List[BlockSeq]):
    defrag2(blocks)
    return checksum(blocks)


m = [map(int, line.removesuffix("\n")) for line in fileinput.input()][0]

nextId = 0
pos = 0
blocks = list()
for i, c in enumerate(m):
    if i % 2 == 0:
        blocks.append(BlockSeq(pos, c, nextId))
        nextId += 1
    pos += c

print(solve(list(blocks)))
print(solve2(blocks))
