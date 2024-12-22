#!/usr/bin/env python
import fileinput
import numpy as np
import itertools


def mixprune(a, b):
    return (a ^ b) % 16777216


a = np.loadtxt(fileinput.input(), np.int64).T

b = np.zeros((2001, len(a)), dtype=np.int64)
for i in range(2000):
    b[i] = a % 10
    a = mixprune(a, a * 64)
    a = mixprune(a, a // 32)
    a = mixprune(a, a * 2048)
b[-1] = a % 10
c = b[1:] - b[:2000]

allseqs = set()
seq2score = [{} for _ in range(len(a))]
for i, j in itertools.product(range(2000 - 4), range(len(a))):
    seq = tuple(map(int, c[i : i + 4, j]))
    allseqs.add(seq)
    seq2score[j].setdefault(seq, int(b[i + 4, j]))

print(np.sum(a))
print(max(sum(seq2score[j].get(seq, 0) for j in range(len(a))) for seq in allseqs))
