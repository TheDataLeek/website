#!/usr/bin/env python3.5

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import itertools
import tqdm

def random_permutation(size: int) -> np.ndarray:
    random_unif = np.random.random(size=size)
    permuted_set = np.zeros(size, dtype=int)
    for i, x in enumerate(random_unif):
        permuted_set[i] = len(random_unif[x <= random_unif])
    return permuted_set

n_vals = [9, 21, 36, 69]
m = 100000

data = np.zeros(m)
for i, n in enumerate(n_vals):
    for j in range(m):
        data[j] = np.where(random_permutation(n) == 1)[0][0] + 1
    print(n, data.mean())

print('--------')
def methodB(p: np.ndarray, v: int):
    x0 = 0
    x1 = len(p)
    count = 0
    while True:
        if len(p[x0:x1]) == 1:
            break
        count += 1
        m = x0 + int((x1 - x0) / 2)
        left = p[x0:m]
        right = p[m:x1]
        if v in left:
            x1 = m
        else:
            x0 = m
    return count

# Method B Simulation
data = np.zeros(m)
for i, n in enumerate(n_vals):
    for j in range(m):
        data[j] = methodB(random_permutation(n), 1)
    print(n, data.mean())
