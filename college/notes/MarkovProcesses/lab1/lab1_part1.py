#!/usr/bin/env python3.5

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import itertools
import tqdm


# def permute(n):
#     u_1, ..., u_n = Unif(0, 1)
#     permuted = []
#     for i in 1:size
#         permuted.append(|{u_i <= u_j; j in [0, n]}|)
#     return permuted
def random_permutation(size: int) -> np.ndarray:
    random_unif = np.random.random(size=size)
    permuted_set = np.zeros(size)
    for i, x in enumerate(random_unif):
        permuted_set[i] = len(random_unif[x <= random_unif])
    return permuted_set

print(random_permutation(3))
#>> [ 3.  1.  2.]
print(random_permutation(5))
#>> [ 2.  1.  3.  5.  4.]
print(random_permutation(10))
#>> [  4.   8.  10.   1.   6.   2.   9.   7.   3.   5.]
print(random_permutation(20))
#>> [ 17.  11.   7.  14.  16.   2.   8.   4.  13.  18.   1.  15.   6.   3.   9.
#     10.  19.  12.  20.   5.]


n = 7
m = 6000
k = 2000
s = np.array([6, 7, 2, 5, 1, 4, 3])
p = 1 / math.factorial(n)

data = []
for i in tqdm.tqdm(range(k)):
    count = 0
    for j in range(m):
        pp = random_permutation(n)
        if (pp == s).all():
            count += 1
    data.append(count)
hist, bin_edges = np.histogram(data, bins=list(range(11)), density=True)
x = np.arange(0, 11, 1, dtype=int)
binomial = np.zeros(len(x))
for i, k in enumerate(x):
    binomial[i] = ((math.factorial(m) / (math.factorial(k) *
                    math.factorial(m - k))) * p**k * (1 - p)**(m - k))
# plotting
plt.figure()
plt.plot(x, binomial, label='Binomial({}, {:0.08})'.format(m, p))
plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0],
        alpha=0.5, label='Experimental Values')
plt.ylabel('Frequency')
plt.xlabel(r'Count of $\sigma$')
plt.legend(loc=0)
plt.savefig('distx.png')


n = 7
m = 6000
k = 2000
s = np.array([6, 7, 2, 5, 1, 4, 3])
p = 1 / math.factorial(n)

data = []
for i in tqdm.tqdm(range(k)):
    count = 0
    while True:
        count += 1
        pp = random_permutation(n)
        if (pp == s).all():
            break
    data.append(count)
hist, bin_edges = np.histogram(data, bins=20, density=True)
x = np.arange(0, int(1e5 / 2), 1, dtype=int)
geom = np.zeros(len(x))
for i, k in enumerate(x):
    geom[i] = (1 - p)**(k - 1) * p
# plotting
plt.figure()
plt.plot(x, geom, label='Geometric({:0.08})'.format(p))
plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0],
        alpha=0.5, label='Experimental Values')
plt.ylabel('Frequency')
plt.xlabel('Number of Trials')
plt.legend(loc=0)
plt.savefig('disty.png')
