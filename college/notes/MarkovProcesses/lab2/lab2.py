#!/usr/bin/env python3.5

import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sympy as sp
import scipy
import scipy.stats as sc_st

sp.init_printing()

FIGSIZE = (8, 4)

def main():
    algo1(3, 4)
    algo2(9, lambda t: t**2 - 10 * t + 26)

    l, t = 3, 4
    T, N = HPP(l, t)
    plt.figure(figsize=FIGSIZE)
    plt.plot(np.arange(t + 2), np.zeros(t + 2))
    plt.plot(t * np.ones(3), np.arange(-1, 2))
    plt.plot(T, np.zeros(len(T)), 'k*', markersize=20)
    plt.title(r'HPP with $N={}$, $\lambda={}$, and $t={}$'.format(N, l, t))
    plt.savefig('HPP')


def HPP(l, t):
    T = [0]
    i = 0
    while True:
        u = random.random()
        i += 1
        T.append(T[i - 1] - math.log(u) / l)
        if T[i] > t:
            N = i - 1
            break
    return T, N


def algo1(l, t):
    num_generate = 100000

    vals = np.zeros(shape=(num_generate, 3))

    for j in range(num_generate):
        TN = 0
        i  = 0
        while True:
            TN1 = TN - (math.log(random.random()) / l)
            i += 1
            if TN1 > t:
                N = i - 1
                break
            else:
                TN = TN1
        vals[j] = [N, TN, TN1]


    plt.figure(figsize=FIGSIZE)
    hist, edges = np.histogram(vals[:, 0], bins='auto', density=True)
    plt.bar(edges[:-1], hist / 6, width=0.5, label='Histogram of Generated Values')
    x = np.arange(int(edges[-1] + 1))
    rv = sc_st.poisson(l * t)
    plt.plot(x, rv.pmf(x), label=r'$Pois(\lambda \cdot t)$')
    plt.xlabel('Density')
    plt.ylabel('Value')
    plt.title(r'Histogram of $N$ Values')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('part1_N_hist.png')

    plt.figure(figsize=FIGSIZE)
    hist, edges = np.histogram(vals[:, 2] - vals[:, 1], bins='auto', density=True)
    plt.bar(edges[:-1], hist, width=edges[1] - edges[0], label='Histogram of Generated Values')
    plt.xlabel('Density')
    plt.ylabel('Value')
    plt.title(r'Histogram of $T(N+1) - T(N)$ Values')
    plt.tight_layout()
    plt.legend(loc=0)
    plt.savefig('part1_TN1_TN_hist.png')

    plt.figure(figsize=FIGSIZE)
    hist, edges = np.histogram(vals[:, 2] - t, bins='auto', density=True)
    plt.bar(edges[:-1], hist, width=edges[1] - edges[0], label='Histogram of Generated Values')
    x = np.linspace(0, int(edges[-1] + 1), 1000)
    rv = lambda x: l * np.exp(-l * x)
    plt.plot(x, rv(x), label=r'$Exp(\lambda)$')
    plt.xlabel('Density')
    plt.ylabel('Value')
    plt.title(r'Histogram of $T(N+1) - t$ Values')
    plt.tight_layout()
    plt.legend(loc=0)
    plt.savefig('part1_TN1_t_hist.png')

    plt.figure(figsize=FIGSIZE)
    hist, edges = np.histogram(vals[:, 2], bins='auto', density=True)
    plt.bar(edges[:-1], hist, width=edges[1] - edges[0], label='Histogram of Generated Values')
    x = np.linspace(t, int(edges[-1] + 1), 1000)
    rv = lambda x: l * np.exp(-l * (x - t))
    plt.plot(x, rv(x), label=r'Shifted $Exp(\lambda)$')
    plt.xlabel('Density')
    plt.ylabel('Value')
    plt.title(r'Histogram of $T(N+1)$ Values')
    plt.tight_layout()
    plt.legend(loc=0)
    plt.savefig('part1_TN1_hist.png')


def algo2(T, l_func):
    x = np.linspace(0, T, 1000)
    plt.figure(figsize=FIGSIZE)
    plt.plot(x, l_func(x))
    plt.plot(x, max(l_func(x)) * np.ones(len(x)), 'k--', alpha=0.5)
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'$\lambda(t)$')
    plt.title(r'Plot of $\lambda(t)$ for $0 \leq t \leq T=9$')
    plt.tight_layout()
    plt.savefig('part2_lambda.png')

    C = max(l_func(x))
    Tvals, N = HPP(C, T)
    Tvals = np.array(Tvals)
    plt.figure(figsize=FIGSIZE)
    plt.plot(np.arange(T + 2), np.zeros(T + 2))
    plt.plot(T * np.ones(3), np.arange(-1, 2))
    plt.scatter(Tvals, np.zeros(len(Tvals)), marker='|', s=200)
    plt.title(r'HPP with $N={}$, $\lambda={}$, and $t={}$'.format(N, C, T))
    plt.xlim(0, T+1)
    plt.savefig('HPPC')


    N, count = NHPP(l_func, T, C)
    plt.figure(figsize=FIGSIZE)
    plt.plot(np.arange(T + 2), np.zeros(T + 2))
    plt.plot(T * np.ones(3), np.arange(-1, 2))
    plt.scatter(N, np.zeros(len(N)), marker='|', s=200)
    plt.title(r'NHPP with $N={}$, $\lambda(t)=t^2-10t+26$, and $t={}$'.format(count, C, T))
    plt.xlim(0, T+1)
    plt.savefig('NHPP')


    num = int(1e5)
    W = np.zeros(num)
    for i in range(num):
        _, w = NHPP(l_func, T, C)
        W[i] = w

    plt.figure(figsize=FIGSIZE)
    hist, edges = np.histogram(W, bins='auto', density=True)
    plt.bar(edges[:-1], hist / 2, width=edges[1] - edges[0], label='Histogram of Generated Values')
    rv = sc_st.poisson(72)
    x = np.arange(int(edges[-1]))
    plt.plot(x, rv.pmf(x), label=r'$Pois\left[ \lambda=\int_0^9 \lambda(t) \, dt \right]$')
    plt.xlabel('Density')
    plt.ylabel('Value')
    plt.title(r'Histogram of $W$ Values')
    plt.tight_layout()
    plt.legend(loc=0)
    plt.savefig('part2_W.png')


def NHPP(l_func, T, C):
    N = [0]
    i = 0
    t = 0
    while True:
        t = t - math.log(random.random()) / C
        if random.random() <= l_func(t) / C:
            N.append(t)
            i += 1
            if N[i] > T:
                count = i - 1
                break
    return N, count



if __name__ == '__main__':
    sys.exit(main())
