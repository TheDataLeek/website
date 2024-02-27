#!/usr/bin/env python3.5

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc_st
from tqdm import tqdm
import seaborn
import concurrent.futures


def main():
    l = 1
    m = 2
    T = 50

    print('---')
    print('n & pi(n)\\')
    rho = l / m
    for i in range(10):
        print('{} & {}\\'.format(i, (1 - rho) * (rho**i)))
    print('---')

    size = 10000
    initial = sc_st.geom(1 - rho, loc=-1)
    nvals = initial.rvs(size=size)

    events, history = MM1_queue(l, m, nvals[0], T)
    filtered = [t for t, s, in events if s < 0]
    plt.figure(figsize=(12, 8))
    plt.plot(filtered, np.zeros(len(filtered)), 'k*')
    plt.savefig('poissonprocess.png')

    points = np.zeros(size)
    time_free = np.zeros(size)
    for i in tqdm(range(size)):
        events, history = MM1_queue(l, m, nvals[i], T)
        points[i] = history[-1]
        time_free[i] = frac_busy(events, history, T)

    hist, edges = np.histogram(points, bins=10, density=True)
    width = edges[1] - edges[0]
    rv = initial
    x = np.arange(0, edges[-1])
    plt.figure(figsize=(12, 8))
    plt.bar(edges[:-1], hist, width,
            label='Simulated Data',
            alpha=0.6)
    plt.plot(x, rv.pmf(x), label=r'$Geom(1 - \rho)$')
    plt.legend(loc=0)
    plt.xlabel('Total Number of Arrivals')
    plt.ylabel('Frequency')
    plt.title(r'Histogram of $X_T$ Values, $N={}$'.format(len(points)))
    plt.savefig('xt_dist.png')

    hist, edges = np.histogram(time_free, bins='auto', density=True)
    width = edges[1] - edges[0]
    plt.figure(figsize=(12, 8))
    plt.bar(edges[:-1], hist, width,
            label='Simulated Data',
            alpha=0.6)
    plt.legend(loc=0)
    plt.xlabel('Fraction of Time Free')
    plt.ylabel('Frequency')
    plt.title(r'Histogram of Time Free, $N={}$'.format(len(time_free)))
    plt.savefig('time_free.png')



def frac_busy(events, history, T):
    free = False
    time_free = 0
    for i, z in enumerate(zip(events, history)):
        event, status = z
        if status == 0:
            if free:
                time_free += event[0] - events[i - 1][0]
            free = True
        else:
            if free:
                time_free += event[0] - events[i - 1][0]
            free = False
    return time_free / T


def MM1_queue(arrival_rate, service_rate, initial_num, time_length):
    """
    arrival_rate => \lambda
    service_rate => \mu
    initial_num  => n
    time_length  => T

    https://en.wikipedia.org/wiki/M/M/1_queue
    """
    arrival = sc_st.expon(scale=1 / arrival_rate)
    service = sc_st.expon(scale=1 / service_rate)

    ctime = 0

    events = []
    history = [initial_num]
    while ctime < time_length:
        new_arrival = arrival.rvs()

        # Only let new services if there's something in the queue
        if history[-1] == 0:
            new_service = 1000
        else:
            new_service = service.rvs()

        if new_arrival < new_service:
            ctime += new_arrival
            events.append((ctime, 1))
            history.append(history[-1] + 1)
        else:
            ctime += new_service
            events.append((ctime, -1))
            history.append(history[-1] - 1)

    return events[:-1], history[:-1]



if __name__ == '__main__':
    sys.exit(main())
