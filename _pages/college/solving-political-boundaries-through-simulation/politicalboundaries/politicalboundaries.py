#!/usr/bin/env python3.6

"""
Simulated Annealing and Genetic Algorithm solutions to districting problem.

Notes on implementation:
    * I like properties and I use them in a couple spots...
    * Use `pip install --user -r requirements.txt` on the requirements file
    available in the root of this git repository.
    * If by some strange happenstance you only have this file, go to the
    following url to get the entire repo.
    https://gitlab.com/thedataleek/politicalboundaries
    * Test coverage is around 80% which I'm happy with. All the super important
    things are tested.

TODO: Multithread
"""

# stdlib imports
import sys       # exits and calls
import os        # path manipulation
import argparse  # argument handling
import math      # exponent - faster than numpy for this
import random    # built-in random func
import re        # dynamically pull out algo names
import queue     # used for SCC labelling

# typing
from typing import Union

# third-party imports
import numpy as np                       # heavy lifting
import matplotlib                        # visualization
matplotlib.use('agg')                    # switch backends for compatibility
import matplotlib.pyplot as plt          # visualization
import matplotlib.animation as animation # animation
from moviepy import editor               # More gif
from tqdm import tqdm                    # Progress bars are nice
from halo import Halo                    # Spinner


FIGSIZE = (4, 4)  # For asset exporting
OUTDIR = './img'
INITIAL_COLORMAP = plt.get_cmap('bwr')
FILL_COLORMAP = plt.get_cmap('nipy_spectral')
DISTRICT_COLORMAP = plt.get_cmap('tab10')
STICKY_NUM = 100
TARGET_VALUE = 35


def main():
    args = get_args()
    system = System(args.filename)
    if args.numdistricts is None:
        args.numdistricts = len(system.matrix[0])
    if args.full:
        generate_report_assets(system, args.numdistricts, args.precision, True)
        simulated_annealing(system, args.numdistricts, args.precision, True, True)
        genetic_algorithm(system, args.numdistricts, args.precision, True, True)
    elif args.report:
        generate_report_assets(system, args.numdistricts, args.precision, args.gif)
    elif args.annealing:
        simulated_annealing(system, args.numdistricts, args.precision,
                            args.animate, args.gif)
    elif args.genetic:
        genetic_algorithm(system, args.numdistricts, args.precision,
                          args.animate, args.gif)
    else:
        print('Running in Demo Mode!!!')
        print('First we\'ll use Simulated Annealing')
        simulated_annealing(system, args.numdistricts, args.precision,
                            False, False)
        print('Now we\'ll try the Genetic Algorithm')
        genetic_algorithm(system, args.numdistricts, args.precision,
                          False, False)


def simulated_annealing(system, numdistricts, precision, animate, makegif):
    """
    Perform simulated annealing on our system with a series of progressively
    improving solutions.
    """
    solution = get_good_start(system, numdistricts)
    history = [solution]  # Keep track of our history
    k = 0.1  # Larger k => more chance of randomly accepting
    Tvals = np.linspace(1, 1e-15, precision,
                        dtype=np.float128)
    cval = solution.value
    iterations_since_increase = 0
    print(f'Running Simulated Annealing with k={k:0.03f}, alpha={1.0 / precision:0.05f}')
    print(f'num_iterations={len(Tvals)}')
    for i, T in tqdm(enumerate(Tvals), total=len(Tvals)):
        new_solution = solution.copy()  # copy our current solution
        new_solution.mutate()  # Mutate the copy
        dv = new_solution.value - cval  # Look at delta of values
        # If it's better, or random chance, we accept it
        if dv > 0 or random.random() < math.exp(dv / (k * T)):
            solution = new_solution
            cval = solution.value
            history.append(new_solution)
            if dv > 0:
                iterations_since_increase = 0
        else:
            iterations_since_increase += 1
        if ((iterations_since_increase > STICKY_NUM) and
                (cval >= TARGET_VALUE)):
            print('Hit a ceiling, aborting algorithm.')
            break

    solution.count = len(Tvals)
    solution.algo = 'Simulated Annealing'
    print(solution)
    print(solution.summary())

    plt.figure(figsize=FIGSIZE)
    plt.plot(np.arange(len(history)),
             [s.value for s in history])
    plt.title('Simulated Annealing Convergence')
    plt.xlabel('Iteration Count')
    plt.ylabel('Value')
    plt.savefig(os.path.join(OUTDIR, 'simulated_annealing_values.png'))

    if animate:
        animate_history(system.filename, system.matrix,
                        history, solution.numdistricts,
                        makegif)


def get_good_start(system, numdistricts):
    """
    Basically, instead of starting with a really bad initial solution for
    simulated annealing sometimes we can rig it to start with a decent one...
    """
    print('Acquiring a good initial solution')
    solution = Solution(system, numdistricts)
    solution.generate_random_solution()  # start with random solution
    for i in tqdm(range(500)):
        new_solution = Solution(system, numdistricts)
        new_solution.generate_random_solution()
        if new_solution.value > solution.value:
            solution = new_solution
    print(f'Starting with Solution[{solution.value}]')
    return solution


def genetic_algorithm(system, numdistricts, precision, animate, makegif):
    """
    Use a genetic algorithm to find a good solution to our district problem
    """
    def get_top_3(solutions):
        solutions.sort(key=lambda s: -s.value)
        return solutions[:3]
    # Start with random initial solution space (3)
    solutions = [Solution(system, numdistricts) for _ in range(100)]
    for s in solutions:
        s.generate_random_solution()  # Initialize our solutions
    solutions = get_top_3(solutions)
    top_history = []  # Keep history of our top solution from each "frame"
    iterations_since_increase = 0
    value_history = []
    iteration_num = 0
    with Halo(text='Running Algorithm', spinner='dots') as spinner:
        while True:
            new_solutions = []
            for _ in range(10):  # Create 10 children per frame
                s1, s2 = np.random.choice(solutions, size=2)
                # Randomly combine two parents
                combined = s1.combine(s2)
                # Mutate as well
                combined.mutate()
                new_solutions.append(combined)
            # Combine everything, giving 13 total solutions
            full_solutions = new_solutions + solutions
            # Keep the top 3 for next generation
            solutions = sorted([(s, s.value) for s in full_solutions],
                                key=lambda tup: -tup[1])
            value_history += [(iteration_num, s[1]) for s in solutions]
            solutions = [_[0] for _ in solutions[:3]]
            # Only record top from generation, and only if it's changed
            if len(top_history) == 0 or solutions[0] != top_history[-1]:
                top_history.append(solutions[0])
                spinner.text = f'Current Generation Top Solution: {str(solutions[0].value)}'
                iterations_since_increase = 0
            else:
                iterations_since_increase += 1
            if ((iterations_since_increase > STICKY_NUM) and
                    (top_history[-1].value >= TARGET_VALUE)):
                print('Hit a ceiling, aborting algorithm.')
                break
            iteration_num += 1

    solution = top_history[-1]
    solution.count = precision
    solution.algo = 'Genetic Algorithm'
    print(solution)
    print(solution.summary())

    value_history = np.array(value_history)
    plt.figure(figsize=FIGSIZE)
    plt.scatter(value_history[:, 0], value_history[:, 1], alpha=0.2, s=10)
    plt.title('Genetic Algorithm Convergence')
    plt.xlabel('Iteration Count')
    plt.ylabel('Value')
    plt.savefig(os.path.join(OUTDIR, 'genetic_algorithm_values.png'))

    if animate:
        animate_history(system.filename, system.matrix,
                        top_history, solution.numdistricts,
                        makegif)


def generate_report_assets(system, numdistricts, precision, makegif):
    """
    Responsible for generating all plots and animations specific to the writeup.
    In order this includes the following.

    1. Basic initial voting areas
    2. Random solution progression
    3. Mutation demonstration
    4. Genetic algorithm combination demonstration
    """
    # First just plot initial map
    plt.figure(figsize=FIGSIZE)
    plt.imshow(system.matrix, interpolation='nearest',
               cmap=INITIAL_COLORMAP)
    plt.axis('off')
    plt.title(system.filename)
    plt.savefig(os.path.join(OUTDIR, system.filename.split('.')[0] + '_initial.png'))

    # Now generate random solution
    solution = Solution(system, numdistricts)
    solution_history = solution.generate_random_solution(history=True)
    animate_history(system.filename, system.matrix,
                    solution_history, solution.numdistricts, makegif,
                    algo_name='generate_random',
                    cmap=FILL_COLORMAP)

    # Now show mutation
    backup = solution.copy()
    fig, axarr = plt.subplots(1, 3, figsize=FIGSIZE)
    axarr[0].imshow(solution.full_mask, interpolation='nearest',
                    cmap=DISTRICT_COLORMAP)
    axarr[0].axis('off')
    axarr[0].set_title('Initial')
    solution.mutate()
    axarr[1].imshow(solution.full_mask, interpolation='nearest',
                    cmap=DISTRICT_COLORMAP)
    axarr[1].axis('off')
    axarr[1].set_title('Mutant')
    axarr[2].imshow(np.abs(backup.full_mask - solution.full_mask),
                    interpolation='nearest',
                    cmap=FILL_COLORMAP)
    axarr[2].axis('off')
    axarr[2].set_title('Difference')
    plt.savefig(os.path.join(OUTDIR, 'mutation.png'))

    # Now show combination
    solution.full_mask[:] = 0
    solution.generate_random_solution()
    fig, axarr = plt.subplots(2, 2, figsize=FIGSIZE)
    axarr[0, 0].imshow(backup.full_mask, interpolation='nearest',
                       cmap=FILL_COLORMAP,
                       vmin=0,
                       vmax=solution.numdistricts)
    axarr[0, 0].axis('off')
    axarr[0, 0].set_title('Parent 1')
    axarr[0, 1].imshow(solution.full_mask, interpolation='nearest',
                       cmap=FILL_COLORMAP,
                       vmin=0,
                       vmax=solution.numdistricts)
    axarr[0, 1].axis('off')
    axarr[0, 1].set_title('Parent 2')

    child, history = backup.combine(solution, keep_history=True)
    axarr[1, 1].imshow(child.full_mask, interpolation='nearest',
                       cmap=FILL_COLORMAP,
                       vmin=0,
                       vmax=solution.numdistricts)
    axarr[1, 1].axis('off')
    axarr[1, 1].set_title('Child')

    sol = axarr[1, 0].imshow(history[0].full_mask, interpolation='nearest',
                             cmap=FILL_COLORMAP,
                             vmin=0,
                             vmax=child.numdistricts)
    axarr[1, 0].axis('off')
    axarr[1, 0].set_title('Step by Step')

    def update(i):
        sol.set_data(history[i].full_mask)
        return sol,

    ani = animation.FuncAnimation(fig, update, len(history),
                                  interval=500, blit=True)
    filename = 'combine'
    ani.save(os.path.join(OUTDIR, filename + '.mp4'))
    editor.VideoFileClip(os.path.join(OUTDIR, filename + '.mp4'))\
            .write_gif(os.path.join(OUTDIR, filename + '.gif'))

    # Now show the difference in k for simulated annealing
    plt.figure(figsize=(6, 6))
    Tvals = np.arange(1, 1e-12, -1.0 / precision)
    dv = -1
    determine_k = lambda T, k: np.exp(dv / (k * T))
    for k in np.linspace(0.01, 1, 100):
        plt.plot(Tvals[::-1], determine_k(Tvals, k))
    plt.xlabel('Algorithm Iteration')
    plt.ylabel('Chance of Accepting')
    plt.title(r'Effect of Differing $k$')
    plt.savefig(os.path.join(OUTDIR, 'kvals.png'))


def animate_history(filename, systemdata, history, numdistricts, makegif, algo_name=None, cmap=None):
    """
    Take our given solution history, and animate it using matplotlib.animate.
    Save to gif if asked.
    """
    print('Saving Animation')
    fig, axarr = plt.subplots(1, 2, figsize=FIGSIZE)
    # Plot our "field"
    systemplot = axarr[0].imshow(systemdata, interpolation='nearest',
                                 cmap=INITIAL_COLORMAP)
    axarr[0].axis('off')
    # Plot our first solution
    sol = axarr[1].imshow(history[0].full_mask, interpolation='nearest',
                          cmap=cmap or DISTRICT_COLORMAP,
                          vmin=0,
                          vmax=numdistricts)
    axarr[1].set_title(f'value {history[0].value:0.03f}')
    axarr[1].axis('off')

    def update_plot(i, N):
        """Animation loop"""
        sol.set_data(history[i].full_mask)
        axarr[1].set_title(f'value {history[i].value:0.03f}')
        plt.suptitle(f'Solution {i}')
        return sol,

    interval = 100  # milliseconds
    ani = animation.FuncAnimation(
        fig,
        update_plot,
        len(history),
        fargs=(len(history) - 1,),
        interval=interval,
        blit=True
    )
    if not algo_name:
        algo_name = re.sub(' ', '_', history[-1].algo.lower())
    filename = f'{algo_name}_solution_{filename.split(".")[0]}'
    ani.save(os.path.join(OUTDIR, filename + '.mp4'))

    if makegif:
        editor.VideoFileClip(os.path.join(OUTDIR, filename + '.mp4'))\
                .write_gif(os.path.join(OUTDIR, filename + '.gif'))

    # Save final solution as separate image
    if history[-1].algo is not None:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(history[-1].full_mask, interpolation='nearest',
                   cmap=DISTRICT_COLORMAP,
                   vmin=0,
                   vmax=numdistricts)
        plt.title(history[-1].algo + ' Final Solution')
        plt.axis('off')
        plt.savefig(os.path.join(OUTDIR, filename + '.png'))


class Solution(object):
    """This is our unique solution class"""
    def __init__(self, system, numdistricts):
        self.system = system
        self.numdistricts = numdistricts
        if numdistricts is None:  # If user doesn't specify
            self.numdistricts = system.width
        # Our solution is simply a numpy array
        self.full_mask = np.zeros((system.height, system.width))
        self.algo = None
        self.count = 0

    def __getitem__(self, key):
        """Allows us to easily index each district and get a Mask back"""
        if key < 1 or key > self.numdistricts:
            raise KeyError('District does not exist!')
        else:
            new_mask = Mask()  # initialize new empty mask
            # Set mask from district
            new_mask.parse_list(self.get_solution(key))
            return new_mask

    def __str__(self):
        """String version is just the string version of numpy array"""
        return str(self.full_mask)

    def __eq__(self, other):
        return (self.full_mask == other.full_mask).all()

    def __ne__(self, other):
        return not (self == other)

    def summary(self):
        """This is literally only here for the grading..."""
        sep = (40 * '-') + '\n'
        summary_string = ''
        summary_string += sep
        summary_string += f'Score: {self.value}\n'
        summary_string += sep
        total_size, percents = self.system.stats
        summary_string += f'Total Population Size: {total_size}\n'
        summary_string += sep
        summary_string += 'Party Division in Population\n'
        for k, v in percents.items():
            summary_string += f'{k}: {v:05f}\n'
        summary_string += sep

        majorities = {k:0 for k in self.system.names.keys()}
        locations = []
        for i in range(1, self.numdistricts + 1):
            majorities[self.system._name_arr[self.majority(i)]] += 1
            locations.append(self[i].location)
        summary_string += 'Number of Districts with Majority by Party\n'
        for k, v in majorities.items():
            summary_string += f'{k}: {v}\n'
        summary_string += sep

        summary_string += 'District Locations (zero-indexed, [y, x])\n'
        for i, loc in enumerate(locations):
            loc_string = ','.join(str(tup) for tup in loc)
            summary_string += f'District {i + 1}:{loc_string}\n'
        summary_string += sep

        summary_string += f'Algorithm: {self.algo}\n'
        summary_string += sep

        summary_string += f'Valid Solution States Explored: {self.count}\n'
        summary_string += sep

        return summary_string[:-1]

    def majority(self, i):
        """
        Tell us who has majority in the specified district
        """
        district = self.system.matrix[self[i].mask.astype(bool)]
        if district.sum() > (len(district) / 2.0):
            return 1
        else:
            return 0

    def copy(self):
        """
        So... Numpy uses memory instances of arrays, meaning you need to tell it
        to actually copy the damn thing otherwise messing with the first will
        mess with all of its successors

        This was a bad bug...
        """
        new_sol = Solution(self.system, self.numdistricts)
        new_sol.full_mask = np.copy(self.full_mask)
        return new_sol

    def show(self, save=False, name='out.png'):
        """Debug function for individual plotting. Deprecated."""
        fig, axarr = plt.subplots(1, 2, figsize=FIGSIZE)
        axarr[0].imshow(self.system.matrix, interpolation='nearest')
        axarr[1].imshow(self.full_mask, interpolation='nearest')
        axarr[1].set_title(f'Value: {self.value}')
        axarr[0].axis('off')
        axarr[1].axis('off')
        if save:
            plt.savefig(os.path.join(OUTDIR, name))
        else:
            plt.show()

    @property
    def is_valid(self):
        """
        A valid solution is one that covers everything. So we do two things
        here, first of which is to make sure that no element in the mask is
        zero, and second check that each district is valid.
        """
        if (self.full_mask == 0).any():
            return False
        for i in range(1, self.numdistricts + 1):
            if not self[i].is_valid:
                return False
        return True

    @property
    def majorities(self):
        """
        Tell us the number of districts with majority in each party
        """
        majorities = {k:0 for k in self.system.names.keys()}
        for i in range(1, self.numdistricts + 1):
            majorities[self.system._name_arr[self.majority(i)]] += 1
        return majorities

    @property
    def value(self):
        """
        This is our fitness function.

        Here's what we're doing here
        1. Make sure we have valid solution
        2. Make sure that the population distribution matches the district
        distribution within 10%
        3. The value of a solution is just the sum of our district solutions
        4. Each district has value equal to the absolute value difference
        between party population sizes. For instance, a district with [R, D, D]
        has value 2.
        5. We also look for the optimal district size which is just
        (width*height/numdistricts), and subtract 1 for every point off we are
        from "optimal"
        6. Lastly we say that independent voters are a fixed effect in "rogue"
        districts, so every district with a rogue voter counts as 0.1 towards
        the total. This can be seen as just the sum for the following district
                [R, D, D]
            sum([-0.9, 1, 1]) = 2.1
        """
        value = 0
        if not self.is_valid:  # if we don't have a valid solution, return 0
            return 0
        # Make sure the number of districts tries to match population
        # distribution within 10%
        size, stats = self.system.stats
        for k, v in self.majorities.items():
            if np.abs((float(v) / self.numdistricts) - stats[k]) >= 0.1:
                return 0
        district_size = int(self.width * self.height / self.numdistricts)
        # Sum up values of each district
        for i in range(1, self.numdistricts + 1):
            values = self.system.matrix[self[i].mask.astype(bool)]
            if len(values) == 0:
                value = 0
                return value
            else:
                # District value is simply abs(num_red - num_blue)
                subvalue = np.abs(len(values[values == 0]) - len(values[values == 1]))
                size_bonus = 0.25 * np.abs(len(values) - district_size)
                if subvalue < len(values):
                    # For any non-uniform values, add 10% their value to account
                    # for independent voter turnout
                    subvalue += (len(values) - subvalue) * 0.1
                value += subvalue
                value -= size_bonus
                # Minimize neighbors (same as minimizing edge length)
                value += -0.1 * len(self.get_district_neighbors(i))
        return value

    def get_solution(self, i):
        """
        Return array just showing district

        If our full_mask looks like this
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        This function returns the following when i=2
            [[0, 0, 1],
             [0, 1, 0],
             [1, 1, 0]]
        """
        return (self.full_mask == i).astype(int)

    def get_random_openspot(self, value):
        """
        Return a random location where our full mask is equal to value

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_random_openspot(1) could return any of
            [[0, 0], [0, 1], [1, 0]]
        """
        openspots = np.where(self.full_mask == value)
        if len(openspots[0]) == 1:
            choice = 0
        elif len(openspots[0]) == 0:
            return None, None  # if no spots exist, return None
        else:
            choice = np.random.randint(0, len(openspots[0]) - 1)
        y = openspots[0][choice]
        x = openspots[1][choice]
        return y, x

    def get_full_openspots(self, value):
        """
        Instead of just returning one random openspot, return all of them.

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_full_openspots(1) will return (not necessarily sorted)
            [[0, 0], [0, 1], [1, 0]]
        """
        openspots = np.where(self.full_mask == value)
        spots = []
        for i in range(len(openspots[0])):
            spots.append((openspots[0][i], openspots[1][i]))
        return spots

    def get_neighbors(self, y, x):
        """
        Get all neighbors of a point that fall within boundary

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_neighbors(0, 1) will return (not necessarily sorted)
            [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]]
        """
        neighbors = [(y + yi, x + xi)
                     for xi in range(-1, 2)
                     for yi in range(-1, 2)
                     if (0 <= y + yi < self.system.height) and
                     (0 <= x + xi < self.system.width) and
                     not (xi == 0 and yi == 0)]
        return neighbors

    def get_district_neighbors(self, i):
        """
        Get all points on the edge of a district

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_district_neighbors(1) will return (not necessarily sorted)
            [[2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
        """
        y, x = self.get_random_openspot(i)
        q = queue.Queue()
        q.put((y, x))
        edges = []
        labels = np.zeros(self.full_mask.shape)
        labels[y, x] = 1
        while not q.empty():
            y, x = q.get()
            if self.full_mask[y, x] == i:
                for yi, xi in self.get_neighbors(y, x):
                    if labels[yi, xi] == 0:
                        q.put((yi, xi))
                        labels[yi, xi] = 1
            else:
                edges.append((y, x))
        return edges

    def get_filtered_district_neighbors(self, i, filter_list):
        """
        Simply a handy filter on get_district_neighbors. Only includes values
        that fall into the filter list

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_filtered_district_neighbors(1, [2]) will return (not necessarily
        sorted)
            [[2, 0], [2, 1], [1, 1], [0, 2]]
        """
        return [(y, x) for y, x in self.get_district_neighbors(i)
                if self.full_mask[y, x] in filter_list]

    def fill(self, keep_history=False):
        districts = list(range(1, self.numdistricts + 1))
        history = []
        while (self.full_mask == 0).any():
            try:
                i = districts[random.randint(0, len(districts) - 1)]
            except ValueError:
                # So here's a neat bug... Sometimes if there's a zero in the
                # corner, get filtered won't find it. So this code is here to
                # forcibly fix this problem.
                for j in range(1, self.numdistricts):
                    if len(self.get_filtered_district_neighbors(j, [0])) != 0:
                        districts = [j]
                        i = j
                        break
            neighbors = self.get_filtered_district_neighbors(i, [0])
            if len(neighbors) == 0:
                districts.remove(i)
            else:
                y, x = neighbors[random.randint(0, len(neighbors) - 1)]
                self.full_mask[y, x] = i
                if keep_history:
                    history.append(self.copy())
        return history

    def generate_random_solution(self, history=False):
        """
        Generate a random solution by picking spawn points and filling around
        them.

        Solutions are not guaranteed to be equal in size, as if one gets boxed
        off it will stay small...
        """
        solution_history = [self.copy()]
        for i in range(1, self.numdistricts + 1):
            y, x = self.get_random_openspot(0)
            self.full_mask[y, x] = i
            if history:
                solution_history.append(self.copy())
        solution_history += self.fill(keep_history=history)
        if history:
            return solution_history

    def mutate(self):
        """
        Pick a random district, find a random neighbor, and if the other
        district is at least size 2, replace the point with our district
        """
        i = np.random.randint(1, self.numdistricts)
        y, x = self.get_random_openspot(i)
        if y is None:
            raise IndexError('No open spots? Something is real bad')
        traversed = set()
        q = queue.Queue()
        q.put((y, x))
        while not q.empty():
            y, x = q.get()
            if (y, x) not in traversed:
                traversed.add((y, x))

                if (self.full_mask[y, x] != i and
                        self[self.full_mask[y, x]].size > 1):
                    old_value = self.full_mask[y, x]
                    self.full_mask[y, x] = i
                    if not self.is_valid:  # make sure new mutation is valid
                        # If not, reset and start over
                        self.full_mask[y, x] = old_value
                    else:
                        break

                for ii, jj in self.get_neighbors(y, x):
                    q.put((ii, jj))

    def combine(self, other_solution, keep_history=False):
        """
        Look at both solutions, alternate between them randomly, and try to
        basically inject one side at a time. Afterwards fill the gaps in with
        fill()
        """
        new_solution = Solution(self.system, self.numdistricts)
        # Randomly order parents to choose from
        pick_order = [self, other_solution]
        random.shuffle(pick_order)
        # Randomly order districts to choose from
        districts = list(range(1, self.numdistricts + 1))
        random.shuffle(districts)
        cursor = 0  # alternates between parents
        history = [new_solution.copy()]
        # place districts
        for i in districts:
            parent_locations = pick_order[cursor][i].location
            open_locations = new_solution.get_full_openspots(0)
            district = Mask()
            # We make every child valid
            district.parse_locations(self.height, self.width,
                                     [(y, x) for y, x in parent_locations
                                      if ((y, x) in open_locations)])
            district.make_valid()
            for y, x in district.location:
                new_solution.full_mask[y, x] = i
            cursor ^= 1
            if keep_history:
                history.append(new_solution.copy())
        # Fill
        for i in range(1, self.numdistricts + 1):
            y, x = new_solution.get_random_openspot(i)
            if y is None:
                y, x = new_solution.get_random_openspot(0)
                new_solution.full_mask[y, x] = i
                if keep_history:
                    history.append(new_solution.copy())
        history += new_solution.fill(keep_history=True)
        if random.random() < 0.1:
            new_solution.mutate()
            history.append(new_solution.copy())
        if keep_history:
            return new_solution, history
        return new_solution

    @property
    def height(self):
        return self.full_mask.shape[0]

    @property
    def width(self):
        return self.full_mask.shape[1]


class System(object):
    """
    Solely for reading in the file and keeping track of where things are
    """
    def __init__(self, filename):
        self.filename = filename
        self.matrix = None
        self.names = dict()
        self.num_names = 0
        self._read_file()

    def __getitem__(self, key):
        """
        Again, lets us access with self[i], and just return every index where
        our matrix is equal to 'D' or 'R'
        """
        if key not in list(self.names.keys()):
            raise KeyError(f'{key} does not exist')
        raw_spots = np.where(self.matrix == self.names[key])
        spots = []
        for i in range(len(raw_spots[0])):
            spots.append([raw_spots[0][i], raw_spots[1][i]])
        return spots

    @property
    def width(self):
        """Just the width of the system"""
        return self.matrix.shape[1]

    @property
    def height(self):
        """Just the height of the system"""
        return self.matrix.shape[0]

    @property
    def _name_arr(self):
        """Internal use, in order list of names ['D', 'R'] probably"""
        return [_[0] for _ in
                sorted(self.names.items(),
                       key=lambda tup: tup[1])]

    @property
    def stats(self):
        """For grading, returns size of system, percent of each party"""
        size = self.width * self.height
        percents = {}
        for k in self.names.keys():
            percents[k] = len(self[k]) / float(size)
        return size, percents

    def _read_file(self):
        """
        We read in the file here. The input file needs to be of a very specific
        format, where there are m rows and n columns, with fields separated by a
        space.

        D R D R D R R R
        D D R D R R R R
        D D D R R R R R
        D D R R R R D R
        R R D D D R R R
        R D D D D D R R
        R R R D D D D D
        D D D D D D R D
        """
        width = 0
        height = 0
        system = []
        with open(self.filename, 'r') as fileobj:
            i = 0
            for line in [re.sub('\n', '', _) for _ in fileobj.readlines()]:
                items = line.split(' ')
                system.append(items)
                width = len(items)
                i += 1
            height = i
        self.matrix = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                try:
                    num = self.names[system[i][j]]
                except KeyError:
                    self.names[system[i][j]] = self.num_names
                    self.num_names += 1
                self.matrix[i, j] = self.names[system[i][j]]

    def empty_state(self):
        """Return an empty version of the system. Deprecated."""
        return np.zeros(self.matrix.shape)


class Mask(object):
    """
    This is the class that tracks each solution

    Solutions are easy, as they're in the form of a bitmask
    """
    def __init__(self, height=0, width=0):
        self.mask = np.zeros((height, width))
        self.width, self.height = width, height

    def __str__(self):
        """Numpy string version of array"""
        return str(self.mask)

    def __eq__(self, other: Union['Mask', np.ndarray]):
        """Tells us if two masks are the same. Used in test code"""
        if isinstance(other, Mask):
            return np.array_equal(self.mask, other.mask)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.mask, other)
        else:
            raise ValueError('Invalid Types Supplied')

    @property
    def size(self):
        """Number of elements in mask"""
        return self.mask.sum()

    def parse_list(self, listvals):
        """given some entry list, set our mask to be those vals"""
        self.mask = np.array(listvals)
        self.height, self.width = self.mask.shape

    def parse_locations(self, height, width, locations):
        self.mask = np.zeros((height, width))
        self.height = height
        self.width = width
        for y, x in locations:
            self.mask[y, x] = 1

    def make_valid(self):
        """
        Makes the mask valid, remains the same if already valid

        Keeps a random connected component
        """
        if not self.is_valid:
            curlab, labels = self.get_labels()
            num_components = labels.max()
            keep = random.randint(1, num_components)
            spots = np.where(labels != keep)
            for i in range(len(spots[0])):
                y, x = spots[0][i], spots[1][i]
                self.mask[y, x] = 0
            assert self.is_valid  # CAUSE IM SCRED

    @property
    def location(self):
        """
        List of locations where mask == 1, returns (y, x) pairs
        """
        raw_spots = np.where(self.mask == 1)
        spots = []
        for i in range(len(raw_spots[0])):
            spots.append((raw_spots[0][i], raw_spots[1][i]))
        return spots

    def get_labels(self):
        """
        Valid masks have a single connected component.

        https://en.wikipedia.org/wiki/Connected-component_labeling

        This is what inspired much of the other code, this pattern is repeated
        throughout the code.
        """
        curlab = 1
        labels = np.zeros(self.mask.shape)
        q = queue.Queue()
        def unlabelled(i, j):
            return ((self.mask[i, j] == 1) and (labels[i, j] == 0))
        for i in range(self.height):
            for j in range(self.width):
                if unlabelled(i, j):
                    labels[i, j] = curlab
                    q.put((i, j))
                    while not q.empty():
                        y0, x0 = q.get()
                        neighbors = [(y0 + y, x0 + x)
                                     for x in range(-1, 2)
                                     for y in range(-1, 2)
                                     if (0 <= y0 + y < self.height) and
                                     (0 <= x0 + x < self.width) and
                                     not (x == 0 and y == 0)]
                        for ii, jj in neighbors:
                            if unlabelled(ii, jj):
                                labels[ii, jj] = curlab
                                q.put((ii, jj))
                    curlab += 1
        return curlab, labels

    @property
    def is_valid(self):
        curlab, _ = self.get_labels()
        if curlab > 2:
            return False
        else:
            return True


def get_args():
    """Get our arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='F', type=str, nargs=1,
                        help='File to load')
    parser.add_argument('-a', '--annealing', action='store_true',
                        default=False,
                        help='Use Simulated Annealing Algorithm?')
    parser.add_argument('-g', '--genetic', action='store_true',
                        default=False,
                        help='Use Genetic Algorithm?')
    parser.add_argument('-n', '--numdistricts', type=int, default=None,
                        help=('Number of districts to form. Defaults to the '
                              'width of the system'))
    parser.add_argument('-z', '--animate', action='store_true', default=False,
                        help='Animate algorithms?')
    parser.add_argument('-p', '--precision', type=int, default=10000,
                        help=('Tweak precision, lower is less. '
                              'In a nutshell, how many loops to run.'))
    parser.add_argument('-r', '--report', action='store_true', default=False,
                        help='Generate all assets for the report')
    parser.add_argument('-j', '--gif', action='store_true', default=False,
                        help='Generate gif versions of animations?')
    parser.add_argument('-F', '--full', action='store_true', default=False,
                        help='Generate everything. Report assets, SA, and GA.')
    args = parser.parse_args()
    args.filename = args.filename[0]  # We only allow 1 file at a time.
    return args


if __name__ == '__main__':
    sys.exit(main())
