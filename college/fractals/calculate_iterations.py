import math as np
def calculate_iterations(px, py, x0, y0, max_iterations):
    x = y = iteration = 0
    while (x**2 + y**2 < 4 and iteration < max_iterations):
        xtmp = x**2 - y**2 + x0
        y = 2 * x * y + y0
        x = xtmp
        iteration += 1
    if iteration < max_iterations:
        iteration += 1 - np.log(np.log(np.sqrt(x**2 + y**2)) /
                                        np.log(2) ) / np.log(2)
    return iteration
