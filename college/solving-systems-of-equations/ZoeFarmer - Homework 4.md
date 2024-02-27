```
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from numpy.linalg import inv
from numpy.linalg import cond

%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib


**1. Write a program that takes as input a $3 \times 3$ matrix $\mathbf{A}$ and a three-element vector $\vec{b}$, uses Gaussian elimination without pivoting to find the solution $\vec{x}$ to the matrix equation, and prints the augmented matrix at every step.**


```
def printaugmented(a, b, n):
    for i in range(n):
        print('|\t{}\t|\t{}\t|'.format(
            '\t'.join([str(round(x, 3)) for x in list(a[i])]),
            round(b[i, 0], 3)))
    print('')

def matrixsolver(matrix, b):
    m, n = matrix.shape
    x = np.zeros(n)
    printaugmented(matrix, b, n)
    for j in range(1, n):
        for i in range(j):
            mult = matrix[j, i] / matrix[i, i]
            matrix[j] = matrix[j] - mult * matrix[i]
            b[j] = b[j] - mult * b[i]
            printaugmented(matrix, b, n)
            
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            b[i] = b[i] - matrix[i, j] * x[j]
        x[i] = b[i] / matrix[i, i]
    return x
```

**(A) Use your program to solve the following set of equations.**

$$
\begin{aligned}
3 x_1 + x_2 - 2 x_3 &=& -17\\
-6 x_1 + 2x_2 + 2x_3 &=& 4\\
-x_1 + 3x_2 + 2x_3 &=& 1
\end{aligned}
$$


```
q1a = np.array([[3, 1, -2],
               [-6, 2, 2],
               [-1, 3, 2]], dtype=float)
q1b = np.array([[-17],
                [4],
                [1]], dtype=float)

print(solve(q1a, q1b))
matrixsolver(q1a, q1b)
```

    [[ 0.22222222]
     [-4.11111111]
     [ 6.77777778]]
    |	3.0	1.0	-2.0	|	-17.0	|
    |	-6.0	2.0	2.0	|	4.0	|
    |	-1.0	3.0	2.0	|	1.0	|
    
    |	3.0	1.0	-2.0	|	-17.0	|
    |	0.0	4.0	-2.0	|	-30.0	|
    |	-1.0	3.0	2.0	|	1.0	|
    
    |	3.0	1.0	-2.0	|	-17.0	|
    |	0.0	4.0	-2.0	|	-30.0	|
    |	0.0	3.333	1.333	|	-4.667	|
    
    |	3.0	1.0	-2.0	|	-17.0	|
    |	0.0	4.0	-2.0	|	-30.0	|
    |	0.0	0.0	3.0	|	20.333	|
    





    array([ 0.22222222, -4.11111111,  6.77777778])



**(B) Use your program to solve the following set of equations.**

$$
\begin{aligned}
0.729 x_1 + 0.81 x_2 + 0.9x_3 &=& 0.6867\\
x_1 + x_2 + x_3 &=& 0.8338\\
1.331 x_1 + 1.21 x_2 + 1.1x_3 &=& 1
\end{aligned}
$$


```
q2a = np.array([[0.729, 0.81, 0.9],
               [1, 1, 1],
               [1.331, 1.21, 1.1]], dtype=float)
q2b = np.array([[0.6867],
               [0.8338],
               [1]], dtype=float)

print(solve(q2a, q2b))
matrixsolver(q2a, q2b)
```

    [[ 0.22454545]
     [ 0.28136364]
     [ 0.32789091]]
    |	0.729	0.81	0.9	|	0.687	|
    |	1.0	1.0	1.0	|	0.834	|
    |	1.331	1.21	1.1	|	1.0	|
    
    |	0.729	0.81	0.9	|	0.687	|
    |	0.0	-0.111	-0.235	|	-0.108	|
    |	1.331	1.21	1.1	|	1.0	|
    
    |	0.729	0.81	0.9	|	0.687	|
    |	0.0	-0.111	-0.235	|	-0.108	|
    |	0.0	-0.269	-0.543	|	-0.254	|
    
    |	0.729	0.81	0.9	|	0.687	|
    |	0.0	-0.111	-0.235	|	-0.108	|
    |	0.0	0.0	0.024	|	0.008	|
    





    array([ 0.22454545,  0.28136364,  0.32789091])



**(C) Use your program to solve the following set of equations.**

$$
\begin{aligned}
3 x_1 + x_2 - 2 x_3 &=& -17\\
6 x_1 + 2x_2 + -4x_3 &=& 4\\
-x_1 + 3x_2 + 2x_3 &=& 1
\end{aligned}
$$


```
q3a = np.array([[3, 1, -2],
               [6, 2, -4],
               [-1, 3, 2]], dtype=float)
q3b = np.array([[-17],
               [4],
               [1]], dtype=float)

try:
    print(solve(q3a, q3b))
except LinAlgError:
    print('Singular Matrix')
matrixsolver(q3a, q3b)
```

    Singular Matrix
    |	3.0	1.0	-2.0	|	-17.0	|
    |	6.0	2.0	-4.0	|	4.0	|
    |	-1.0	3.0	2.0	|	1.0	|
    
    |	3.0	1.0	-2.0	|	-17.0	|
    |	0.0	0.0	0.0	|	38.0	|
    |	-1.0	3.0	2.0	|	1.0	|
    
    |	3.0	1.0	-2.0	|	-17.0	|
    |	0.0	0.0	0.0	|	38.0	|
    |	0.0	3.333	1.333	|	-4.667	|
    
    |	3.0	1.0	-2.0	|	-17.0	|
    |	0.0	0.0	0.0	|	38.0	|
    |	nan	nan	nan	|	-inf	|
    


    -c:14: RuntimeWarning: divide by zero encountered in double_scalars
    -c:15: RuntimeWarning: invalid value encountered in multiply





    array([ nan,  nan,  nan])



**2. Compute the condition number of the three matrices in problem 1 using the $||.||_\infty$ norm. What do those numbers tell you about those matrices? Does that line up with what happened when you ran your Gaussian elimination code on them?**

The condition number for a given matrix $m$ is given by $||m||_\infty \cdot ||m^{-1}||_\infty$.


```
infinitynorm = lambda m: max([np.sum([np.abs(entry) for entry in row])
                          for row in m])

print('Matrix 1A Condition Number: {}'.format(infinitynorm(q1a) * infinitynorm(inv(q1a))))
print('Matrix 2A Condition Number: {}'.format(infinitynorm(q2a) * infinitynorm(inv(q2a))))
print('Matrix 3A Condition Number: {}'.format(infinitynorm(q3a) * infinitynorm(inv(q3a))))
```

    Matrix 1A Condition Number: 3.4999999999999996
    Matrix 2A Condition Number: 232.59190909090682
    Matrix 3A Condition Number: nan


These answers make sense in terms of our calculated solutions. We know the third matrix is Singular, meaning no answer can be calculated. The new information we glean comes from the second matrix's condition number. This high value indicates that even a small error in $\vec{b}$ will result in a large error in the solution.

**3. Solve the system in problem 1b. by hand using Gaussian elimination with partial pivoting. Do you get the same answer with and without pivoting? Why or why not?**

System 1b is the following.

$$
\left[
\begin{array}{ccc|c}
0.729 & 0.81 & 0.9 & 0.6867\\
1 & 1 & 1 & 0.8338\\
1.331 & 1.21 & 1.1 & 1
\end{array}
\right]
$$

Using partial pivoting we first switch the rows around and then solve using partial pivoting.

$$
\begin{array}{ccc}
\left[
\begin{array}{ccc|c}
0.729 & 0.81 & 0.9 & 0.6867\\
1 & 1 & 1 & 0.8338\\
1.331 & 1.21 & 1.1 & 1
\end{array}
\right] & L_1 \leftrightarrow L_3 &
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
1 & 1 & 1 & 0.8338\\
0.729 & 0.81 & 0.9 & 0.6867\\
\end{array}
\right]\\
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
1 & 1 & 1 & 0.8338\\
0.729 & 0.81 & 0.9 & 0.6867\\
\end{array}
\right] & -\frac{L_1}{1.331} + L_2 \rightarrow L_2 &
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
0 & 0.0909 & 0.17355 & 0.0824\\
0.729 & 0.81 & 0.9 & 0.6867\\
\end{array}
\right]\\
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
0 & 0.0909 & 0.17355 & 0.0824\\
0.729 & 0.81 & 0.9 & 0.6867\\
\end{array}
\right] & -\frac{0.729 L_1}{1.331} + L_3 \rightarrow L_3 &
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
0 & 0.0909 & 0.17355 & 0.0824\\
0 & 0.1472 & 0.2975 & 0.1389\\
\end{array}
\right]\\
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
0 & 0.0909 & 0.17355 & 0.0824\\
0 & 0.1472 & 0.2975 & 0.1389\\
\end{array}
\right] & L_2 \leftrightarrow L_3 &
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
0 & 0.1472 & 0.2975 & 0.1389\\
0 & 0.0909 & 0.17355 & 0.0824\\
\end{array}
\right]\\
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
0 & 0.1472 & 0.2975 & 0.1389\\
0 & 0.0909 & 0.17355 & 0.0824\\
\end{array}
\right] & - \frac{0.0909L_2}{0.1472} + L_3 \rightarrow L_3 &
\left[
\begin{array}{ccc|c}
1.331 & 1.21 & 1.1 & 1\\
0 & 0.1472 & 0.2975 & 0.1389\\
0 & 0 & -0.0101 & -0.0033\\
\end{array}
\right]\\
\end{array}
$$

Solving using back-substitution we obtain the following.

$$
\begin{cases}
z = \frac{-0.0033}{-0.0101} = 0.224545454545\\
y = \frac{0.1389 - (0.2975 z)}{0.1472} = 0.281363636364\\
x = \frac{1 - (1.1z) - (1.21y))}{1.331} = 0.327890909091\\
\end{cases}
$$

We also go through this problem computationally in order to make sure our work is correct.


```
m = np.array([[0.729, 0.81, 0.9, 0.6867],
            [1, 1, 1, 0.8338],
            [1.331, 1.21, 1.1, 1]], dtype=np.float64)
print('Initial Matrix')
print(m)
m[[0, 2]] = m[[2, 0]]
print('First Pivot')
print(m) # Pivot
m[1] = m[1] - (m[0] / m[0, 0])
print('First Column')
print(m)
m[2] = m[2] - (m[0] * m[2, 0] / m[0, 0])
print(m) # Clear first column
m[[1, 2]] = m[[2, 1]]
print('Second Pivot')
print(m) # Pivot
m[2] = m[2] - (m[1] * m[2, 1] / m[1, 1])
print('Second Column')
print(m) # Clear second column

print('Variables')
z = m[2, 3] / m[2, 2]
y = (m[1, 3] - (z * m[1, 2])) / m[1, 1]
x = (m[0, 3] - (z * m[0, 2]) - (y * m[0, 1])) / m[0, 0]

print(x, y, z)
```

    Initial Matrix
    [[ 0.729   0.81    0.9     0.6867]
     [ 1.      1.      1.      0.8338]
     [ 1.331   1.21    1.1     1.    ]]
    First Pivot
    [[ 1.331   1.21    1.1     1.    ]
     [ 1.      1.      1.      0.8338]
     [ 0.729   0.81    0.9     0.6867]]
    First Column
    [[ 1.331       1.21        1.1         1.        ]
     [ 0.          0.09090909  0.17355372  0.0824852 ]
     [ 0.729       0.81        0.9         0.6867    ]]
    [[ 1.331       1.21        1.1         1.        ]
     [ 0.          0.09090909  0.17355372  0.0824852 ]
     [ 0.          0.14727273  0.29752066  0.13899151]]
    Second Pivot
    [[ 1.331       1.21        1.1         1.        ]
     [ 0.          0.14727273  0.29752066  0.13899151]
     [ 0.          0.09090909  0.17355372  0.0824852 ]]
    Second Column
    [[ 1.331       1.21        1.1         1.        ]
     [ 0.          0.14727273  0.29752066  0.13899151]
     [ 0.          0.         -0.01010101 -0.00331203]]
    Variables
    0.224545454545 0.281363636364 0.327890909091


These are the same answers we obtain without partial pivoting because the error is contained even though our matrix is ill-conditioned.

**4. Write down the $PA = LU$ factorization of the system in problem 1b and use those factors to solve this system:**

$$
\left[
\begin{array}{ccc|c}
0.729 & 0.81 & 0.9 & 0.7\\
1 & 1 & 1 & 0.8\\
1.331 & 1.21 & 1.1 & 1.1\\
\end{array}
\right]
$$

**Do this by hand, in the most efficient way possible. Please explain why LU factorization is a good idea.**

We've already done the work for the $PA = LU$ factorization in the prior problem, so now we'll just write it all down.

$$
\begin{array}{ccccc}
\left[
\begin{array}{ccc}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0\\
\end{array}
\right] &
\left[
\begin{array}{ccc}
0.729 & 0.81 & 0.9\\
1 & 1 & 1\\
1.331 & 1.21 & 1.1\\
\end{array}
\right] &
= &
\left[
\begin{array}{ccc}
1 & 0 & 0\\
\frac{0.729}{1.331} & 1 & 0\\
\frac{1}{1.331} & \frac{0.0909}{0.1472} & 1\\
\end{array}
\right] &
\left[
\begin{array}{ccc}
1.331 & 1.21 & 1.1\\
0 & 0.14727273 & 0.29752066\\
0 & 0 & -0.01010101\\
\end{array}
\right]\\
P & A & = & L & U\\
\end{array}
$$

The big advantage of $PA = LU$ factorization is that it allows one to solve a system who's corresponding $\vec{A}$ has already been solved, but the corresponding $\vec{b}$ has changed.

We'll now solve the new system.


```
m = np.array([[1.331, 1.21, 1.1, 0.7],
              [0, 0.1472727, 0.29752066, 0.8],
              [0, 0, -0.010101, 1.1]])

print('Variables')
z = m[2, 3] / m[2, 2]
y = (m[1, 3] - (z * m[1, 2])) / m[1, 1]
x = (m[0, 3] - (z * m[0, 2]) - (y * m[0, 1])) / m[0, 0]

print(x, y, z)
```

    Variables
    -114.412498418 225.432359657 -108.9001089


**5. What is the difference between Gaussian elimination and the Gauss-Jordan method, in terms of the way they work? Which one is more computationally efficient?**

Gaussian Elimination relies on transforming the initial system $\vec{A}$ into an upper triangular matrix. The Gauss-Jordan method on the other hand fully transforms the matrix in its reduced row echelon form. In terms of computational efficacy, Gaussian elimination requires less work as it relies on the previous step to eliminate work for itself. Fully reducing a system to its reduced form can be computationally expensive as it requires many more steps.
