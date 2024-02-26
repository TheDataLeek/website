Intro to Statistics
===================

We start with an introduction to histograms, assuming that the reader is
familiar with the absolute basic terminology of statistics. A histogram
is just a way to display data similar to a bar chart.

[!ht]

  ------------------- -----------------------------------
  Unimodal            Rise to a single peak and decline
  Bimodal             Two separate peaks
  Multimodal          Any number of peaks
  Symmetric           Right and left sides mirrored
  Positively Skewed   Data stretches to right
  Negatively Skewed   Data stretches to left
  ------------------- -----------------------------------

[table:histograms]

The relative frequency of a group of values is number of times the value
occurs divided by the number of observations, while the absolute
frequency is the numerator.

Measuring Data Location
-----------------------

The mean (average) is a useful way to measure the center of data. Where
$\bar{x}$ is the sample mean and $\bar{\mu}$ is the population mean.

$$\bar{x} = \frac{x_1 + x_2 + \cdots + x_n}{n} = \left( \frac{1}{n} \right) \sum^n_{i=1} x_i$$

We can also use the median (center) where again $\tilde{x}$ is the
sample median and $\tilde{\mu}$ is the population median. The median
divides data up into two equal parts, but this concept can be extended
to allow for quartiles and percentiles.

$$\tilde{x} = \begin{cases}
                            \text{Single middle value}\\
                            \text{Average of two middle values}
                        \end{cases}$$

As well as the mode, which is the most frequent data point.

A trimmed mean is a compromise between the mean and median. With a
trimmed mean trims the ends in order to remove outliers.

Measuring Variability
---------------------

We can measure variability of our data with a variety of different
methods, for instance the range is the difference between the largest
data point and the smallest.

The sample variance (denoted $s^2$) is given by

$$s^2 = \frac{\Sigma {\left( x_i - \bar{x} \right) }^2}{n-1} = \frac{S_{xx} }{n-1}$$

While the sample standard deviation is given by the square root of the
variance,

$$s = \sqrt{s^2}$$

Probability
===========

An experiment is anything who’s outcome is uncertain. The sample space
($\mathcal{S}$) of an experiment, is the set of all possible outcomes
for said experiment. An event is any subset of outcomes contained in the
sample space. Since events are subsets, we can pull in set theory and
the concepts associated.

Axioms of Probability
---------------------

[itemize] @ For any event $A$, $0 \le P(A) \le 1$. @
$P(\mathcal{S}) = 1$. @ If $A_1, A_2, A_3, \ldots$ is an infinite
collection of disjoint events, then

$$P(A_1 \cup A_2 \cup A_3 \cup \cdots) = \sum^\infty_{i=1}P(A_i)$$ @ For
any event $A$, if $P(A) + P(A^\prime) = 1$, then
$P(A) = 1 - P(A^\prime)$. @ For any two events,
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$ @ For any three events,
$$\begin{aligned}
            P(A \cup B \cup C) =& P(A) + P(B) + P(C)\\
                                & - P(A \cap B) - P(A \cap C) - P(B \cap C)\\
                                & + P(A \cap B \cap C)
        \end{aligned}$$

Conditional Probability
-----------------------

We can condition the probability of events on the outcomes of other
events. This uses the notation $P(A|B)$ where we say the conditional
probability of $A$ given that $B$ has occurred.

For any two events $A$ and $B$ with $P(B) > 0$, the conditional
probability of $A$ given that $B$ has occurred is defined by

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

We also have a couple rules that apply.

@ The Multiplication Rule $ \to P(A \cap B) = P(A|B) \cdot P(B)$. @ The
Law of Total Probability @@ Let $A_1, \ldots, A_k$ be mutually exclusive
and exhaustive events. Then for any other event $B$, $$\begin{aligned} 
                    P(B) =& P(B|A_1)P(A_1) + \cdots + P(B|A_k)P(A_k)\\
                         =& \sum^k_{i=1} P(B|A_i)P(A_i)
                \end{aligned}$$ @ Bayes’ Theorem @@ Let
$A_1, \ldots, A_k$ be a collection of $k$ mutually exclusive and
exhaustive events with prior probabilities
$P(A_i) (i = 1, 2, \ldots, k)$. Then for any other event $B$ for which
$P(B) > 0$, the posterior probability of $A_j$ given that $B$ has
occurred is $$\begin{aligned}
                    P(A_j|B) =& \frac{P(A_j \cap B)}{P(B)}\\
                             =& \frac{P(B|A_j)P(A_j)}{\sum^k_{i=1} P(B|A_i) \cdot P(A_i)}
                    j = 1, 2, \ldots, k
                \end{aligned}$$

Independence
------------

Two events $A$ and $B$ are independent if $P(A|B) = P(A)$ and dependent
otherwise, which means that $P(A \cap B) =
    P(A) \cdot P(B)$.

Discrete Random Variables and Probability Distributions
=======================================================

For a given sample space $\mathcal{S}$ of some experiment, a random
variable (rv) is any rule that associates a number with each outcome in
$\mathcal{S}$. We usually use uppercase letters for random variables
($X, Y, Z$) and lowercase letters for particular values ($x, y, z$).

We have discrete and continuous random variables, which are defined as
the common definition. However they differ in one respect, which is that
with continuous random variables no single point has positive
probability, only intervals have probability.

Probability Distributions for Discrete Random Variables
-------------------------------------------------------

The probability mass function (pmf) of a discrete random variable is
defined for every number $x$ by
$p(x)=P(X=x)=P(\text{all }s\in\mathcal{S}:X(s)=x)$.

The cumulative distribution function (cdf) $F(x)$ of a discrete random
variable $X$ with pmf $p(x)$ is defined for every number $x$ by
$$F(x) = P(X \le x) = \sum_{y:y \le x} p(y)$$ For any number $x$, $F(x)$
is the probability that the observed value of $X$ will be *at most* $x$.

Expected Values and Variance
----------------------------

Let $X$ be a discrete random variable with set of possible values $D$
and pmdf $p(x)$. The expected value, of mean of $X$, denoted $E(X)$ or
$\mu_X$, or just $\mu$ is $$E(X) = \mu_X = \sum_{x \in D} x \cdot p(x)$$

This has some defining rules

$$E(aX + b) = a \cdot E(X) + b$$

We can also calculate the variance and standard deviation, which are
measures of spread and distribution.

Let $X$ have pmf $p(x)$, and expected value $\mu$. Then the variance of
$X$, denoted by $V(X)$, or $\sigma^2_X$, or just $\sigma^2$ is

$$V(X) = \sum_D {(x - \mu)}^2 \cdot p(x) = E[(X - \mu^2)]$$

The standard deviation of $X$ is

$$\sigma_X = \sqrt{\sigma^2_X}$$

We have a shortcut formula for $\sigma^2$.

$$V(X) = \sigma^2 = \left[ \sum_D x^2 \cdot p(x) \right] - \mu^2 = E(X^2) - {[E(X)]}^2$$

And again, we have some rules.

$$\sigma_{aX} = |a| \cdot \sigma_X, \sigma_{X+b} = \sigma_X$$

Geometric and Bernoulli Random Variables
----------------------------------------

Any random variable whose only possible outcomes are 0 and 1 are called
Bernoulli Random Variables. For any Bernoulli Random Variable we can
establish the pmf.

$$p(x) = \begin{cases}
                        p^x {(1 - p)}^{x-1} &\to x = 1, 2, 3, \ldots\\
                        0 &\to Otherwise
                   \end{cases}$$

Where $p$ can be any value in $[0,1]$. Depending on the value of $p$ we
get different members of the Geometric Distribution. Therefore a
Bernoulli Random Variable is the measure of outcomes of binary
experiments. It is a discrete variable that takes on values 0 or 1, with
$\pi_1 = p(X=1)$. On the other hand, Geometric Random Variables measure
the time (number of trials) until a certain outcome occurs.

The Binomial Probability Distribution
-------------------------------------

There are many experiments that conform to the following requirements,
which mark it as a binomial experiment.

@ The experiment consists of a sequence of $n$ smaller experiments call
trials, where $n$ is fixed in advance of the experiment. @ Each trial
can result in one of the same two possible outcomes which we generally
denote by Success and Failure. @ The trials are independent, so that the
outcome of any particular trial does not influence the outcome of any
other trial. @ The probability of Success from trial to trial is
constant by which we denote $p$.

Therefore the binomial random variable $X$ is defined as the number of
Successes in $n$ trials. Since this depends on two factors, we write the
pmf as

$$b(x;n, p) = \begin{cases}
                            \binom{n}{x} p^x {(1 - p)}^{n - x} &\to x = 0, 1, 2, 3, \ldots, n\\
                            0 &\to \text{Otherwise}
                        \end{cases}$$

If $X \to Bin(n, p)$, then $E(X) = np$, $V(X) = np(1 - p) = npq$, and
$\sigma_X = \sqrt{npq}$ where $q = 1 - p$.

Hypergeometric Distribution
---------------------------

We need to make some initial assumptions to use this distribution.

@ The population consists of $N$ elements. (A finite population) @ Each
element can be characterized as a Success of a Failure, and there are
$M$ successes in the population. @ A sample of $n$ elements is selected
without replacement in such a way that each subset of size $n$ is
equally likely to be chosen.

Like the binomial probability distribution, $X$ is the number of
successes in the sample.

$$P(X=x) = h(x;n, M, N) = \frac{\binom{M}{x} \binom{N-M}{n-x} }{\binom{N}{n} }$$

The mean and variance of this distribution are

$$E(X) = n \cdot \frac{M}{N}\qquad
            V(X) = \left( \frac{N-n}{N-1} \right) \cdot n \cdot \frac{M}{N} \cdot \left( 1 - \frac{M}{N} \right)$$

Negative Binomial Distribution
------------------------------

Again, we need to start with some assumptions.

@ The experiment consists of a sequence of independent trials. @ Each
trial can either result in Success of Failure. @ The probability of
Success is constant from trial to trial. @ The experiment continues
until a total of $r$ successes have been observed.

The pmf of the negative binomial distribution with parameters $r = $ the
number of Successes, and $p=P(S)$ is

$$nb(x;r,p) = \binom{x+r-1}{r-1} p^r {(1-p)}^x \quad x = 0, 1, 2, \ldots$$

The special case where $r=1$ is called the geometric distribution. The
mean and variance are as follows

$$E(X) = \frac{r(1-p)}{p} \qquad V(X) = \frac{r(1-p)}{p^2}$$

The Poisson Distribution
------------------------

A discrete random variable $X$ is said to have a Poisson Distribution
with parameter $\mu \; (\mu > 0)$ if the pmf of $X$ is

$$p(x; \mu) = \frac{e^{-\mu} \cdot \mu^x}{x!} \qquad x = 0, 1, 2, 3, \ldots$$

Suppose that in the binomial pmf we let $n \to \infty$ and $p \to 0$ in
such a way that $np$ approaches a value $\mu
    > 0$. Then $b(x;n,p)\to p(x;\mu)$.

The mean and variance of $X$ are refreshingly easy for the Poisson
Distribution.

$$E(X) = V(X) = \mu$$

We mostly use the Poisson distribution to measure events that occur over
time. The structure of this distribution requires us to make some
assumptions about the data being collected.

@ There exists a parameter $\alpha > 0$ such that for any short time
interval of length $\Delta t$, the probability that exactly one occurs
is $\alpha \cdot \Delta t + o(\Delta t)$ @ The probability of more than
one event occurring during $\Delta t$ is $o(\Delta t)$. @ The number of
events that occur during $\Delta t$ is independent of the number that
occur prior to this time interval.

We also can establish that
$P_k(t) = e^{-\alpha t} \cdot {(\alpha t)}^k/k!$ so that the number of
events during a time interval of length $t$ is a Poisson rv with
parameter $\mu = \alpha t$.The expected number of events during any such
time interval is $\alpha t$, so the expected number during a unit time
interval is $\alpha$.

The occurrence of events over time as described in known as the Poisson
Process.

Continuous Random Variables
===========================

Let $X$ be a continuous random variable. Then the probability
distribution of $X$ (pdf) is such that for any two numbers $a$ and $b$
where $a \le b$,

$$P(a \le X \le b) = \int^b_a f(x)\, dx$$

In essence, continuous random variables replace the $\Sigma$ with a
$\int$. Any pdf must be greater than or equal to zero, and the area
under the entire region must equal 1.

A continuous random variable $X$ is said to have uniform distribution on
$[A, B]$ if the pdf of $X$ is

$$f(x;A, B) =
        \begin{cases}
            \frac{1}{B-A} &\to A \le x \le B\\
            0 &\to Otherwise
        \end{cases}$$

Percentiles of Continuous Distributions
---------------------------------------

The $n$th percentile is defined as

$$p = F(\eta (p)) = \int^{\eta(p)}_{-\infty} f(y) \, dy$$

Median
------

The median can be found using our percentile formula, and is again
denoted $\tilde{\mu}$, and it satisfies $F(\tilde{\mu})=0.5$.

Expected Values and Variance
----------------------------

Expected value is pretty much the same

$$\mu_X = E(X) = \int^\infty_{-\infty} x \cdot f(x) \, dx$$

While the variance is

$$\sigma^2_X = V(X) = \int^\infty_{-\infty} {(x - \mu)}^2 \cdot f(x) \, dx = E[(X-\mu)^2] = E(X^2) - [E(X)]^2$$

The same properties apply, and the standard deviation remains
$\sigma_X = \sqrt{V(X)}$.

Probability Distributions
=========================

The Normal Distribution
-----------------------

A continuous random variable is said to have normal distribution with
parameters $\mu$ and $\sigma$ if the pdf of $X$ is

$$f(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi} } e^{\frac{-{(x-\mu)}^2}{2\sigma^2} }$$

This is often written as $X \to N(\mu, \sigma^2)$.

### The Standard Normal Distribution

If $\mu = 0$ and $\sigma=1$ this is defined as the standard normal
distribution (denoted by $Z$) with pdf

$$f(z;0,1) = \frac{1}{\sqrt{2\pi} } e^{-z^2/2}$$

Where the cdf is denoted by $\Phi(z)$.

We use tables to determine the values of these cdfs, which are used as
reference for other distributions.

### $z$ Values

$z_\alpha$ is the $z$ value for which $\alpha$ of the area under the $z$
curve lies to the right of $z_\alpha$.

### Non-Standard Normal Distributions

When we’re dealing with a nonstandard normal distribution, we can
standardize to the standard normal distribution with standardized
variable $Z = (X - \mu)/\alpha$. This means that

$$\begin{aligned}
                P(a \le X \le b) &=&  P\left( \frac{a - \mu}{\sigma} \le Z \le \frac{b - \mu}{\sigma} \right)\\
                                 &=& \Phi \left( \frac{b - \mu}{\sigma} \right) - \Phi \left( \frac{a - \mu}{\sigma} \right)\\
                P(X \le a) &=& \Phi \left( \frac{a - \mu}{\sigma} \right)\\
                P(X \ge b) &=& 1 - \Phi \left( \frac{b - \mu}{\sigma} \right)
            \end{aligned}$$

Exponential Distribution
------------------------

This distribution is handy to model the distribution of lifetimes,
mostly due to its memoryless property. This means that the distribution
remains the same regardless of what happened prior.

$$f(x:\lambda) =
        \begin{cases}
            \lambda e^{-\lambda x} &\to x \ge 0\\
            0 &\to Otherwise
        \end{cases}$$

Where we can calculate

$$\mu = \frac{1}{\lambda} \qquad \sigma^2 = \frac{1}{\lambda^2}$$

With cdf

$$F(x; \lambda) =
        \begin{cases}
            0 \to& x < 0\\
            1 - e^{-\lambda x} \to& x \ge 0
        \end{cases}$$

The Gamma Distribution
----------------------

We need to first discuss the Gamma Function. For $\alpha > 0$, the gamma
function $\Gamma (\alpha)$ is defined by
$$\Gamma(\alpha) = \int^\infty_0 x^{\alpha - 1} e^{-x} \, dx$$ Where

@ For any $\alpha > 1$,
$\Gamma(\alpha) = (\alpha - 1)\Gamma(\alpha - 1)$ @ For any positive
integer $n$, $\Gamma(n) = (n - 1)!$ @ $\Gamma(1/2) = \sqrt{\pi}$.

Now we can define the distribution to be

$$f(x; \alpha) =
        \begin{cases}
            \frac{x^{\alpha - 1}e^{-x} }{\Gamma(\alpha)} &\to x \ge 0\\
            0 &\to Otherwise
        \end{cases}$$

A random variable is said to have Gamma Distribution if the pdf of $X$
is

$$f(x; \alpha, \beta) =
        \begin{cases}
            \frac{x^{\alpha - 1}e^{-x/\beta} }{\beta^\alpha \Gamma(\alpha)} &\to x \ge 0\\
            0 &\to Otherwise
        \end{cases}$$

With mean and variance

$$E(X) = \mu = \alpha \beta \qquad V(X) = \sigma^2 = \alpha \beta^2$$

And cdf of the standard gamma distribution

$$F(x;\alpha) =
        \int^x_0 \frac{y^{\alpha - 1}e^{-y} }{\Gamma(\alpha)} \, dy$$

### Chi-Squared

$$f(x; v) = 
            \begin{cases}
                \frac{x^{v/2 - 1}e^{-x/2} }{2^{v/2} \Gamma(v/2)} &\to x \ge 0\\
                0 &\to x < 0
            \end{cases}$$

Weibull Distribution
--------------------

$$f(x; \alpha, \beta) = 
        \begin{cases}
            \frac{\alpha}{\beta^\alpha} x^{\alpha - 1} e^{-(x/\beta)^\alpha} &\to x \ge 0\\
            0 &\to x < 0
        \end{cases}$$

With mean and variance

$$\mu = \beta \Gamma ( 1 + 1/\alpha ) \qquad \sigma^2 = \beta^2 \left[ \Gamma(1 + 2/\alpha) - {\left( \Gamma(1 +
        1/\alpha) \right)}^2 \right]$$

And cdf

$$f(x; \alpha, \beta) = 
        \begin{cases}
            0 &\to x < 0\\
            1 - e^{-(x/\beta)^\alpha} &\to x \ge 0
        \end{cases}$$

Lognormal Distribution
----------------------

$$f(x; \mu, \sigma) = 
        \begin{cases}
            \frac{e^{-[\ln(x) - \mu]^2/(2\sigma^2)} }{\sigma x \sqrt{2 \pi} } &\to x \ge 0\\
            0 &\to x < 0
        \end{cases}$$

$$E(X) = e^{\mu + \sigma^2 / 2} \qquad V(X) = e^{2\mu + \sigma} \left( e^{\sigma^2} - 1\right)$$

Since it has normal distribution it can be expressed in terms of the
standard normal distribution $Z$.

Beta Distribution
-----------------

$$\begin{aligned}
            f(x; \alpha, \beta, A, B) =\\
            \begin{cases}
                \frac{1}{B-A} \cdot \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \cdot \Gamma(\beta)}
                {\left( \frac{x - A}{B - A} \right)}^{\alpha - 1} {\left(\frac{B - x}{B - A}\right)}^{\beta - 1}
                &\to A \le x \le B\\
                0 &\to Otherwise
            \end{cases}
        \end{aligned}$$

$$\mu = A + (B - A) \cdot \frac{\alpha}{\alpha + \beta} \qquad \sigma^2 = \frac{ {(B - A)}^2 \alpha
    \beta}{ {(\alpha+\beta)}^2 (\alpha + \beta + 1)}$$

Functions of Random Variables
=============================

This is a relatively straightforward concept. If we have a function of a
random variable, we can express this as an inequality and solve for the
cdf. Examples follow, and derivations left to the reader.

Let $X$ be a random variables with continuous distribution. Let
$Y = X^2$.

$$\begin{aligned}
        F_Y(y) &=& P\left( Y \le y \right)\\
               &=& P\left( X^2 \le y \right)\\
               &=& P\left( -\sqrt{y} \le X^2 \le \sqrt{y} \right)\\
               &=& F_x(\sqrt{y}) - F_x(-\sqrt{y})\\
        \text{Now differentiate to obtain $f_x$}\\
        f_Y(y) &=& \frac{1}{2\sqrt{y} } \left[ f_X(\sqrt{y}) + f_x(-\sqrt{y}) \right]
    \end{aligned}$$

Joint Probability Distributions
===============================

A joint probability distribution is one of the form where

$$F(a, b) = P\left( X \le a, Y \le b \right) \qquad -\infty < a, b < \infty$$

For joint discrete random variables we simply sum the two sets together.
With continuous random variables we doubly integrate them together.

Two random variables are said to be independent if

$$p(x, y) = p_X(x) \cdot p_Y(y)$$

To be honest, this concept is fairly straightforward. All joint
distributions look the same, save we have to represent their cdf with
two or more integrals. The big trick here is to *DRAW A PICTURE FIRST*.
This will save most headaches.

To find the marginal distribution from a joint distribution, integrate
(or sum) over the opposite variable.

$$f_X(x) = \int_y f(x, y) \, dy \qquad
    f_Y(y) = \int_x f(x, y) \, dx$$

Covariance
----------

The covariance between two variables is
$$Cov(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E(XY) - \mu_X \cdot \mu_Y$$

Correlation
-----------

$$p_{X, Y} = \frac{Cov(X, Y)}{\sigma_X \cdot \sigma_Y}$$

Properties
----------

$$\begin{aligned}
            {\text{Cov} }(aX + b, cY + d) = ac{\text{Cov} }(X, Y)\\
            {\text{Corr} }(aX + b, cY + d) = sign(ac) {\text{Corr} }(XY)\\
            -1 \le {\text{Corr} }(XY) \le 1
        \end{aligned}$$

Sums of Independent Random Variables
------------------------------------

We can determine the sum of two random variables accordingly. This
process is called the convolution of the two variables. The cumulative
distribution function is given

$$\begin{aligned}
            F_{X + Y}(a) &=& P\left\{ X + Y \le a \right\}\\
            &=& \iint\limits_{x + y \le a} f_X(x) f_Y(y) \, dx \, dy\\
            &=& \int_{-\infty}^\infty \int_{-\infty}^{a-y}f_X(x) f_Y(y) \, dx \, dy\\
            &=& \int_{-\infty}^\infty F_X(a - y)f_Y(y) \, dy
        \end{aligned}$$

If we differentiate, we obtain the probability mass function

$$f_{X+Y}(a)= \int_{-\infty}^\infty f_X(a - y)f_Y(y) \, dy$$

We can apply this concept to a slew of identically distributed random
variables.

Conditional Distributions
-------------------------

### Discrete

According to Bayes

$$P\left( E|F \right) = \frac{P(EF)}{P(F)}$$

If $X$ and $Y$ are discrete random variables we can continue this
definition to find the conditional probability mass function of $X$
given $Y$.

$$p_{X|Y}\left( x|y \right) = P\left\{ X = x | Y = y \right\} = \frac{p\left( x, y \right)}{p_Y(y)}$$

We see that the cumulative distribution function is also found

$$F_{X|Y}\left( x|y \right) = P\left\{ X \le x | Y = y \right\} = \sum_{a \le x} p_{X|Y}\left( a|y \right)$$

### Continuous

Extending the previous concepts we can apply Bayes’ notion of
conditionality to continuous random variables.

$$f_{X|Y}\left( x|y \right) = \frac{f\left( x, y \right)}{f_Y(y)}$$

Using this we can define the generalized form to be

$$P\left\{ X \in A|Y = y \right\} = \int_A f_{X|Y}\left( x|y \right)\,dx$$

With corresponding cdf

$$F_{X|Y}\left( a|y \right) \equiv P\left\{ X\le a | Y = y \right\} =
                \int_{\infty}^a f_{X|Y}\left( x|y \right) \, dx$$

Joint Probability Distribution of Functions of Random Variables
---------------------------------------------------------------

Point Estimation
================

We can use point estimation to determine certain parameters about a set
of data. $\theta$ is merely an estimate for some parameter, based on
given sample data.

@ Obtain Sample Data from each population under study. @ Based on the
sample data, estimate $\theta$ @ Conclusions based on sample estimates.

Note, different samples produce different estimates, even if the same
estimator is used. This means that we are interested in determining how
to find the best estimator with least error. Error can be defined in a
couple ways. The squared error is defined as
${( \hat{\theta} - \theta )}^2$ while the mean squared error is defined
as $MSE=E[{( \hat{\theta} - \theta )}^2]$. If among two estimators one
has a smaller MSE than another, the one with a smaller MSE is better.
Another good quality is unbiasedness ($E[\hat{\theta}] = \theta$), and
another quality is small variance ($Var[\hat{\theta}]$).

The standard error of an estimator is its $\sigma$. This roughly tells
us how accurate our estimation is.

Moments
-------

@ Equate sample characteristics to the corresponding population values.
@ Solve these equations for unknown parameters. @ The solution formula
is the estimator.

For $k = 1, 2, 3, \ldots$ the $k$th population moment, or $k$th moment
of the distribution $f(x)$ is $E(X^k)$.

Therefore the $k$th sample moment is

$$\frac{1}{n} \cdot \sum_{i=1}^n X^k_i$$

This system for the most part assumes that any sample characteristic is
indicative of the population.

Maximum Likelihood Estimators
-----------------------------

To find the MSE, a few things need to be done. Let’s assume that we’re
given a set of observations with the same distribution with unknown pdf.
First we need to find the joint density function for all observations,
which when the observations are independent is merely their product.
This joint distribution function is our likelihood function. We now need
to find the maximal value, by either taking its derivative and setting
it equal to zero, or by first taking the $\log$ and then deriving
following by setting equal to zero.

Central Limit Theorem
=====================

Any estimator has its own probability distribution. This distribution is
often referred to as the sampling distribution of the estimator.$\sigma$
is again referred to as the standard error of the estimator. This leads
to an interesting insight, that is $\overline{X}$ based on a large $n$
tends to be closer to $\mu$ than otherwise.

$$\begin{aligned}
            E(\overline{X}) \approx \mu\\
            V(\overline{X}) \approx \sigma^2 / n
        \end{aligned}$$

Let $X_1, X_2, \ldots, X_n$ be a random sample from a distribution with
mean $\mu$ and variance $\sigma^2$. If $n$ is sufficiently large[^1],
$\overline{X}$ has approximately a normal distribution with
$\mu_{\overline{X} } = \mu$ and $\sigma^2_{\overline{X} } = \sigma^2 / n$.
The larger $n$ is, the better the approximation.

Intervals
=========

The CLT tells us that as $n$ increases, the sample mean is normally
distributed. We can normalize our sample mean.

$$Z = \frac{\overline{X} - \mu}{\sigma / \sqrt{n} }$$

This allows us to define a confidence interval. We know

$$P \left( -1.96 < \frac{\overline{X} - \mu}{\sigma / \sqrt{n} } < 1.96 \right) = 0.95$$

Which means that the $100 ( 1 - \alpha)\%$ confidence interval is
defined as

$$\left(
    \overline{X} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n} },
    \overline{X} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n} }
    \right)$$

This confidence interval tells us that if this experiment were to be
performed again and again, 95% of the time our newly calculated interval
would contain the true population mean.

We can replace the instance of $\sigma$ (which is rarely known) with our
sample standard deviation, $S$.

The $t$ Distribution
--------------------

When our $n$ is less than 40, we need to use the $t$ distribution, which
has the exact same normalization process, save we now call it a $t$
distribution with $n - 1$ degrees of freedom.

Let $t_v$ denote the $t$ distribution with degrees of freedom $v$.

@ Each $t_v$ curve is bell shaped and centered at 0. @ Each $t_v$ curve
is more spread out than the standard normal. @ As $v$ increases, the
spread of $t_v$ decreases. @ $ \lim_{v \to \infty} t_v = z$.

One Sample $t$ Confidence Interval
----------------------------------

This confidence interval is defined as

$$\left(
        \overline{X} - t_{\alpha/2, n - 1} \cdot \frac{\sigma}{\sqrt{n} },
        \overline{X} + t_{\alpha/2, n - 1} \cdot \frac{\sigma}{\sqrt{n} }
        \right)$$

Confidence Intervals for Population Proportion
----------------------------------------------

If we have a certain proportion that we know about a population we can
emulate it with a binomial random variable, and

$$\sigma_X = \sqrt{np ( 1 - p)}$$

The natural estimator for $p$ is $\hat{p} = X / n$, or the fraction of
“successes” that we identify. We know that $\hat{p}$ has normal
distribution, and that
$E(\hat{p}) = P, \sigma_{\hat{p} } = \sqrt{p (1 - p) / n}$, therefore our
confidence interval is

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} }$$

Confidence Intervals for Variance of a Normal Population
--------------------------------------------------------

If we have our random sample again, then we also know that

$$\frac{(n - 1) S^2}{\sigma^2} = \frac{\sum {\left( X_i - \overline{X} \right)}^2}{\sigma^2}$$

has chi-squared distribution with $n - 1$ degrees of freedom, therefore
the confidence interval for the variance is defined as

$$\left(
        \frac{(n - 1)s^2}{\chi^2_{\alpha/2, n - 1} },
        \frac{(n - 1)s^2}{\chi^2_{1 - \alpha/2, n - 1} }
        \right)$$

Hypotheses Tests for One Sample
===============================

A statistical hypothesis is a claim about a value of a parameter. We
have two different types of hypotheses, the null hypothesis and the
alternative hypothesis. The null hypothesis is the status quo, while the
alternative hypothesis is the research hypothesis. The objective of
testing is to decide whether or not the null is valid. At the core, this
process initially favors the null hypothesis.

We need to consider three difference cases,

@ $H_a : \theta \neq \theta_0$ @ $H_a : \theta > \theta_0$ @
$H_a : \theta < \theta_0$

And we have two different types of errors:

@ A **Type I Error** is when the null is rejected but is true. @ A
**Type II Error** is when the null kept, but it is false.

We need a test statistic in order to determine the null’s validity. One
easy way is to standardize $\overline{X}$.

$$Z = \frac{\overline{X} - \mu}{\sigma / \sqrt{n} }$$

And we have three types, lower-tailed, upper-tailed and two-tailed.

We also need to consider proportions, in which case we standardize
again.

$$Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0 (1 - p_0)}{n} } }$$

And then we use $p$-values, which is the probability that any $z$-test
will occur on the standard normal curve. The smaller the $p$-value, the
more evidence there is that the null hypothesis is false.

$$P-Values =
    \begin{cases}
        1 - \Phi(z)\\
        \Phi(z)\\
        2 \left[ 1 - \Phi({\left\lvert z \right\rvert}) \right]
    \end{cases}$$

$t$ tests work the same way.

When $H_0$ is true, the $p$-values are distributed uniformly.

[^1]: n \> 40
