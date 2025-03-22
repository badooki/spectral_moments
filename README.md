# SPECTAL MOMENT ESTIMATOR

## Understanding the eigenvalue spectrum of an infinitely large matrix from its sample submatrices

Given a finite size data matrix $` X \in \mathbb{R}^{P\times Q} `$, we typically look at the eigenvalues of its covariance matrix $` \frac{1}{Q}XX^\top `$.
But, if $` X `$ is a submatrix of an underlying infinite matrix, the eigenvalue density obtained from $` X `$ is a biased estimate of the true eigenvalue density!

We fix this fundamental problem in our paper: "Estimating the Spectral Moments of the Kernel Integral Operator from Finite Sample Matrices": [Link to the paper](https://arxiv.org/abs/2410.17998).

The strategy is to obtain the unbiased estimator of the *moments* of the true eigenvalues (i.e. *spectral moments*), and one can choose their favorite algorithm to convert the estimated moments to a distribution.
The code in this repository computes the unbiased estimate of the spectral moments, using a dynamic programming method.

---

## Code

Requirement:
- JAX 

`momest` function in `estimator.py` implements our dynamic programming algorithm for computing the spectral moments from a measurement matrix. A typical usage will look like
```
moment_estimates = momest(X0,X1,kmax=5,reps=1)
```
The function computes the moment estimates from the 2nd moment to kmax-th moment. For example, If kmax=4 is given, the function will return the 2nd, 3rd, and 4th moments.

If there is a noise in measurement, one can provide two measurements X0 and X1 matrices from two trials. This will help reduce trial-to-trial noise. If only one instance of the measurement matrix X is available, simply provide X as X0 and X1. 

Since this algorithm only averages over the cyclic paths of increasing indices, the function also provides an option to repeat this algorithm over multiple random row-column permutations of the provided matrix/matrices. The number of repetitions is specified by reps.

The Jupyter notebooks in this repository reproduce the plots in the paper.
