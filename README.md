# SPECTAL MOMENT ESTIMATOR

## Understanding the eigenvalue spectrum of an infinitely large matrix from its sample submatrices

Given a finite size data matrix $` X \in \mathbb{R}^{P\times Q} `$, we typically look at the eigenvalues of its covariance matrix $` \frac{1}{Q}XX^\top `$.
But, if $` X `$ is a submatrix of an underlying infinite matrix, the eigenvalue density obtained from $` X `$ is a biased estimate of the true eigenvalue density!

We fix this fundamental problem in our paper: "Estimating the Spectral Moments of the Kernel Integral Operator from Finite Sample Matrices": [Link to the paper](https://arxiv.org/abs/2410.17998).

The strategy is to obtain the unbiased estimator of the *moments* of the true eigenvalues (i.e. *spectral moments*), and one can choose their favorite algorithm to convert the estimated moments to a distribution.
The code in this repository computes the unbiased estimate of the spectral moments, using a dynamic programming method.

---

## Code
`momest` function in `estimator.py` implements our dynamic programming algorithm for computing the spectral moments from a measurement matrix (or two measurement matrices if two trials are available).
The Jupyter notebooks in this repository reproduce the plots in the paper.
