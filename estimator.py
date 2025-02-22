import jax.numpy as jnp
from jax import random, jit
import jax

# The first two functions cumsum2 and getfun are the helper functions, with the dynamic programming algorithm in getfun.
# Use momest to estimate the spectral moments.

def cumsum2(x,axis=0):
    return jnp.cumsum(x,axis=axis)-x

def getfun(X0,X1,kmax):
    P=jnp.shape(X0)[0]
    Q=jnp.shape(X0)[1]
    idcs = jnp.tile(jnp.arange(P)[:, None], (1, Q))
    def compute_m_one(i):
        S = P * jnp.where(idcs == i, X0, 0)
        X0sub = jnp.where(idcs >= i, X0, 0)
        X1sub = jnp.where(idcs >= i, X1, 0)
        m_one = []
        for k in jnp.arange(kmax-1) + 1:
            S = ((k + 1) ** 2) / ((P - k) * (Q - k)) * cumsum2(cumsum2(S, axis=0) * X1sub, axis=1) * X0sub
            m_one.append(jnp.sum((jnp.sum(S, axis=0) / P) * X1[i, :]) / Q)
        return jnp.array(m_one)  
    return compute_m_one

# This is the function that computes the moment estimates from the 2nd moment to kmax-th moment.
# If kmax=4 is given, the function will return the 2nd, 3rd, and 4th moments.
# If there is a noise in measurement, one can provide two measurements X0 and X1 matrices from two trials. This will help reduce trial-to-trial noise.
# If only one instance of the measurement matrix X is available, simply provide X as X0 and X1. 
# Since this algorithm only averages over the cyclic paths of increasing indices, 
# the function also provides an option to repeat this algorithm over multiple random row-column permutations of the provided matrix/matrices.
# The number of repetitions is specified by reps.
def momest(X0,X1,kmax=5,reps=1):
    P=jnp.shape(X0)[0]
    Q=jnp.shape(X0)[1]
    key = random.PRNGKey(0)
    Mvs2=[]
    for u in range(reps):
        key1,key2,key = random.split(key,3)
        X0_new=random.permutation(key2,random.permutation(key1, X0, axis=0),axis=1)
        X1_new=random.permutation(key2,random.permutation(key1, X1, axis=0),axis=1)
        compute_m_one = getfun(X0_new,X1_new,kmax)
        # Vectorize the computation across the first dimension
        compute_m_one_vmap = jax.vmap(compute_m_one, in_axes=(0,))
        Ms = compute_m_one_vmap(jnp.arange(P))
        Ms0 = jnp.sum(Ms, axis=0) / P
        Mvs2.append(jnp.squeeze(Ms0))    
    mv2=jnp.mean(jnp.array(Mvs2),axis=0)
    return mv2
