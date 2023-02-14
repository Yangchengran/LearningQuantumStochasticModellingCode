
'''
This module includes the transition matrix of all of the examples, such as 
1. Perturbed coin 2. The causal asymmetry  3. Phase enhanced example
The transition matrix acting on the right probability vectors.

I restrict all stochastic matrix acting on column vectors.
'''
#%%
# from pandas import array
from scipy import *
# from scipy.sparse import base
# from scipy.sparse.csc import csc_matrix
# from scipy.sparse.csr import csr_matrix
# import scipy.sparse as sps
# from scipy.sparse.linalg import eigs

# from scipy.stats import norm, uniform 
# from scipy.integrate import quad
import scipy.linalg as LA
import numpy as np


# from .tools import digits, digits_to_int

#%%
def causal_asymmetry(p, q) :
    """
    transitiona matrix causal asymmetry example
    -------------------
    Parameters
    -------------------
    p: float
    q: float
    -------------------
    
    """
    T0 = np.array([
        [1.-p, q*(1.-p), 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])

    T1 = np.array([
        [0., 0., 0.],
        [0., 1.-q, 1.],
        [0., 0., 0.]
    ])

    T2 = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [p, p*q, 0.]
    ])

    return np.array([T0, T1, T2])

def pcoin(p):
    """
    transition matrix for pcoin

    -------------------
    Parameters
    -------------------
    p:float
    -------------------
    
    """
    T0 = np.array([
        [1.-p,p],
        [0.,0.]
    ])
    T1 = np.array([
        [0.,0.],
        [p,1.-p]
    ])
    return np.array([T0,T1])


def qcycle_de(p, delta):
    """
    transition matrix for qcycle with parameters

    -------------------
    Parameters
    -------------------
    p: float
    delta:float
    -------------------
    
    """
    T0 = np.array([
        [1.-p-delta, 0., p],
        [0., 0., 0.],
        [0., 0., 0.]
    ])

    T1 = np.array([
        [0., 0., 0.],
        [p, 1.-p, 0.],
        [0., 0., 0.]
    ])

    T2 = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [delta, p, 1.-p]
    ])

    return np.array([T0, T1, T2])

def dis_renewal(N):
    """
    transition matrix for discrete renewal process with uniform distribution

    -------------------
    Parameters
    -------------------
    N: int

    -------------------
    
    """
    T0 = np.zeros((N,N))
    T1 = np.zeros((N,N))
    for i in range(N):
        if i < N-1:
            T0[i+1,i] = (N-i-1.)/(N-i)
        T1[0,i] = 1./(N-i)
    
    return np.array([T0,T1])


# %%

