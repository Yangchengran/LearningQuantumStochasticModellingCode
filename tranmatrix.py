
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

def nemo(p):
    """
    transition matrix for nemo process

    -------------------
    Parameters
    -------------------
    p:float    
    -------------------
    
    """
    T0 = np.array([
        [p,0.,0.5],
        [0.,0.,0.],
        [0.,0.,0.]
    ])

    T1 = np.array([
        [0.,0.,0.5],
        [1.-p, 0., 0.],
        [0.,1.0,0.]
    ])

    return np.array([T0,T1])



def bias_pcoin(p,q):
    """
    transition matrix for bias cpoin
    
    -------------------
    Parameters
    -------------------
    p:float
    q:float
    -------------------
    
    """
    T0 = np.array([
        [1.-p,q],
        [0.,0.]
    ])

    T1 = np.array([
        [0.,0.],
        [p,1.-q]
    ])

    return np.array([T0,T1])




#%%


def dis_ren_trc(N,J):
    '''
    transition matrix truncated discrete renewal process.

    Parameters:
    --------------------

    1. N| int: the number of causal states.
    2. J| int: the number of truncated states.

    Return: 
    --------------------
    2 times N-J times N-J array: the transition probability.
    '''
    trans = np.zeros((2,N-J,N-J))
    for i in range(N-J):
        if i != N-J - 1:
            trans[0][i+1,i] = (N-i-1)/(N-i) 
        else:
            trans[0][i,i] = (N-i-1.)/(N-i)
        trans[1][0,i] = 1./(N-i)
    
    return trans

# %%


def dising(J0,delta,N,T):
    """
    Description:
    -------------------
    Transition matrix for Dyson Ising spin chains
    
    Parameters:
    -------------------
    J0: float | Interaction tension
    delta: float | Interaction parameter 
    N: int | Interaction length
    T: float | Temperature

    
    Return:
    -------------------
    transition matrix: array | 2^N by 2^N
    """
    dim = 2**N

    # Evaluate the internal energy
    def internX(s):
        x = 0.
        for j in range(N-1):
            for k in range(1,N-j):
                x += J0*k**(-delta) * s[j]*s[j+k]
        return x
    
    # Evaluate the energy between two blocks
    def interY(sL,sR):
        y = 0. 
        for j in range(N):
            for k in range(N-j,N+1):
                y += J0*k**(-delta) * sL[j]*sR[(j+k) % N]
        return y

    def decimal_to_spin(i,length):
        s = [2*int(d) - 1 for d in format(i,"0{}b".format(length))]
        return s 

    def spin_to_decimal(s):
        L = len(s)
        di = 0
        for i,x in enumerate(s):
            v = (x + 1) // 2
            di += v * 2**(L-i-1)
        return int(di)


    def transfer_matrix():
        V = np.zeros((dim,dim))
        for i in range(dim):
            sL = decimal_to_spin(i,N)
            for j in range(dim):
                sR = decimal_to_spin(j,N) 
                E = 0.5*internX(sL) + interY(sL,sR) + 0.5*internX(sR) # Some value
                # print(E,sL,sR)
                V[j,i] = np.exp(-E/T)
        return V
    
    def block_trans():
        transm = transfer_matrix()
        lam,V = LA.eig(transm.T)
        i = lam.argmax()
        eta = max(np.real(lam))
        dd = np.real(V[:,i])
        transition = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                transition[i,j] = dd[i]/dd[j] * transm[i,j]/eta
        return transition

    def trans():
        block = block_trans()
        # print(block)
        trans = np.zeros((2,dim,dim)) #trans[0] represents the spin down; trans[1] represents spin up 
        for i in range(dim):
            sL = decimal_to_spin(i,N) 
            sR_down = np.append(sL[1:],-1)
            sR_up = np.append(sL[1:],1)
            j_down = spin_to_decimal(sR_down)
            # print(sL,sR_down, j_down)
            j_up = spin_to_decimal(sR_up)
            for k in range(dim):
                if k < 2**(N-1):
                    # print(i,j_down,k)
                    trans[0,j_down,i] += block[k,i]
                else:
                    trans[1,j_up,i] += block[k,i]
        
        return trans

    return trans()



# def sparse_dising(J0,delta:float,N,T) -> np.ndarray:
#     '''
#     Dyson Ising Spin chain

#     use the spin method directly

#     'Parameters:'
#     --------------------

#     J0: Interaction tension
#     delta: Interaction parameter 
#     N: Interaction length
#     T: Temperature

#     Return: 
#     --------------------
#     Sparse transition probability
#     '''

#     dim = 2**N

#     def decimal_to_spin(i,length):
#         s = [2*int(d) - 1 for d in format(i,"0{}b".format(length))]
#         return s 

#     def spin_to_decimal(s):
#         L = len(s)
#         di = 0
#         for i,x in enumerate(s):
#             v = (x + 1) // 2
#             di += v * 2**(L-i-1)
#         return int(di)
    
#     def site_energy(spins):
#         s0 = spins[0]

#         E = 0.0

#         for k in range(1,N+1):
#             E += J0/(k**delta)*s0*spins[k]
        
#         return E

#     # csc_matrix

#     # Construct the transfer matrix

#     transfer_matrix = []

#     for next_spin in [-1,1]:

#         col_ind = []
#         row_ind = []
#         data = []

#         for j in range(dim):

#             sL = decimal_to_spin(j,N)
#             sR = np.append(sL[1:],next_spin)

#             i = spin_to_decimal(sR)

#             row_ind.append(i)
#             col_ind.append(j)

#             E = site_energy(np.append(sL,next_spin))

#             data.append(np.exp(-E/T))

#         Tx = csc_matrix((data,(row_ind,col_ind)), shape=(dim,dim))
#         transfer_matrix.append(Tx)

        
#     # Evaluate the transition matrix according to the transfer matrix

#     # Lately we want to use sparse matrix method
#     tot_transfer = sum(transfer_matrix)

#     lamb, vec = eigs(tot_transfer.T,k=1)

#     eta = np.real(lamb[0]) # The leading eigenvalue

#     vec = np.real(vec.flatten())


#     # V = tot_trans - sps.identity(dim)

#     # e = csc_matrix(np.ones((1,dim)))
#     # A = sps.vstack([V[1:],e])
    

#     # b = np.zeros((dim,1))
#     # b[-1] = 1.0
#     # b = sps.csc_matrix(b)
#     # probs = sps.linalg.spsolve(A,b)

#     diag = sps.diags(vec)

#     inv_diag = sps.diags(1.0/vec)

#     transition = [diag.dot(matr).dot(inv_diag)/eta for matr in transfer_matrix]

#     return transition

    



        


# # %%


# def th_cycle(alpha):
#     '''
#     Description:
#     -------------------
#     Transition matrix for three states cycle 

#     Parameters
#     -------------------
#     alpha: float | parameter of the stochastic processes. 

#     Return
#     -------------------

#     trans: ndarray| the transition matrix of the stochastic processes.

#     '''
#     def kraus_operators(alpha):
#         X = np.array([
#             [1., -alpha+alpha*exp(1.j*2.*pi/3.)],
#             [0.,exp(1.j*2.*pi/3.)]
#         ])

#         U,S,V = LA.svd(X)
#         idx = S.argsort()[::-1]
#         U = U[:,idx]
#         S = S[idx]
#         V_dag = V.conj().transpose()[idx,:]
#         u0 = U[:,0]
#         u1 = U[:,0]
#         v0 = V_dag[:,0]
#         v1 = V_dag[:,1]

#         # Evaluate the quantum states
#         sigma2 = u0
#         sigma1 = v0
#         sigma0 = X.dot(u0)/LA.norm(X.dot(u0))

#         A1 = 1./S[0]*X
#         norm0 = sqrt(1.-(S[1]/S[0])**2)
#         A0 = norm0*LA.kron(sigma0.reshape(2,1),v1.reshape(1,2).conj())
#         p = sigma0.conj().transpose().dot(A0.conj().transpose()).dot(A0).dot(sigma0)
#         p = real(p)

#         return [A0,A1],p,[sigma0,sigma1,sigma2]

# # def transition(alpha):
#     kr,_,states = kraus_operators(alpha)

#     sigma0,sigma1,sigma2 = states
#     A0, A1 = kr
#     p = LA.norm(A0.dot(sigma0))**2
#     fp = LA.norm(A1.dot(sigma2))**2

#     T0 = np.array([
#         [p, 0. ,1. - fp],
#         [0.,0.,0.],
#         [0.,0.,0.]
#     ])

#     T1 = np.array([
#         [0.,0.,fp],
#         [1.-p,0.,0.],
#         [0.,1.,0.]
#     ])

#     return np.array([T0,T1])
# # %%



# def Rk_golden(R,k,p):
#     """
#     Description:
#     -------------------
#     transition matrix for R-k golden mean process
    
#     Parameters:
#     -------------------
#     R: int | Markov order
#     k: int | cryptic order
#     p: float | probability
    
#     Return:
#     -------------------
    
#     """
#     T0 = np.zeros((R+k,R+k))
#     T1 = np.zeros((R+k,R+k))

#     T0[1,0] = 1-p
#     T1[0,0] = p
#     for i in range(1,R):
#         T0[i+1,i] = 1.0
#     for i in range(R,R+k):
#         j = (i+1) % (R+k)
#         T1[j,i] = 1.0
    
#     return np.array([T0,T1])


# # %%



# def cyclic_walk(n,cdf,cycle_time=1):

#     """
#     Description:
#     -------------------
#     Transition matrix for Cyclic random walks
    
#     Parameters:
#     -------------------
#     n: int | the number of classical bits.
#     cdf: cdf | the random variable. (some input function)
#     cycle_time: int | the number of cycles we consider.
    
#     Return:
#     -------------------
    
#     """

#     N = 2**n

#     delta = 1.0/(2*N)

#     ptot = cdf(cycle_time-delta) - cdf(-cycle_time-delta) # The total truncation probability

    
#     def state2state_prob(current_state, next_state):
#         point = next_state - current_state

#         # rotate the current state to the center of the cycle
#         if point < 0:
#             point += 1
#         points = [point + i * 1.0 for i in range(-cycle_time,cycle_time)] 

#         probs = [cdf(point+delta) - cdf(point-delta) for point in points]

#         return sum(probs)/ptot
            

#     transition = np.zeros((N,N,N))
#     for i in range(N):
#         current_state = i/N
#         for j in range(N):
#             next_state = j/N
#             transition[j,j,i] = state2state_prob(current_state,next_state)
    
#     return transition



# # %%



# def sparse_cyclic_walk(n,cdf,cycle_time=1):
#     """
#     Description:
#     -------------------
#     sparse transition matrix for Cyclic random walks
    
#     Parameters:
#     -------------------
#     n: int | the number of classical bits.
#     cdf: cdf | the random variable. (some input function)
#     cycle_time: int | number of cycles we consider.
    
#     Return:
#     -------------------
    
#     """

#     N = 2**n

#     delta = 1.0/(2*N)

#     ptot = cdf(cycle_time-delta) - cdf(-cycle_time-delta) # The total truncation probability

    
#     def state2state_prob(current_state, next_state):
#         point = next_state - current_state

#         # rotate the current state to the center of the cycle
#         if point < 0:
#             point += 1
#         points = [point + i * 1.0 for i in range(-cycle_time,cycle_time)] 

#         probs = [cdf(point+delta) - cdf(point-delta) for point in points]

#         return sum(probs)/ptot
    
#     data = []
#     for i in range(N):

#             current_state = i/N
#             prob = state2state_prob(current_state,0.0)

#             data.append(prob)


#     ind = list(range(N))

#     transition = []
#     for j in range(N):


#         indptr = np.zeros(N+1,dtype=int)

#         indptr[j+1:N] = N
#         indptr[N] = N

#         mm = csr_matrix((data,ind,indptr),shape=(N,N))

#         x = ind.pop(0)
#         ind.append(x)
#         transition.append(mm)

    
#     return transition

# # %%

# def dual_poisson(p,Q,N):
#     """
#     Description:
#     -------------------
#     Dual Poisson process
    
#     Parameters:
#     -------------------
#     p: float | probability
#     Q: list of float | decay rate of the two channels
#     N: int | the number of state
    
#     Return:
#     -------------------
    
#     """

#     q1,q2 = Q

# 	# The survival probability
#     phi = lambda k : p*q1**k + (1-p)*q2**k 


# 	# Construct transition matrix
#     T0 = np.zeros((N,N))
#     T1 = np.zeros((N,N))

#     for i in range(N-1):
#         p = phi(i+1)/phi(i)
#         T0[i+1,i] = p
#         T1[0,i] = 1 - p

#     p =  phi(N)/phi(N-1)
#     T0[N-1,N-1] = p
#     T1[0,N-1] = 1 - p

#     return [T0,T1]

# # %%


# def four_state_machine(v):
#     """
#     Description:
#     -------------------
#     4 state model
    
#     Parameters:
#     -------------------
#     v: ndarray | probability
    
#     Return:
#     -------------------
    
#     """
#     trans = np.zeros((2,4,4))

#     for x in range(2):
#         for i in range(4):
#             seq = digits(i,base=2,pad=2)
#             seq[0] = seq[1]
#             seq[1] = x
#             j = digits_to_int(seq)
#             if x == 0:
#                 trans[x,j,i] = v[i]  
#             else:
#                 trans[x,j,i] = 1-v[i]
    
#     return trans
            

# # %%
# def ptb_2cycle(p,delta):
#     if p+delta > 1.:
#         raise ValueError("p and delta should not be greater than 1.")
#     T0 = np.array([
#         [p,0.,0.],
#         [0.,0.,0.],
#         [delta,0.,0.]
#     ])

#     T1 = np.array([
#         [0.,1.,0.],
#         [1.-p-delta ,0.,0.],
#         [0.,0.,0.]
#     ])

#     T2 = np.array([
#         [0.,0.,0.],
#         [0.,0.,1.],
#         [0.,0.,0.]
#     ])

#     return np.array([T0,T1,T2])


# def test_sample(p):
#     T0 = np.array([
#         [1.-p, p],
#         [0., 0.],
#     ])
#     T1 = np.array([
#         [0., 0.],
#         [p, 0.],
#     ])
#     T2 = np.array([
#         [0.,0.],
#         [0., 1.-p]
#     ])
#     return np.array([T0, T1, T2])

# #%%
# def gaussian(dx, mean=0, var=1,lim=(-3,3)):
#     f = lambda x: norm.pdf(x,loc = mean, scale=var)
#     eps = 1e-8
#     xlim = arange(lim[0],lim[1]+eps,dx)
#     N = len(xlim)-1
#     transitions = np.zeros((N,1,1))
#     # print(xlim)
#     for i in range(N):
#         # print(quad(f,xlim[i],xlim[i+1]))
#         transitions[i,0,0],_ = quad(f,xlim[i],xlim[i+1])
#     p = sum(transitions)
#     transitions /= p

#     return transitions


# # %%
# def ornstein_uhlenbeck(dx, dt, theta, var=1, lim=(-3, 3)):
#     eps = 1e-8
#     xlim = arange(lim[0], lim[1]+eps, dx)
#     N = len(xlim)-1
#     transitions = np.zeros((N, N, N))

#     for i in range(N):
#         for j in range(N):
#             xmean = (xlim[i] + xlim[i+1])/2.
#             xmean *= exp(-dt*theta)
#             # D = var**2/2.
#             # scale = D/theta * (1-exp(-2.*theta*dt))
#             # scale = sqrt(scale)
#             scale = var
#             f = lambda x: norm.pdf(x, loc = xmean,scale=scale)
#             transitions[j,j,i],_ = quad(f,xlim[j],xlim[j+1])
#         transitions[:, :, i] /= sum(transitions[:, :, i])

        
#     return transitions

# #%%
# def wiener_process(dx, dt, mean=0., var=1., lim=(-3, 3)):
#     eps = 1e-8
#     xlim = arange(lim[0], lim[1]+eps, dx)
#     N = len(xlim)-1
#     transitions = np.zeros((N, N, N))

#     for i in range(N):
#         for j in range(N):
#             xmean = (xlim[i] + xlim[i+1])/2.
#             f = lambda x: norm.pdf(x, loc=xmean, scale=var)
#             transitions[j, j, i], _ = quad(f, xlim[j], xlim[j+1])
#         transitions[:, :, i] /= sum(transitions[:, :, i])
#     return transitions