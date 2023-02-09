import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import *
#import KrausOperatorTraining2 as KOT
import scipy.linalg as LA
#import matplotlib.pyplot as plt


class KrausOperator(nn.Module):
    def __init__(self, alphabet_size,dim,kraus_operator = None):
        super().__init__()
        if kraus_operator is None:
#            print("none")
            self.Kraus_operators = nn.Parameter(torch.rand(alphabet_size,dim,dim,dtype=torch.float64,requires_grad=True))
        else:
            self.Kraus_operators = nn.Parameter(torch.tensor(kraus_operator))

        self.dim = dim
        self.alphabet_size = alphabet_size
        self.normalized = False

    def forward(self,sequence):
        log_p = torch.tensor([0.],dtype=torch.float64)
        rho,rhol,lam = self.leading_eigenvector_value()
        rho = rho/torch.trace(rho.mm(rhol))
        # Normalize the Kraus operators
        Kraus_operators = self.Kraus_operators/torch.sqrt(lam)
        L = len(sequence)
        for i in range(L):
            x = sequence[i]
            rho = Kraus_operators[x].mm(rho).mm(Kraus_operators[x].transpose(0,1))
            if i < L-1:
                px = torch.trace(rho)
                log_p = log_p - torch.log2(px)
            else:
                px = torch.trace(rho.mm(rhol))
                log_p = log_p - torch.log2(px)
#            log_p = log_p - torch.log2(px)
            rho = rho/px


        return log_p

    def leading_eigenvector_value(self):
        rhor = torch.eye(self.dim,dtype=torch.float64,requires_grad = True)/self.dim
        Ite_num = 200
        # Compute the leading right eigen-vector
        for _ in range(Ite_num):
            rhov = torch.zeros(self.dim,self.dim,dtype=torch.float64,requires_grad = True)
            for i in range(self.alphabet_size):
                rhov = rhov + self.Kraus_operators[i].mm(rhor).mm(self.Kraus_operators[i].transpose(0,1))
            mu = torch.trace(rhov)
            rhor = rhov/mu

        # Compute the leading left eigenvector
        rhol = torch.eye(self.dim,dtype=torch.float64,requires_grad = True)/self.dim
        for _ in range(Ite_num):
            rhov = torch.zeros(self.dim,self.dim,dtype=torch.float64,requires_grad = True)
            for i in range(self.alphabet_size):
                rhov = rhov + self.Kraus_operators[i].transpose(0,1).mm(rhol).mm(self.Kraus_operators[i])
            mu = torch.trace(rhov)
#            print(rhov)
            rhol = rhov/mu
        return rhor,rhol,mu


    #  This function generates the unitary operator according to the MPS.
    def renormalizaiton(self):
        # This method doesn't work for MPS that has more than one leading eigenvalue.
        with torch.no_grad():
            rhol = torch.diag(torch.ones(self.dim,dtype=torch.float64))/self.dim
            Ite_num = 200
    #        print(self.Kraus_operators)
            for _ in range(Ite_num):
                rhov = torch.zeros(self.dim,self.dim,dtype=torch.float64)
                for i in range(self.alphabet_size):
                    rhov += self.Kraus_operators[i].transpose(0,1).mm(rhol).mm(self.Kraus_operators[i])
                mu = torch.trace(rhov).item()
    #            print(rhov)
                rhol = rhov/mu
            M = LA.sqrtm(rhol.numpy())
            M = torch.tensor(M)
            for i in range(self.alphabet_size):
                self.Kraus_operators[i] = M.mm(self.Kraus_operators[i]).mm(M.inverse())/sqrt(mu)
        self.normalized = True


    def ini_vec(self):
        with torch.no_grad():
            sigma0 = torch.rand(self.dim,1,dtype= torch.float64)
            sigma0 = sigma0/torch.norm(sigma0)
            Ite_num = 200
            for _ in range(Ite_num):
                sigma0 = self.Kraus_operators[0].mm(sigma0)
                # print(sigma0)
                mu = torch.norm(sigma0)
                sigma0 = sigma0/mu
            self.sigma0 = sigma0

        return self.sigma0


    # This functions generate the logarithm of probability conditions on a certain past 
    # input: a past and a future sequence
    # output: The logarithm of conditional probability of the future sequence
    def log_cond_prob(self,past,seq):
        if not self.normalized:
            self.renormalizaiton()
        compressed_q_state = self.ini_vec()
        for x in past:
            compressed_q_state = self.Kraus_operators[x].mm(compressed_q_state)
            compressed_q_state /= torch.norm(compressed_q_state)
        
        log_p = torch.tensor(0.,dtype = torch.float64)
        for x in seq:
            compressed_q_state = self.Kraus_operators[x].mm(compressed_q_state)
            p = torch.norm(compressed_q_state)**2
            # print(p)
            log_p += torch.log2(p)
            compressed_q_state /= torch.sqrt(p)
        return log_p

        



# define the Kraus operator for complex values.

# This is used for complex matrix multiplication
class cMatrix():
    def __init__(self,matrix_re,matrix_im):
        self.re = matrix_re
        self.im = matrix_im

    def mm(self,cmatrix):
        matrix_re = self.re.mm(cmatrix.re)-self.im.mm(cmatrix.im)
        matrix_im = self.re.mm(cmatrix.im)+self.im.mm(cmatrix.re)

        return cMatrix(matrix_re,matrix_im)

    def __truediv__(self,re_num):

        return cMatrix(self.re/re_num,self.im/re_num)

    def __add__(self, cmatrix):
        return cMatrix(self.re+cmatrix.re,self.im+cmatrix.im)

    def transpose(self):
        return cMatrix(self.re.transpose(0,1),self.im.transpose(0,1))

    def conjugate(self):
        return cMatrix(self.re,-self.im)


    def __repr__(self):
        return "Real:"+str(self.re)+"\n"+"Image:"+str(self.im)
    
    def to_numpy(self):
        return self.re.detach().numpy() + 1.j*self.im.detach().numpy()

    def norm_l2(self):
        X = self.re.transpose(0,1).mm(self.re) + self.im.transpose(0,1).mm(self.im)

        return torch.sqrt(torch.trace(X))


class cKrausOperator(nn.Module):
    def __init__(self,alphabet_size,dim,kraus_operator = None):
        super(cKrausOperator,self).__init__()
        if kraus_operator is None:
            self.Kraus_op_re = nn.Parameter(torch.rand(alphabet_size,dim,dim))
            self.Kraus_op_im = nn.Parameter(torch.rand(alphabet_size,dim,dim))
            self.ckra_op = [cMatrix(self.Kraus_op_re[i],self.Kraus_op_im[i]) for i in range(alphabet_size)]
        else:
            self.Kraus_op_re = nn.Parameter(torch.stack([kraus_operator[i].re for i in range(alphabet_size)]))
            self.Kraus_op_im = nn.Parameter(torch.stack([kraus_operator[i].im for i in range(alphabet_size)]))
            self.ckra_op = kraus_operator

        self.dim = dim
        self.alphabet_size = alphabet_size
        self.normalized = False


    def leading_eig(self):
        rhor = cMatrix(torch.eye(self.dim,requires_grad=True)/self.dim,torch.zeros(self.dim,self.dim,requires_grad=True))

        ite_num = 200
        for _ in range(ite_num):
            matrix_re = torch.zeros(self.dim,self.dim,requires_grad=True)
            matrix_im = torch.zeros(self.dim,self.dim,requires_grad=True)
            rhov = cMatrix(matrix_re,matrix_im)
#             self.ckra_op[0].dot(rhor)
            for x in range(self.alphabet_size):
#                 print(rhor,self.ckra_op[x].transpose().conjugate())
                rhov = rhov + self.ckra_op[x].mm(rhor).mm(self.ckra_op[x].transpose().conjugate())

            mu = torch.trace(rhov.re)
            rhor = rhov/mu

        rhol = cMatrix(torch.eye(self.dim,requires_grad=True),torch.zeros(self.dim,self.dim,requires_grad=True))
        for _ in range(ite_num):
            matrix_re = torch.zeros(self.dim,self.dim,requires_grad=True)
            matrix_im = torch.zeros(self.dim,self.dim,requires_grad=True)
            rhov = cMatrix(matrix_re,matrix_im)
#             self.ckra_op[0].dot(rhor)
            for x in range(self.alphabet_size):
#                 print(rhor,self.ckra_op[x].transpose().conjugate())
                rhov = rhov + self.ckra_op[x].transpose().conjugate().mm(rhol).mm(self.ckra_op[x])

            mu = torch.trace(rhov.re)
            rhol = rhov/mu
        return rhor,rhol,mu

    def forward(self,sequence):
        log_p = torch.tensor([0.],requires_grad = True)
        rho,rhol,lam = self.leading_eig()
        rho = rho/torch.trace(rho.mm(rhol).re)
#        rhol2 = rhol
        # Normalize the Kraus operators
#        print(lam)
        Kraus_operators = [self.ckra_op[i]/torch.sqrt(lam) for i in range(self.alphabet_size)]
        L = len(sequence)
        for i in range(L):
            x = sequence[i]
            rho = Kraus_operators[x].mm(rho).mm(Kraus_operators[x].transpose().conjugate())
            if i < L-1:
                px = torch.trace(rho.re)
                log_p = log_p - torch.log2(px)
            else:
              #  print(rhol)
#                 rho = rho.mm(rhol)
                px = torch.trace(rho.mm(rhol).re)
                log_p = log_p - torch.log2(px)
#            log_p = log_p - torch.log2(px)
            rho = rho/px


        return log_p

    def renormalization(self):
        rho,rhol,lam = self.leading_eig()
        ### Convert the matrix to numpy
        rhol_np = rhol.to_numpy()
        M = LA.sqrtm(rhol_np) ### The sqrtm uses Shur decomposition which works here for rhol is Hermision and M is also Hermitian
        M_inv = LA.inv(M)
        KraOp = [M.dot(self.ckra_op[i].to_numpy()).dot(M_inv) for i in range(self.alphabet_size)]
        KraOp /= sqrt(lam.detach().numpy())
        self.Kraus_op_re = nn.Parameter(torch.Tensor(real(KraOp)))
        self.Kraus_op_im = nn.Parameter(torch.Tensor(imag(KraOp)))
        # Tensor will convert the data type to torch.float32
        self.ckra_op = [cMatrix(self.Kraus_op_re[i], self.Kraus_op_im[i]) for i in range(self.alphabet_size)]
        self.normalized = True
        
        pass

    def ini_vec(self):
        
        sigma0 = cMatrix(torch.rand(self.dim, 1), torch.rand(self.dim, 1))
        sigma0 = sigma0/sigma0.norm_l2()
        Ite_num = 200
        for _ in range(Ite_num):
            sigma0 = self.ckra_op[0].mm(sigma0)
            # print(sigma0)
            mu = sigma0.norm_l2()
            sigma0 = sigma0/mu
        self.sigma0 = sigma0

        return self.sigma0

    def log_cond_prob(self, past, seq):
        if not self.normalized:
            self.renormalization()
        compressed_q_state = self.ini_vec()
        for x in past:
            compressed_q_state = self.ckra_op[x].mm(compressed_q_state)
            compressed_q_state /= compressed_q_state.norm_l2()

        log_p = torch.tensor(0.)
        for x in seq:
            compressed_q_state = self.ckra_op[x].mm(compressed_q_state)
            p = compressed_q_state.norm_l2()**2
            # print(p)
            log_p += torch.log2(p)
            compressed_q_state /= torch.sqrt(p)
        return log_p


   

