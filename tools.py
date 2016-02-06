def simple_hash(s, base):
    """Very simple hashing function for string"""
    cumsum= 0
    for i in range(1, len(s)+1):
        cumsum+= base**(len(s)-i) * ord(s[i])
    return cumsum

import cmath
def DFT(x):
    """Computes the discrete Fourier transform of an input sampling"""
    N= len(x)
    X= [0]*N
    for k in range(N):
        Xk= 0
        for n in range(N):
            Xk+= x[n]*cmath.exp(-2j*cmath.pi*k*n/N)
        X[k]= Xk
    return X

def inverseDFT(X):
    """Computes the inverse of the discrete Fourier transform, revealing the
    original sampling"""
    N= len(X)
    x= [0]*N
    for n in range(N):
        xn= 0
        for k in range(N):
            xn+= X[k]*cmath.exp(2j*cmath.pi*k*n/N)
        x[n]= xn/N
    return x
