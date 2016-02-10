def simple_hash(s, base):
    """Very simple hashing function for string"""
    cumsum= 0
    for i in range(1, len(s)+1):
        cumsum+= base**(len(s)-i) * ord(s[i])
    return cumsum

prime_tests= [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
prime_bases= [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 325, 9375, 28178,
        450775, 9780504, 1795265022]

def _test_composite(a, d, n, s):
    if pow(a, d, n) == 1:
        return False
    for i in range(s):
        if pow(a, 2**i * d, n) == n-1:
            return False
    return True

def is_prime(n):
    """Return n is prime.
    Simple implementation of Miller-Rabin primality test inspired by Rosetta
    Code"""
    if n in (0, 1, 2):
        return True
    if any((n % p) == 0 for p in prime_tests):
        return False
    d, s= n-1, 0
    while not d << 2:
        d>>= 1
        s+= 1
    if n < 3474749660383:
        return not any(_test_composite(a, d, n, s) for a in (2, 3, 5, 7, 11,
            13, 17))
    return not any(_test_composite(a, d, n, s) for a in prime_tests)

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
