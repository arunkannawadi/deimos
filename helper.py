## Helper script defining some convenience functions
## Author: Arun Kannawadi

import numpy as np
import os, sys
import math
sys.path.append('/disks/shear15/arunkannawadi/Moments_Metacal/mydeimos')
from metacal import shear_pixelgrid

def shear_matrix(g1, g2, kappa=None):
    """ Return the 2x2 shear operator

    If kappa is None, the determinant of the shear operator is set to 1.
    If kappa is a real number, the determinant of the shear operator is (1-kappa)(1-|g|)
    """
    A = np.array([[1.-g1, -g2],[-g2, 1.+g1]])
    if kappa is None:
        norm = 1./np.sqrt(1.-(g1**2+g2**2))
    else:
        norm = (1.-kappa)
    A *= norm
    return A

def generate_pixelgrid(centroid, size, scale=1.0):
    x = np.arange(-0.5*size[0]-centroid[0],0.5*size[0]-centroid[0]-0.1,1.)*scale
    y = np.arange(-0.5*size[1]-centroid[1],0.5*size[1]-centroid[1]-0.1,1.)*scale
    X,Y = np.meshgrid(x,y)
    return X,Y

def get_weight_image(grid, gauss_sigma=1., gauss_e1=0., gauss_e2=0., A=None):
    """ Generate
    """

    if gauss_e2==0. and A is None:
        X, Y = grid
        x, y = X[0], Y[:,0]

        np.testing.assert_array_equal(X[0],X[-1])
        np.testing.assert_array_equal(Y[:,0],Y[:,-1])

        q = ((1.-gauss_e1)/(1.+gauss_e1))
        sx = gauss_sigma/np.sqrt(q)
        sy = gauss_sigma*np.sqrt(q)

        weight = np.outer(np.exp(-0.5*y**2/sy**2), np.exp(-0.5*x**2/sx**2))

    else:
        E = shear_matrix(gauss_e1, gauss_e2, kappa=None)
        sheared_grid = shear_pixelgrid(grid, E)
        X, Y = sheared_grid

        weight = np.exp(-0.5*(X**2+Y**2)/gauss_sigma**2)

    return weight

## Helper routines
def doublet_to_singlet(i,j):
    """ Convert an index (i,j) to a single number for accessing
    """

    if (i<0)|(j<0):
        raise ValueError(" The doublet indices must be non-negative integers ")

    if (i!=int(i))|(j!=int(j)):
        raise TypeError(" The doublet indices must be integers (non-negative) ")
    
    if not (isinstance(i,int) & isinstance(j, int)):
        import warnings
        warnings.warn(" The doublet indices should be of type int ")
        i, j = int(i), int(j)

    n = i+j

    k = n*(n+1)/2
    k += i

    assert k>=0 

    return k

def singlet_to_doublet(k):
    """ Convert an accessing index k to the doublet index (i,j)
    """

    if k<0:
        raise ValueError(" The singlet index must be non-negative (and integral) ")

    if (k!=int(k)):
        raise TypeError(" The singlet index must be an integer (and non-negative) ")

    if not isinstance(k,int):
        import warnings
        warnings.warn(" The singlet index should be of type int ")
        k = int(k)

    ## Find the smallest non-negative integer n such that
    ## 1+2+...+(n+1) = (n+1)(n+2)/2 >= k+1
    n = int(np.ceil(0.5*(-3+np.sqrt(9.+8.*k))))
    i = k - n*(n+1)/2
    j = n - i

    assert i>=0
    assert j>=0

    return i, j

def get_conversion_dicts(nmax=8):
    """ Pre-compute the conversions between singlet and doublet indices
    """

    if nmax<0:
        raise ValueError(" The maximum order must be non-negative (and integral) ")

    if (nmax!=int(nmax)):
        raise TypeError(" The maximum order must be an integer (and non-negative) ")

    if not isinstance(nmax,int):
        import warnings
        warnings.warn(" The maximum order should be of type int ")
        nmax = int(nmax)

    kmax = (nmax+1)*(nmax+2)/2

    d2s, s2d = {}, {}
    
    for k in xrange(kmax):
        i,j = singlet_to_double(k)
        s2d[k] = (i,j)
        d2s[(i,j)] = k

    return d2s, s2d

def factorial(n):
    ## Pre-compute the frequently encountered factorials, although it hardly makes the overall code faster.
    fact = [1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,6227020800,87178291200,1307674368000,20922789888000,355687428096000,6402373705728000,121645100408832000,2432902008176640000]
    try:
        fn = fact[n]
    except IndexError:
        fn = math.factorial(n)

    return fn
