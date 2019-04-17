import numpy as np
import galsim
from scipy.special import binom

## Metacalibration routines

def shear_matrix(g1, g2, kappa=None):
    """ Return the 2x2 shear operator

    If kappa is None, the determinant of the shear operator is set to 1.
    If kappa is a real number, the determinant of the shear operator is (1-kappa)(1-|g|)
    """
    A = np.array([[1.-g1, -g2],[-g2, 1.+g1]])
    if kappa is None:
        norm = 1./(1.-np.sqrt(g1**2+g2**2))
    else:
        norm = (1.-kappa)
    A *= norm
    return A

def generate_pixelgrid(centroid, size, scale=1.0):
    x = np.arange(-0.5*size[0]-centroid[0],0.5*size[0]-centroid[0]+0.1*scale,scale)
    y = np.arange(-0.5*size[1]-centroid[1],0.5*size[1]-centroid[1]+0.1*scale,scale)
    X,Y = np.meshgrid(x,y)
    return X,Y

def shear_pixelgrid(grid, A):
    """ Shear the pixel grid to generate an affine grid
    """
    X, Y = grid
    x, y = X.flatten(), Y.flatten()
    xy = np.stack([x,y])
    assert xy.shape==(2,X.size)
    sheared_xy = np.dot(A,xy)
    x, y = sheared_xy[0], sheared_xy[1]
    sheared_grid = (np.reshape(x,X.shape),np.reshape(y.Y.shape))
    return sheared_grid

def shear_monomial(m, n, sheared_grid):
    X, Y = sheared_grid
    monomial = (X**m)*(Y**n)
    return monomial

def shear_gaussian():
    return sheared_gaussian

## DEIMOS-specific

def generate_deweighting_matrix(sigma, e1, e2, nw=6):
    """ Construct the deweighting matrix given the Gaussian weight parameters, up to order nw.
    Admissible values of nw are: 0, 2, 4, 6
    """

    if not nw in [0,2,4,6]:
        raise ValueError('Cannot construct deweighting matrix to order %d' % (nw))

    c1 = (1.-e1)**2 + e2**2
    c2 = (1.+e1)**2 + e2**2

    kmax = (nw+3)*(nw+4)/2
    DW = np.eye(6,kmax)

    for kdw in xrange(6):
        i,j = single_to_doublet(kdw)

        if nw>=2:
            ## 2nd order
            DW[kdw, doublet_to_singlet(i+2,j)] = 0.5*c1/sigma**2
            DW[kdw, doublet_to_singlet(i+1,j+1)] = -2.*e2/sigma**2
            DW[kdw, doublet_to_singlet(i,j+2)] = 0.5*c2/sigma**2

        if nw>=4:
            ## 4th order
            DW[kdw, doublet_to_singlet(i+4,j)] = 0.125*c1**2/sigma**4
            DW[kdw, doublet_to_singlet(i+3,j+1)] = -c1*e2/sigma**4 ## check this
            DW[kdw, doublet_to_singlet(i+2,j+2)] = 0.25*(c1*c2 + 8*e2**2)/sigma**4
            DW[kdw, doublet_to_singlet(i+1,j+3)] = c2*e2/sigma**4  ## check this
            DW[kdw, doublet_to_singlet(i,j+4)] = 0.125*c2**2/sigma**4

        if nw>=6:
            ## 6th order
            DW[kdw, doublet_to_singlet(i+6,j)] = c1**3/(48.*sigma**6)
            DW[kdw, doublet_to_singlet(i+5,j+1)] = -0.25*(c1**2)*e2/sigma**6
            DW[kdw, doublet_to_singlet(i+4,j+2)] = 0.0625*(c2*c1**2 + c1*e2**2)/sigma**6
            DW[kdw, doublet_to_singlet(i+3,j+3)] = -(3.*c1*c2*e2 + 8.*e2**3)/sigma**6
            DW[kdw, doublet_to_singlet(i+2,j+4)] = 0.0625*(c1*c2**2 + c2*e2**2)/sigma**6
            DW[kdw, doublet_to_singlet(i+1,j+5)] = -0.25*e2*c2**3/sigma**6
            DW[kdw, doublet_to_singlet(i,j+6)] = c2**3/(48.*sigma**6)

    return DW

def measure_moments(m, n, image, grid, gauss_sigma=0., gauss_e1=0., gauss_e2=0., A=None):
    """ Measure the moments of an image on the pixel grid, with an optional elliptical Gaussian weight function.
        If gauss_sigma = 0, then unweighted moments are returned (for PSF)
        if A is not None, then the grid and the weight (if defined) are sheared before computing the moments
    """

    return Qmn

def psf_correction(image_moments, psf_moments, matrix_inv=False):
    """ Given the PSF-convolved deweighted galaxy moments and PSF moments, calculate the moments of the intrinsic galaxy
    return intrinsic_moments
    """

    if matrix_inv:
        ## Most generic and elegant way involving matrix inversion
        P = np.eye(6)

        for k1 in xrange(6):
            for k2 in xrange(6):
                i,j = singlet_to_doublet(k1)
                k,l = singlet_to_doublet(k2)

                if (i>=k)&(j>=l):
                    P[k1,k2] = binom(i,k)*binom(j,l)*psf_moments[doublet_to_singlet(i-k, j-l)]

        Pinv = np.linalg.inv(P)

        gal_moments = np.dot(Pinv, image_moments)

    else:
        ## Rudimentary implementation of Table 1 of Melchior et al (2012)

        psf00 = psf_moments[doublet_to_singlet(0,0)]

        ## 0-th order moments
        gal_moments = image_moments / psf00

        ## 1-st order moments
        ## Assumes (0,0) --> 0
        gal_moments[1:] -= psf_moments[1:]*gal_moments[0]/psf00

        ## 2-nd order moments
        gal_moments[doublet_to_singlet(0,2)] -= 2.*psf_moments[doublet_to_singlet(0,1)]*gal_moments[doublet_to_singlet(0,1)]/psf00
        gal_moments[doublet_to_singlet(1,1)] -= (psf_moments[doublet_to_singlet(1,0)]*gal_moments[doublet_to_singlet(0,1)] + psf_moments[doublet_to_singlet(0,1)]*gal_moments[doublet_to_singlet(1,0)])/psf00
        gal_moments[doublet_to_singlet(2,0)] -= 2.*psf_moments[doublet_to_singlet(1,0)]*gal_moments[doublet_to_singlet(1,0)]/psf00

    return gal_moments

def moments_to_ellipticity(moment_vector, etype='epsilon'):
    """ Calculate the ellipticity given the moments

        @param moment_vector     a numpy array containing the moments
        @param etype             must be one of 'linear', 'flux_norm', 'chi' or 'epsilon' (default)

        @returns ellipticity tuple, defined by etype
    """

    if etype=='epsilon':
        denom = moment_vector[doublet_to_singlet(0,2)] + moment_vector[doublet_to_singlet(2,0)]
        denom += 2*np.sqrt( moment_vector[doublet_to_singlet(0,2)]*moment_vector[doublet_to_singlet(2,0)] - moment_vector[doublet_to_singlet(1,1)]**2 )
    elif etype=='chi':
        denom = moment_vector[doublet_to_singlet(0,2)] + moment_vector[doublet_to_singlet(2,0)]
    elif etype=='linear':
        denom = 1.
    elif etype=='flux_norm':
        denom = moment_vector[doublet_to_singlet(0,0)]
    else:
        raise ValueError(" etype must be one of 'linear', 'chi' or 'epsilon' ")

    e1 = (moment_vector[doublet_to_singlet(2,0)] - moment_vector[doublet_to_singlet(0,2)])/denom
    e2 = 2.*moment_vector[doublet_to_singlet(1,1)]/denom

    return e1, e2

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
