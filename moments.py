## Script implementing the moment calculations
## Author: Arun Kannawadi

import numpy as np
import galsim
import os, sys
sys.path.append('/disks/shear15/arunkannawadi/Moments_Metacal/mydeimos')
from helper import *

def measure_moments(m, n, image, grid, weight, A=None):
    """ Measure the moments of an image on the pixel grid, with an optional elliptical Gaussian weight function.
        If gauss_sigma = 0, then unweighted moments are returned (for PSF)
        if A is not None, then the grid and the weight (if defined) are sheared before computing the moments
    """

    X, Y = grid
    if isinstance(image, galsim.Image):
        image = image.array

    ## If the grid is rectangular and A is None, then it's faster to raise the 1-D arrays to m/n power and grid them
    if A is None:
        x, y = X[0], Y[:,0]

        np.testing.assert_array_equal(X[0],X[-1])
        np.testing.assert_array_equal(Y[:,0],Y[:,-1])

        xm, yn = x**m, y**n
        XmYn = np.outer(yn,xm)
    else:
        Xm, Yn = X**m, Y**n
        XmYn = Xm*Yn

    Qmn = np.sum( (XmYn)*weight*image )
    return Qmn

def moments_to_size(moment_vector, size_type='det', flux_normalized=True):
    """ Calculate the size given the moments

        @param moment_vector    a numpy array containing the moments
        @param size_type        must be one of 'trace' or 'det' (default)

        @returns the size defined by size_type
    """

    if not size_type in ['trace','det']:
        raise ValueError("'size_type' must be 'trace' or 'det'")

    if size_type=='det':
        size = ( moment_vector[doublet_to_singlet(2,0)]*moment_vector[doublet_to_singlet(0,2)] - moment_vector[doublet_to_singlet(1,1)]**2 )**0.25
    else:
        size = (moment_vector[doublet_to_singlet(2,0)]+moment_vector[doublet_to_singlet(0,2)])

    if flux_normalized:
        if size_type=='det':
            size /= np.sqrt(moment_vector[doublet_to_singlet(0,0)])
        else:
            size /= moment_vector[doublet_to_singlet(0,0)]
    return size

def moments_to_ellipticity(moment_vector, etype='epsilon'):
    """ Calculate the ellipticity given the moments

        @param moment_vector     a numpy array containing the moments
        @param etype             must be one of 'linear', 'flux_norm', 'chi' or 'epsilon' (default)

        @returns ellipticity tuple, defined by etype
    """

    if etype=='epsilon':
        denom = moments_to_size(moment_vector, size_type='trace', flux_normalized=False)
        #denom = moment_vector[doublet_to_singlet(0,2)] + moment_vector[doublet_to_singlet(2,0)]
        #denom += 2*np.sqrt( moment_vector[doublet_to_singlet(0,2)]*moment_vector[doublet_to_singlet(2,0)] - moment_vector[doublet_to_singlet(1,1)]**2 )
        denom += 2.*(moments_to_size(moment_vector,size_type='det', flux_normalized=False))**2
    elif etype=='chi':
        #denom = moment_vector[doublet_to_singlet(0,2)] + moment_vector[doublet_to_singlet(2,0)]
        denom = moments_to_size(moment_vector, size_type='trace', flux_normalized=False)
    elif etype=='linear':
        denom = 1.
    elif etype=='flux_norm':
        denom = moment_vector[doublet_to_singlet(0,0)]
    else:
        raise ValueError(" etype must be one of 'linear', 'chi' or 'epsilon' ")

    e1 = (moment_vector[doublet_to_singlet(2,0)] - moment_vector[doublet_to_singlet(0,2)])/denom
    e2 = 2.*moment_vector[doublet_to_singlet(1,1)]/denom

    return e1, e2

def moments_to_rho4(moment_vector):
    """ Calculate the radial fourth moment

        @param moment_vector    a numpy array containing the moments

        @returns                radial fourth moment (rho4)
    """

    try:
        rho4 = (moment_vector[doublet_to_singlet(4,0)]+moment_vector[doublet_to_singlet(0,4)]+2.*moment_vector[doublet_to_singlet(2,2)])/moment_vector[doublet_to_singlet(0,0)]
        return rho4
    except IndexError:
        max_index = max(doublet_to_singlet(4,0), doublet_to_singlet(2,2), doublet_to_singlet(0,4))
        print "Up to %d moments are needed to compute the radial fourth moment" % (max_index)
        return -1
