import numpy as np
import galsim
from scipy.special import binom
from helper import *
from moments import *

## Metacalibration routines

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
        i,j = singlet_to_doublet(kdw)

        if nw>=2:
            ## 2nd order corrections
            denom = sigma**2
            DW[kdw, doublet_to_singlet(i+2,j+0)] = 0.5*c1/denom
            DW[kdw, doublet_to_singlet(i+1,j+1)] = -2.*e2/denom
            DW[kdw, doublet_to_singlet(i+0,j+2)] = 0.5*c2/denom

        if nw>=4:
            ## 4th order corrections
            denom = sigma**4
            DW[kdw, doublet_to_singlet(i+4,j+0)] = 0.125*c1**2/denom
            DW[kdw, doublet_to_singlet(i+3,j+1)] = -c1*e2/denom
            DW[kdw, doublet_to_singlet(i+2,j+2)] = 0.25*(c1*c2 + 8*e2**2)/denom
            DW[kdw, doublet_to_singlet(i+1,j+3)] = -c2*e2/denom
            DW[kdw, doublet_to_singlet(i+0,j+4)] = 0.125*c2**2/denom

        if nw>=6:
            ## 6th order corrections
            denom = 48.*sigma**6
            DW[kdw, doublet_to_singlet(i+6,j+0)] = c1**3/denom
            DW[kdw, doublet_to_singlet(i+5,j+1)] = -12.*(c1**2)*e2/denom
            DW[kdw, doublet_to_singlet(i+4,j+2)] = (3.*c2*c1**2 + 48.*c1*e2**2)/denom
            DW[kdw, doublet_to_singlet(i+3,j+3)] = -(24.*c1*c2*e2 + 64.*e2**3)/denom
            DW[kdw, doublet_to_singlet(i+2,j+4)] = (3.*c1*c2**2 + 48.*c2*e2**2)/denom
            DW[kdw, doublet_to_singlet(i+1,j+5)] = -12.*e2*c2**2/denom
            DW[kdw, doublet_to_singlet(i+0,j+6)] = c2**3/denom

    return DW

def psf_correction(image_moments, psf_moments, matrix_inv=False):
    """ Given the PSF-convolved deweighted galaxy moments and PSF moments, calculate the moments of the intrinsic galaxy
    return intrinsic_moments
    """

    if matrix_inv:
        print "Doing matrix inversion"
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
        print "NOT doing matrix inversion"

        psf00 = psf_moments[doublet_to_singlet(0,0)]

        ## 0-th order moments
        gal_moments = image_moments[:6] / psf00

        ## 1-st order moments
        ## Assumes (0,0) --> 0
        gal_moments[1:] -= psf_moments[1:]*gal_moments[0]/psf00

        ## 2-nd order moments
        gal_moments[doublet_to_singlet(0,2)] -= 2.*psf_moments[doublet_to_singlet(0,1)]*gal_moments[doublet_to_singlet(0,1)]/psf00
        gal_moments[doublet_to_singlet(1,1)] -= (psf_moments[doublet_to_singlet(1,0)]*gal_moments[doublet_to_singlet(0,1)] + psf_moments[doublet_to_singlet(0,1)]*gal_moments[doublet_to_singlet(1,0)])/psf00
        gal_moments[doublet_to_singlet(2,0)] -= 2.*psf_moments[doublet_to_singlet(1,0)]*gal_moments[doublet_to_singlet(1,0)]/psf00

    return gal_moments

def deimos(gal_img, psf_img, nw=6, scale=1., psf_scale=None, round_moments=False, matrix_inv=False):
    if not isinstance(gal_img, galsim.Image):
        gal_img = galsim.Image(gal_img)

    gauss_moments = gal_img.FindAdaptiveMom(round_moments=round_moments, strict=False)
    if gauss_moments.moments_status!=0:
        return -99,-99

    centroid = [gauss_moments.moments_centroid.x - gal_img.center.x, gauss_moments.moments_centroid.y - gal_img.center.y]

    if psf_scale is None:
        ## Set psf_scale = scale, same as the galaxy scale
        psf_scale = scale

    psf_grid = generate_pixelgrid(centroid, psf_img.array.shape, scale=psf_scale)
    gal_grid = generate_pixelgrid(centroid, gal_img.array.shape, scale=scale)
   
    ## If round_moments is True, we need a circular weight function. The observed_shape cannot be used as it contains the ellipticity of the object.
    ## Hence, the ellipticity has to be explicitly set to zero.
    weight = get_weight_image(gal_grid, gauss_moments.moments_sigma, gauss_moments.observed_shape.g1*(not round_moments), gauss_moments.observed_shape.g2*(not round_moments) )
    DW = generate_deweighting_matrix(gauss_moments.moments_sigma, gauss_moments.observed_shape.g1*(not round_moments), gauss_moments.observed_shape.g2*(not round_moments), nw=nw)
    
    kmax = (nw+3)*(nw+4)/2
    weighted_img_moments, psf_moments = np.zeros(kmax), np.zeros(6)
    for k in xrange(kmax):
        m,n = singlet_to_doublet(k)
        if k<6:
            psf_moments[k] = measure_moments(m,n,psf_img,psf_grid,1.)
        weighted_img_moments[k] = measure_moments(m,n,gal_img,gal_grid,weight)

    deweighted_img_moments = np.dot(DW, weighted_img_moments)
    psf_corrected_moments = psf_correction(deweighted_img_moments, psf_moments, matrix_inv=matrix_inv)

    ellipticity = moments_to_ellipticity(psf_corrected_moments)

    return ellipticity
