## Main script implementing the DEIMOS method
## Author: Arun Kannawadi

import numpy as np
import galsim
from scipy.special import binom
import os, sys
sys.path.append('/disks/shear15/arunkannawadi/Moments_Metacal/mydeimos/')
from helper import *
from moments import *

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
        gal_moments = image_moments[:6] / psf00

        ## 1-st order moments
        ## Assumes (0,0) --> 0
        gal_moments[1:] -= psf_moments[1:]*gal_moments[0]/psf00

        ## 2-nd order moments
        gal_moments[doublet_to_singlet(0,2)] -= 2.*psf_moments[doublet_to_singlet(0,1)]*gal_moments[doublet_to_singlet(0,1)]/psf00
        gal_moments[doublet_to_singlet(1,1)] -= (psf_moments[doublet_to_singlet(1,0)]*gal_moments[doublet_to_singlet(0,1)] + psf_moments[doublet_to_singlet(0,1)]*gal_moments[doublet_to_singlet(1,0)])/psf00
        gal_moments[doublet_to_singlet(2,0)] -= 2.*psf_moments[doublet_to_singlet(1,0)]*gal_moments[doublet_to_singlet(1,0)]/psf00

    return gal_moments

def deimos(gal_img, psf_img, nw=6, scale=1., psf_scale=None, etype='chi', w_sigma=None, w_sigma_scalefactor=1., psf_w_sigma=None, round_moments=False, hsmparams=None, matrix_inv=False):
    """ Calculate the ellipticity of the galaxy using the DEIMOS method (Melchior et al., 2012) given the galaxy and PSF postage stamps.

        @params gal_img         Postage stamp of the PSF-convolved galaxy. This can either be a galsim.Image instance or a NumPy array.
        @params psf_img         Postage stamp of the PSF. This can either be a galsim.Image instance or a NumPy array.

        @params nw                      The maximum order of correction terms to include. This can be an integer or a list of admissible values.
                                        The admissible values are 0, 2, 4, 6. [Default: 6]
        @params scale                   The pixel scale of gal_img. [Default: 1]
        @params psf_scale               The pixel scale of psf_img. If set to None, this will take the value of 'scale'. [Default: 1]
        @params etype                   The type of ellipticity to return. The admissible options are 'chi', 'epsilon', 'linear' (BJ02) and
                                        'flux_norm' (similar to 'linear' but normalised by the flux). The 'epsilon' type ellipticities fail if the determinant
                                        of the deweighted second moment matrix is negative. [Default: 'chi']
        @params w_sigma                 The width of the (circularised) Gaussian weight function, in units of pixels, to calculate the weighted moments of the galaxy.
                                        The moments_sigma value from galsim.ShapeData can be passed on without worrying about the pixel scale.
                                        If set to a non-positive value or None, adaptive weights are computed internally. [Default: None]
        @params w_sigma_scalefactor     The number by which the w_sigma is to be scaled. [Default: 1]
        @params psf_w_sigma             The width of the (circularised) Gaussian weight function, in units of pixels, to calculate the weighted moments of the PSF.
                                        If set to a negative value, unweighted moments are used. If set to 0, adaptive weights are computed internally.
                                        If set to None, the same width used for the galaxy image (along with w_sigma_scalefactor) is used (recommended). [Default: None]
        @params round_moments           Setting this true will force the Gaussian to be circular (and slightlu faster computations). [Default: False]
        @params hsmparams               A galsim.hsm.HSMParams instance to change the default settings in the calculation of adaptive moments.
                                        If None, default values are used. [Default: None]
        @params matrix_inv              Whether to incorportate PSF-correction as a matrix operation or not. This choice has little impact on the results. [Default: False]

        @returns                        A tuple or a list of tuples (depending on the type of 'nw') containing ellipticities of the type 'etype' and a status flag.
                                        0 indicates reliable measurement, -1 indicates failure of adaptive moments routine. -2 indicates that the determinant of the
                                        deweighted quadrupole moments is negative and -3 indicates that at least one of the deweighted (2,0) or (0,2) moments is negative.
    """

    if not isinstance(gal_img, galsim.Image):
        gal_img = galsim.Image(gal_img)
    if not isinstance(psf_img, galsim.Image):
        psf_img = galsim.Image(psf_img)

    ## Decide if psf_hsm is required in the first place or not. We calculate it nevertheless
    calculate_psf_hsm = True if ((psf_w_sigma is None or psf_w_sigma>=0) and not round_moments) else False

    psf_hsm = psf_img.FindAdaptiveMom(round_moments=round_moments, strict=False, hsmparams=hsmparams)
    gal_hsm = gal_img.FindAdaptiveMom(round_moments=round_moments, strict=False, hsmparams=hsmparams)

    if ((gal_hsm.moments_status!=0) or ((psf_hsm.moments_status!=0) and calculate_psf_hsm)):
        len_nw = len(nw) if hasattr(nw,'__iter__') else 1
        return [(-99,-99,-1)]*len_nw

    ## Centroid calculation
    gal_centroid = [gal_hsm.moments_centroid.x - gal_img.center.x, gal_hsm.moments_centroid.y - gal_img.center.y]
    if calculate_psf_hsm:
        psf_centroid = [psf_hsm.moments_centroid.x - psf_img.center.x, psf_hsm.moments_centroid.y - psf_img.center.y]
    else:
        psf_centroid = [-0.5,-0.5]

    gal_scale = scale
    if psf_scale is None:
        ## Set psf_scale = scale, same as the galaxy scale
        psf_scale = gal_scale

    psf_grid = generate_pixelgrid(psf_centroid, psf_img.array.shape, scale=psf_scale)
    gal_grid = generate_pixelgrid(gal_centroid, gal_img.array.shape, scale=gal_scale)
   
    ## If round_moments is True, we need a circular weight function. The observed_shape cannot be used as it contains the ellipticity of the object.
    ## Hence, the ellipticity has to be explicitly set to zero.
    if round_moments:
        gal_weight_g1, gal_weight_g2 = 0., 0.
        psf_weight_g1, psf_weight_g2 = 0., 0.
    else:
        gal_weight_g1, gal_weight_g2 = gal_hsm.observed_shape.g1, gal_hsm.observed_shape.g2
        psf_weight_g1, psf_weight_g2 = psf_hsm.observed_shape.g1, psf_hsm.observed_shape.g2
    if w_sigma is None or w_sigma<=0.:
        gal_weight_sigma = gal_hsm.moments_sigma
    else:
        gal_weight_sigma = w_sigma

    gal_weight_sigma *= w_sigma_scalefactor

    if psf_w_sigma is None:
        psf_weight_sigma = gal_weight_sigma
    elif psf_w_sigma==0:
        psf_weight_sigma = psf_hsm.moments_sigma

    gal_weight = get_weight_image( gal_grid, gal_weight_sigma, gal_weight_g1, gal_weight_g2 )
    if psf_w_sigma<0.:
        psf_weight = 1.
    else:
        psf_weight = get_weight_image( psf_grid, psf_weight_sigma, psf_weight_g2, psf_weight_g2 )
    
    if hasattr(nw,'__iter__'):
        nw_list = nw
    else:
        nw_list = [ nw ]

    nw_max = max(nw_list)
    gal_DW = generate_deweighting_matrix( gal_weight_sigma, gal_weight_g1, gal_weight_g2, nw=nw_max)
    if psf_w_sigma<0.:
        psf_DW = np.eye(6,gal_DW.shape[1])
    else:
        psf_DW = generate_deweighting_matrix( psf_weight_sigma, psf_weight_g1, psf_weight_g2, nw=nw_max)

    kmax = (nw_max+3)*(nw_max+4)/2
    weighted_gal_moments, weighted_psf_moments = np.zeros(kmax), np.zeros(kmax)
    for k in xrange(kmax):
        m,n = singlet_to_doublet(k)
        weighted_psf_moments[k] = measure_moments(m,n,psf_img,psf_grid,psf_weight,A=None)
        weighted_gal_moments[k] = measure_moments(m,n,gal_img,gal_grid,gal_weight,A=None)

    ellipticity = [ ]
    for _nw in nw_list:
        kmax = (_nw+3)*(_nw+4)/2
        deweighted_gal_moments = np.dot(gal_DW[:,:kmax], weighted_gal_moments[:kmax])
        deweighted_psf_moments = np.dot(psf_DW[:,:kmax], weighted_psf_moments[:kmax])
        psf_corrected_moments = psf_correction(deweighted_gal_moments, deweighted_psf_moments, matrix_inv=matrix_inv)
        ellip = moments_to_ellipticity(psf_corrected_moments, etype=etype)

        if ((psf_corrected_moments[doublet_to_singlet(0,2)]<0) or (psf_corrected_moments[doublet_to_singlet(2,0)]<0)):
            status = -3
        else:
            status = 0 if moments_to_size(psf_corrected_moments, size_type='det')>=0 else -2

        ellipticity.append( ellip+(status,) )

    if hasattr(nw,'__iter__'):
        return ellipticity
    else:
        return ellipticity[0]
