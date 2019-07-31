import numpy as np
import galsim
import os, sys, time

sys.path.append('/Users/arunkannawadi/moments_metacal/')
sys.path.append('/disks/shear15/arunkannawadi/Moments_Metacal/')
from helper import *
from moments import *
from deimos import *

def test_psf_correction():
    t1 = time.time()

    gal_sigma, psf_sigma = 2.6, 0.8
    scale = 0.1
    gal_shape = galsim.Shear(g1=0.4,g2=0.3)
    psf_shape = galsim.Shear(g1=0.0,g2=0.0)
    psf = galsim.Gaussian(sigma=psf_sigma).shear(psf_shape)
    gal = galsim.Gaussian(sigma=gal_sigma).shear(gal_shape)

    psf_img = psf.drawImage(scale=scale, method='no_pixel')
    psf_grid = generate_pixelgrid(centroid=[-0.5,-0.5],size=psf_img.array.shape,scale=scale)

    gal_img = galsim.Convolve([gal,psf]).drawImage(scale=scale, method='no_pixel')
    gal_hsm = gal_img.FindAdaptiveMom()
    gal_grid = generate_pixelgrid(centroid=[-0.5,-0.5],size=gal_img.array.shape,scale=scale)
    weight_img = get_weight_image(gal_grid, gal_hsm.moments_sigma, gal_hsm.observed_shape.g1, gal_hsm.observed_shape.g2)

    psf_moments, gal_weighted_moments = np.zeros(6), np.zeros(6)
    for k in xrange(6):
        i,j = singlet_to_doublet(k)
        psf_moments[k] = measure_moments(i,j,psf_img,psf_grid,weight=1.)
        gal_weighted_moments[k] = measure_moments(i,j,gal_img,gal_grid,weight=1.)

    gal_moments = psf_correction(gal_weighted_moments, psf_moments, matrix_inv=False)
    gal_int_shape = moments_to_ellipticity(gal_moments)
    print "Without matrix inversion, shape=", gal_int_shape, " and size=", moments_to_size(gal_moments)

    gal_moments = psf_correction(gal_weighted_moments, psf_moments, matrix_inv=True)
    gal_int_shape = moments_to_ellipticity(gal_moments)
    print "With matrix inversion, shape=", gal_int_shape, " and size=", moments_to_size(gal_moments)

    t2 = time.time()
    print "test_psf_correction() completed in %f seconds." % (t2-t1)

def test_deweighting():
    t1 = time.time()

    gal_n, gal_hlr = 1.5, 2.5
    psf_sigma = 0.7
    scale = 0.1
    gal_shape = galsim.Shear(g1=0.4,g2=0.3)
    psf_shape = galsim.Shear(g1=0.01,g2=0.02)
    psf = galsim.Gaussian(sigma=psf_sigma).shear(psf_shape)
    gal = galsim.Sersic(n=gal_n, half_light_radius=gal_hlr).shear(gal_shape)
    img = galsim.Convolve([gal,psf]).drawImage(scale=scale)
    hsm = img.FindAdaptiveMom()
    grid = generate_pixelgrid(centroid=[-0.5,-0.5], size=img.array.shape,scale=scale)
    weight_img = get_weight_image(grid, hsm.moments_sigma*scale, hsm.observed_shape.g1, hsm.observed_shape.g2)

    ndw = 4
    unweighted_moments = np.zeros((ndw+1)*(ndw+2)/2)
    for k in xrange((ndw+1)*(ndw+2)/2):
        i,j = singlet_to_doublet(k)
        unweighted_moments[k] = measure_moments(i,j,img,grid,weight=1.)
    print "Unweighted moments: "
    print unweighted_moments

    weighted_moments = np.zeros(91)
    for k in xrange(len(weighted_moments)):
        i,j = singlet_to_doublet(k)
        weighted_moments[k] = measure_moments(i,j,img,grid,weight=weight_img)

    for nw in [0,2,4,6,8,10]:
        kmax = (nw+3)*(nw+4)/2    
        DW = generate_deweighting_matrix(hsm.moments_sigma*scale, hsm.observed_shape.g1, hsm.observed_shape.g2, nw=nw)
        deweighted_moments = np.dot(DW, weighted_moments[:kmax])
        print "Deweighted moments with nw = ", nw
        print deweighted_moments

    ta = time.time()
    DW1 = generate_deweighting_matrix(hsm.moments_sigma*scale, hsm.observed_shape.g1, hsm.observed_shape.g2, nw=16, ndw=ndw)
    tb = time.time()
    DW2 = calculate_deweighting_matrix(hsm.moments_sigma*scale, hsm.observed_shape.g1, hsm.observed_shape.g2, nw=16, ndw=ndw)
    tc = time.time()

    print "generate_deweighting_matrix took  %f seconds." % (tb-ta)
    print "calculate_deweighting_matrix took %f seconds." % (tc-tb)
    # print DW1
    # print DW2
    np.testing.assert_array_almost_equal(DW1, DW2, decimal=10)

    t2 = time.time()
    print "test_deweighting() completed in %f seconds." % (t2-t1)

def test_deimos():
    t1 = time.time()

    gal_n, gal_hlr = 1.5, 2.5
    psf_sigma = 0.7
    scale, psf_scale = 0.1, 0.05
    gal_shape = galsim.Shear(g1=0.4,g2=0.3)
    psf_shape = galsim.Shear(g1=0.01,g2=0.02)
    psf = galsim.Gaussian(sigma=psf_sigma).shear(psf_shape)
    gal = galsim.Sersic(n=gal_n, half_light_radius=gal_hlr).shear(gal_shape)
    gal_img = galsim.Convolve([gal,psf]).drawImage(scale=scale, method='no_pixel')
    psf_img = psf.drawImage(scale=psf_scale, method='no_pixel')

    print "True shape = ", gal_shape.e1, gal_shape.e2
    nw = [0,2,4,6,8,10]
    int_shape = deimos(gal_img, psf_img, nw=nw, psf_w_sigma=None, scale=scale, psf_scale=psf_scale)
    print "Measured shape for nw=", nw, " is ", int_shape

    t2 = time.time()
    print "test_deimos() completed in %f seconds." % (t2-t1)

if __name__=='__main__':
    test_psf_correction()
    test_deweighting()
    test_deimos()
