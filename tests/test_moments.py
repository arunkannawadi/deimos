import numpy as np
import galsim
import os, sys, time

sys.path.append('/Users/arunkannawadi/moments_metacal/')
sys.path.append('/disks/shear15/arunkannawadi/Moments_Metacal/')
from helper import *
from moments import *
from test_helper import assert_all

def test_moments():
    t1 = time.time()

    ## Unweighted and weighted moments
    sigma, scale = 3.7, 1.0
    q, theta = 0.4, 30.
    int_shape = galsim.Shear(q=q,beta=theta*galsim.degrees)
    gauss = galsim.Gaussian(sigma=sigma).shear(int_shape)
    img = gauss.drawImage(scale=scale, method='no_pixel')
    hsm = img.FindAdaptiveMom()
    grid = generate_pixelgrid(centroid=[-0.5,-0.5],size=img.array.shape, scale=scale)
    weight = get_weight_image(grid, hsm.moments_sigma,hsm.observed_shape.g1, hsm.observed_shape.g2)
    unweighted_moments, weighted_moments = np.zeros(15), np.zeros(15)
    for k in xrange(14):
        i,j = singlet_to_doublet(k)
        unweighted_moments[k] = measure_moments(i,j,img,grid,weight=1.)
        weighted_moments[k] = measure_moments(i,j,img,grid,weight=weight)
    
    moments = weighted_moments
    print moments[0]
    print moments[1]/moments[0], moments[2]/moments[0]
    print moments_to_size(moments), hsm.moments_sigma*scale/np.sqrt(2.), sigma
    print moments_to_ellipticity(moments), int_shape.g1, int_shape.g2
    print moments_to_ellipticity(moments,etype='chi'), int_shape.e1, int_shape.e2
    print hsm.moments_centroid, img.array.shape 
    print moments_to_rho4(moments), hsm.moments_rho4*hsm.moments_sigma**4

    assert_all(unweighted_moments,sigma,int_shape,-1)
    assert_all(weighted_moments,sigma/np.sqrt(2.),int_shape,-1)

    n, hlr = 1.5, 4.2
    int_shape = galsim.Shear(g1=0.4, g2=0.3)
    sersic = galsim.Sersic(n=n, half_light_radius=hlr).shear(int_shape)
    img = sersic.drawImage(scale=scale, method='no_pixel')
    hsm = img.FindAdaptiveMom()
    grid = generate_pixelgrid(centroid=[-0.5,-0.5],size=img.array.shape,scale=scale)
    weight = get_weight_image(grid, hsm.moments_sigma,hsm.observed_shape.g1,hsm.observed_shape.g2)
    for k in xrange(14):
        i,j = singlet_to_doublet(k)
        unweighted_moments[k] = measure_moments(i,j,weight,grid,weight=1.)
        weighted_moments[k] = measure_moments(i,j,img,grid,weight=weight)
   
    moments = weighted_moments
    print moments[0]
    print moments[1]/moments[0], moments[2]/moments[0]
    print moments_to_size(moments)*np.sqrt(2.), hsm.moments_sigma*scale, moments_to_size(unweighted_moments)
    print moments_to_ellipticity(moments), hsm.observed_shape
    print moments_to_ellipticity(moments,etype='chi'), hsm.observed_shape.e1, hsm.observed_shape.e2
    print hsm.moments_centroid, img.array.shape
    print moments_to_rho4(moments), hsm.moments_rho4*hsm.moments_sigma**4
    assert_all(weighted_moments,hsm.moments_sigma/np.sqrt(2.),hsm.observed_shape,hsm.moments_rho4)

    t2 = time.time()
    print "test_moments() completed in %f seconds. " % (t2-t1)

if __name__=='__main__':
    test_moments()
