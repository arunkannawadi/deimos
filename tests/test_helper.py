import os, sys, time
sys.path.append('/Users/arunkannawadi/moments_metacal/')
sys.path.append('/disks/shear15/arunkannawadi/Moments_Metacal/')

from helper import *
from moments import *
from deimos import *

## A helper function
def assert_all(moments, sigma, int_shape, rho4):
    g1, g2 = 0.0, 0.0
    assert abs(moments_to_size(moments)-sigma)<1e-3
    assert abs(moments[1]/moments[0])<1e-9
    assert abs(moments[2]/moments[0])<1e-9
    assert abs(moments_to_ellipticity(moments)[0] - int_shape.g1)<1e-6
    assert abs(moments_to_ellipticity(moments)[1] - int_shape.g2)<1e-6
    assert abs(moments_to_ellipticity(moments,etype='chi')[0] - int_shape.e1)<1e-6
    assert abs(moments_to_ellipticity(moments,etype='chi')[1] - int_shape.e2)<1e-6
    #if rho4>=0:
    #    assert abs(moments_to_rho4(moments)-rho4)<1e-4

## Singlet-doublet indices test
def test_indices(printout=False):
    t1 = time.time()
    for k in xrange(45):
        i,j = singlet_to_doublet(k)
        if printout:
            print "k = ",k, " --> ", (i,j)

        ## Test reversibility
        l = doublet_to_singlet(i,j)
        assert k==l

    set_k = [ ]
    for i in xrange(9):
        for j in xrange(9):
            if i+j<=8:
                set_k.append( doublet_to_singlet(i,j) )
    assert set(set_k)==set(range(45))
    t2 = time.time()
    print "test_indices() completed in %.2f seconds." % (t2-t1)

def test_weight_image():
    t1 = time.time()
    sigma, g1, g2 = 3.7, 0.4, 0.3
    size, scale = 280, 0.5
    grid = generate_pixelgrid(centroid=[-0.5,-0.5],size=(size,size),scale=scale)
    weight = get_weight_image(grid, gauss_sigma=sigma, gauss_e1=g1, gauss_e2=g2)
    moments = np.zeros(15)
    for k in xrange(14):
        i,j = singlet_to_doublet(k)
        moments[k] = measure_moments(i,j,weight,grid,weight=1.)

    print moments
    int_shape = galsim.Shear(g1=g1,g2=g2)
    print moments[0]*scale**2, 2.*np.pi*sigma**2
    print moments[1]/moments[0], moments[2]/moments[0]
    print moments_to_size(moments), sigma
    print moments_to_ellipticity(moments), int_shape.g1, int_shape.g2
    print moments_to_ellipticity(moments,etype='chi'), int_shape.e1, int_shape.e2
    print moments_to_rho4(moments), 2*(sigma/scale)**4
    assert_all(moments, sigma, galsim.Shear(g1=g1,g2=g2),-1)

    t2 = time.time()
    print "test_weight_image() completed in %f seconds." % (t2-t1)

if __name__=='__main__':
    test_indices()
    test_weight_image()
