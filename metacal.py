import numpy as np

def shear_pixelgrid(grid, A):
    """ Shear the pixel grid to generate an affine grid
    """
    X, Y = grid
    x, y = X.flatten(), Y.flatten()
    xy = np.stack([x,y])
    assert xy.shape==(2,X.size)
    sheared_xy = np.dot(A,xy)
    x, y = sheared_xy[0], sheared_xy[1]
    sheared_grid = (np.reshape(x,X.shape),np.reshape(y,Y.shape))
    return sheared_grid

def shear_monomial(m, n, sheared_grid):
    X, Y = sheared_grid
    monomial = (X**m)*(Y**n)
    return monomial

def shear_gaussian():
    return sheared_gaussian

