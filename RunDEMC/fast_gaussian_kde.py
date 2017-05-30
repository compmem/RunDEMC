"""
A faster gaussian kernel density estimate (KDE).
Intended for computing the KDE on a regular grid (different use case than 
scipy's original scipy.stats.kde.gaussian_kde()).
-Joe Kington
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.signal

def fast_kde(x, y, gridsize=(200, 200), extents=None, nocorrelation=False, weights=None):
    """
    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    This function is typically several orders of magnitude faster than 
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and 
    produces an essentially identical result.

    Input:
        x: The x-coords of the input data points
        y: The y-coords of the input data points
        gridsize: (default: 200x200) A (nx,ny) tuple of the size of the output 
            grid
        extents: (default: extent of input data) A (xmin, xmax, ymin, ymax)
            tuple of the extents of output grid
        nocorrelation: (default: False) If True, the correlation between the
            x and y coords will be ignored when preforming the KDE.
        weights: (default: None) An array of the same shape as x & y that 
            weighs each sample (x_i, y_i) by each value in weights (w_i).
            Defaults to an array of ones the same size as x & y.
    Output:
        A gridded 2D kernel density estimate of the input points. 
    """
    #---- Setup --------------------------------------------------------------
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)
    
    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    nx, ny = gridsize
    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input x & y arrays!')

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = list(map(float, extents))
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    #---- Preliminary Calculations -------------------------------------------

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6) # For 2D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**2) 

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel 
    kernel = np.sum(kernel, axis=0) / 2.0 
    kernel = np.exp(-kernel) 
    kernel = kernel.reshape((kern_ny, kern_nx))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.  
    norm_factor = 2 * np.pi * cov * scotts_factor**2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    return np.flipud(grid)

def fast_1d_kde(x, nx=200, extents=None, weights=None):
    """
    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 1D histogram of the data.

    This function is typically several orders of magnitude faster than 
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and 
    produces an essentially identical result.

    Input:
        x: The x-coords of the input data
        gridsize: (default: 200x200) Size of the output grid
        extents: (default: extent of input data) A (xmin, xmax)
            tuple of the extents of output grid
        weights: (default: None) An array of the same shape as x 
            weighs each sample x_i by each value in weights (w_i).
            Defaults to an array of ones the same size as x.
    Output:
        A gridded 1D kernel density estimate of the input points. 
    """
    #---- Setup --------------------------------------------------------------
    x = np.array(x)
    x = np.squeeze(x)
    
    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input array!')

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = list(map(float, extents))
    dx = (xmax - xmin) / (nx - 1)

    #---- Preliminary Calculations -------------------------------------------

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    #xyi = np.vstack((x,y)).T
    xi = x
    xi -= xmin
    xi /= dx
    xi = np.floor(xi)
    #xi = np.vstack([xi,np.zeros_like(xi)])

    # Next, make a histogram of x
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, np.vstack([xi,np.zeros_like(xi)])), 
                                shape=(nx,1)).toarray()[:,0]

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(np.atleast_2d(xi),rowvar=1,bias=False) + np.finfo(xi.dtype).eps
    
    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 5) # For 1D (n**(-1/(d+4))

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    #std_devs = np.diag(np.sqrt(cov))
    std_devs = np.sqrt(cov)
    #kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)
    #kern_nx = np.round(scotts_factor * 2 * np.pi * std_devs)
    kern_nx = np.round(scotts_factor * 2 * np.pi * std_devs)
    if kern_nx <= 0:
        kern_nx = 1

    # Determine the bandwidth to use for the gaussian kernel
    #inv_cov = np.linalg.inv(cov * scotts_factor**2) 
    inv_cov = 1./(cov * scotts_factor**2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    #yy = np.array([0]) #.np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    #xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    #kernel = np.vstack((xx.flatten(), yy.flatten()))
    #kernel = xx
    #kernel = np.dot(inv_cov, kernel) * kernel 
    kernel = np.dot(inv_cov, xx) * xx
    #kernel = np.sum(kernel, axis=0) / 2.0 
    kernel = np.exp(-kernel/2.0) 
    #kernel = kernel.reshape((kern_ny, kern_nx))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    #grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T
    grid = sp.signal.convolve(grid, kernel, mode='same')

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.  
    #norm_factor = 2 * np.pi * cov * scotts_factor**2
    norm_factor = 2 * np.pi * cov * scotts_factor**2
    #norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * np.sqrt(norm_factor)
    
    # Normalize the result
    grid /= norm_factor

    #np.squeeze(np.flipud(grid))
    return grid,np.linspace(xmin,xmax,nx)#+dx
