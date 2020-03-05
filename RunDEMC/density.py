# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import numpy as np
import scipy as sp
import scipy.sparse
import scipy.signal


"""
A faster gaussian kernel density estimate (KDE).
Intended for computing the KDE on a regular grid (different use case than
scipy's original scipy.stats.kde.gaussian_kde()).
-Joe Kington
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'


def fast_2d_kde(x, y, gridsize=(200, 200), extents=None,
                nocorrelation=False, weights=None):
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
    xyi = np.vstack((x, y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1, 0] = 0
        cov[0, 1] = 0

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6)  # For 2D

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
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

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


def fast_1d_kde(x, nx=200, extents=None, weights=None,
                kstr='Gaussian'):
    """
    Performs a kernel density estimate over a regular grid using a
    convolution of the kernel (Gaussian or Epanechnikov) with a 1D
    histogram of the data.

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
        kstr: (default: 'Gaussian') String indicating the kernel
            to be used.
    Output:
        A gridded 1D kernel density estimate of the input points.
    """
    #---- Setup --------------------------------------------------------------
    x = np.array(x)
    x = np.squeeze(x)

    # proc the kernel
    kchar = kstr[0].upper()
    if kchar == 'G':
        ktype = 'Gaussian'
    elif kchar == 'E':
        ktype = 'Epanechnikov'
    else:
        raise ValueError("Unknown kernel type.")

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
    grid = sp.sparse.coo_matrix((weights, np.vstack([xi, np.zeros_like(xi)])),
                                shape=(nx, 1)).toarray()[:, 0]

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(np.atleast_2d(xi), rowvar=1, bias=False) + \
        np.finfo(xi.dtype).eps

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 5)  # For 1D (n**(-1/(d+4))

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
    inv_cov = 1. / (cov * scotts_factor**2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    # yy = np.array([0]) #.np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    #xx, yy = np.meshgrid(xx, yy)

    if ktype == 'Gaussian':
        # Then evaluate the gaussian function on the kernel grid
        #kernel = np.vstack((xx.flatten(), yy.flatten()))
        #kernel = xx
        #kernel = np.dot(inv_cov, kernel) * kernel
        kernel = np.dot(inv_cov, xx) * xx
        #kernel = np.sum(kernel, axis=0) / 2.0
        kernel = np.exp(-kernel / 2.0)
        #kernel = kernel.reshape((kern_ny, kern_nx))
    elif ktype == 'Epanechnikov':
        kernel = (1 - (xx * dx)**2) * .75

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    #grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T
    grid = sp.signal.convolve(grid, kernel, mode='same')

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    if ktype == 'Gaussian':
        #norm_factor = 2 * np.pi * cov * scotts_factor**2
        norm_factor = 2 * np.pi * cov * scotts_factor**2
        #norm_factor = np.linalg.det(norm_factor)
        norm_factor = n * dx * np.sqrt(norm_factor)
    elif ktype == 'Epanechnikov':
        norm_factor = cov * scotts_factor**2
        norm_factor = n * dx * norm_factor

    # Normalize the result
    grid /= norm_factor

    # 1/0

    # np.squeeze(np.flipud(grid))
    return grid, np.linspace(xmin, xmax, nx)  # +dx


def fast_pdf(dat, x, nbins, extrema=None):
    yg, xg = fast_1d_kde(dat, nx=nbins, extents=extrema)
    pdf = np.interp(x, xg, yg, left=0, right=0)
    return pdf


def vhist(h, b=None, reverse=False):
    """
    Create a variable bin-width histogram from an existing histogram
    by combining bins.

    Works by taking a pass from end backwards.  You can optionally
    reverse the direction (i.e., to get the result from both
    directions and combine them.)
    """
    # make the b if necessary
    if b is None:
        b = list(range(len(h)))

    # make sure they are lists
    h = list(h)
    b = list(b)

    # see if reverse
    if reverse:
        h.reverse()
        b.reverse()

    merged = True
    while merged:
        merged = False
        for i in range(len(h), 1, -1):
            # check for immediate identity
            if h[i - 2] == h[i - 1]:
                # merge them and continue
                h.pop(i - 1)
                b.pop(i - 1)
                continue
            if (i - 3) < 0:
                # skip b/c we need 3
                continue
            # pick the three bins to process
            v = h[i - 3:i]

            # Merge if
            #   1) all three are the same value or
            #   2) 0>1<2 (i.e., a drop, then rise)
            if ((v[0] == v[1]) & (v[1] == v[2])) or \
               ((v[0] > v[1]) & (v[2] > v[1])):
                # combine the three bins into two
                # say we merged
                merged = True
                # pick the first ind
                ind = i - 3
                # get the mid bin and height
                mid = h.pop(ind + 1)
                mid_bin = b.pop(ind + 1)
                # calc the three widths
                w1 = (mid_bin - b[ind])
                w2 = (b[ind + 1] - mid_bin) / 2.  # half
                w3 = (b[ind + 2] - b[ind + 1])
                # calc half the middle area
                mid_area = (w2 * mid)  # half
                # and the full left and right bin areas
                left_area = h[ind] * w1
                right_area = h[ind + 1] * w3
                # add the new area to the left and right
                h[ind] = (left_area + mid_area) / (w1 + w2)
                h[ind + 1] = (right_area + mid_area) / (w2 + w3)
                # adjust the bin edge
                b[ind + 1] = (mid_bin + w2)

    if reverse:
        # reverse it back
        h.reverse()
        b.reverse()
    return h, b


def hist_pdf(dat, x, nbins, extrema=None):
    """
    Calculate PDF with an interpolation of a variable bin histogram.
    """
    # set the extrema
    if extrema is None:
        #extrema = (x.min(),x.max())
        extrema = (dat.min(), dat.max())
    # allocate for final pdf the size of x
    pdf = np.zeros_like(x)

    # calculate the starting bin widths
    xdist = extrema[1] - extrema[0]
    bw = xdist / float(nbins - 1)
    bins = np.array([extrema[0] + bw * i - bw * .5 for i in range(nbins + 1)])
    #bw = xdist/float(nbins-1)
    #bins = np.array([extrema[0]+bw*i for i in range(nbins+1)])

    # calc the starting histogram
    hi = np.histogram(dat, bins, density=True)[0]

    # make copies of the bins and hists for forward and backward passes
    # b = [[bins[0]-bw] + bins.tolist() + [bins[-1]+bw] for i in range(2)]
    # b[1].reverse()
    # h = [[0.]+hi.tolist()+[0.] for i in range(2)]
    # h[1].reverse()
    #bs = [bins.tolist() for i in range(2)]
    #hs = [hi.tolist() for i in range(2)]
    bs = [[bins[0] - bw] + bins.tolist() + [bins[-1] + bw] for i in range(2)]
    hs = [[0.] + hi.tolist() + [0.] for i in range(2)]

    #b = [bins.tolist() for i in range(3)]
    #h = [hi.tolist() for i in range(3)]
    for s in range(len(bs)):
        # set if reverse
        reverse = s == 1
        # calc variable bins
        h, b = vhist(hs[s], bs[s], reverse=reverse)
        h = np.asarray(h)
        b = np.asarray(b)

        # remember to shift by (bw*.5)
        #pdf += np.interp(x,np.asarray(b[s][:-1])+(bw*.5),h[s])
        # combine interpolation from each edge
        pdf += (np.interp(x, b[:-1], h) + np.interp(x, b[1:], h)) / 2.

    return pdf / len(bs)  # ,b,h


def hist_pdf2(dat, x, nbins, extrema=None, scale=0.0):
    """
    Generate pdf from histograms.
    """

    if extrema is None:
        extrema = (x.min(), x.max())
    bscale = []
    pdf = np.zeros_like(x)
    xdist = extrema[1] - extrema[0]
    max_bin = xdist / float(nbins[0] - 1)
    for nb in nbins:
        # set the bins so that the first and last bin are centered on
        # the endpoints
        bw = xdist / float(nb - 1)
        bins = np.array([extrema[0] + bw * i - bw * .5 for i in range(nb + 1)])
        # determine the nonlinear scaling factor for the bins
        # scales by the max binsize, so starts at one and then
        # increases or decreases depending on the scale
        bscale.append((max_bin / bw)**scale)
        # add pdf scaled by bin width, interpolating missing points
        pdf += (bscale[-1]) * np.interp(x, bins[:-1] + (bw * .5),
                                        np.histogram(dat, bins, normed=True)[0])

    # normalize by bin_widths
    pdf /= np.asarray(bscale).sum()

    return pdf


def hist_stack(dat, x, nbins, extrema=None):
    """
    """
    if extrema is None:
        extrema = (x.min(), x.max())
    xdist = extrema[1] - extrema[0]
    max_bin = xdist / float(nbins[0] - 1)
    hists = []
    xvals = []
    for nb in nbins:
        # set the bins so that the first and last bin are centered on
        # the endpoints
        bw = xdist / float(nb - 1)
        bins = np.array([extrema[0] + bw * i - bw * .5 for i in range(nb + 1)])
        hists.append(np.histogram(dat, bins, normed=True)[0])
        xvals.append(bins[:-1] + (bw * .5))

    return hists, xvals


def freq_pmf(dat, x, nbins, extrema=None):
    """
    """
    # set the range
    if extrema is None:
        extrema = (x[0], x[-1])
    # get the range
    xdist = extrema[1] - extrema[0]
    # set the bin width from the nbins over that range
    bw = xdist / float(nbins)
    # define the bins
    bins = np.array([extrema[0] + bw * i - bw * .5 for i in range(nbins + 1)])
    # calc the pmf of the model data
    dpmf = np.histogram(dat, bins=bins)[0] / float(len(dat))

    # get the freqs of observed data
    xpmf = np.histogram(x, bins=bins)[0]
    # combine the observations (sum of log)
    # res = np.sum([np.sum([np.log(dpmf[b])]*xpmf[b])
    #              for b in range(len(bins))])

    pmf = []
    for b in range(nbins):
        pmf.extend([dpmf[b]] * xpmf[b])
    return np.asarray(pmf)


import scipy.optimize


def calc_scale_factor(dat, nbins=200, scale=1.0, verbose=True):
    """
    Find out scaling factor for your data to maximize acceptance rate.
    """
    # get total and props
    total = np.sum([len(d) for d in dat])
    props = [len(d) / float(total) for d in dat]

    def to_min(scale, *args):
        total_pdf = 0.0
        for d, p in zip(dat, props):
            # get the scaled pdf
            pdf = fast_pdf(d * scale, d * scale, nbins) * p

            # take log and add it to total
            total_pdf += np.log(pdf).sum()

        return np.abs(total_pdf)

    # run the minimization
    if verbose:
        disp = 1
    else:
        disp = 0
    best_scale = scipy.optimize.fmin(to_min,
                                     [scale],
                                     disp=disp)

    # return best result
    if verbose:
        print(best_scale)
    return float(best_scale)


# Box Cox transformation utils
def boxcox(x, lambdax):
    """
    Performs a box-cox transformation to data vector X.
    WARNING: elements of X should be all positive!
    """
    if np.any(x <= 0):
        raise ValueError("Nonpositive value(s) in X vector")
    return np.log(x) if np.abs(lambdax) < 1.0e-5 else (x**lambdax - 1.0) / lambdax


def boxcox_loglike(x, lambdax):
    """
    Computes the log-likelihood function for a transformed vector Xtransform.
    """
    n = len(x)
    xb = boxcox(x, lambdax)
    S2 = (lambdax - 1.0) * np.log(x).sum()
    S = np.sum((xb - xb.mean())**2)
    S1 = (-n / 2.0) * np.log(S / n)
    return float(S2 + S1)


def best_boxcox_lambdax(x, lambdax=0, verbose=False):

    def to_min(lambdax, *args):
        # return the neg so maximize log like
        return -boxcox_loglike(x, lambdax)

    # run the minimization
    if verbose:
        disp = 1
    else:
        disp = 0
    best_lambdax = scipy.optimize.fmin(to_min,
                                       [lambdax],
                                       disp=disp)
    return float(best_lambdax)


from .dists import normal


def kdensity(x, extrema=None, kernel="gaussian",
             binwidth=None, nbins=512, weights=None,
             # bw="nrd0",
             adjust=1.0, cut=3, xx=None):
    """Calculate kernel density via FFT.
    """
    # function (x, bw = "nrd0", adjust = 1, kernel = c("gaussian",
    #     "epanechnikov", "rectangular", "triangular", "biweight",
    #     "cosine", "optcosine"), weights = NULL, window = kernel,
    #     width, give.Rkern = FALSE, n = 512, from, to, cut = 3, na.rm = FALSE,
    #     ...)
    # {
    #     if (length(list(...)))
    #         warning("non-matched further arguments are disregarded")
    #     if (!missing(window) && missing(kernel))
    #         kernel <- window
    #     kernel <- match.arg(kernel)
    #     if (give.Rkern)
    #         return(switch(kernel, gaussian = 1/(2 * sqrt(pi)), rectangular = sqrt(3)/6,
    #             triangular = sqrt(6)/9, epanechnikov = 3/(5 * sqrt(5)),
    #             biweight = 5 * sqrt(7)/49, cosine = 3/4 * sqrt(1/3 -
    #                 2/pi^2), optcosine = sqrt(1 - 8/pi^2) * pi^2/16))
    #     if (!is.numeric(x))
    #         stop("argument 'x' must be numeric")
    #     name <- deparse(substitute(x))
    #     x <- as.vector(x)
    #     x.na <- is.na(x)
    #     if (any(x.na)) {
    #         if (na.rm)
    #             x <- x[!x.na]
    #         else stop("'x' contains missing values")
    #     }
    #     N <- nx <- length(x)
    N = len(x)
    nx = len(x)
    #     x.finite <- is.finite(x)
    #     if (any(!x.finite)) {
    #         x <- x[x.finite]
    #         nx <- length(x)
    #     }
    #     if (is.null(weights)) {
    #         weights <- rep.int(1/nx, nx)
    #         totMass <- nx/N
    #     }
    #     else {
    #         if (length(weights) != N)
    #             stop("'x' and 'weights' have unequal length")
    #         if (!all(is.finite(weights)))
    #             stop("'weights' must all be finite")
    #         if (any(weights < 0))
    #             stop("'weights' must not be negative")
    #         wsum <- sum(weights)
    #         if (any(!x.finite)) {
    #             weights <- weights[x.finite]
    #             totMass <- sum(weights)/wsum
    #         }
    #         else totMass <- 1
    #         if (!isTRUE(all.equal(1, wsum)))
    #             warning("sum(weights) != 1  -- will not get true density")
    #     }
    if weights is None:
        weights = np.ones(nx) / float(nx)
        totMass = nx / float(N)
    else:
        totMass = 1.0

    #     n.user <- n
    #     n <- max(n, 512)
    #     if (n > 512)
    #         n <- 2^ceiling(log2(n))
    nbins_user = nbins
    nbins = max(nbins, 512)
    if nbins > 512:
        nbins = int(2**np.ceil(np.log2(nbins)))
    #     if (missing(bw) && !missing(width)) {
    #         if (is.numeric(width)) {
    #             fac <- switch(kernel, gaussian = 4, rectangular = 2 *
    #                 sqrt(3), triangular = 2 * sqrt(6), epanechnikov = 2 *
    #                 sqrt(5), biweight = 2 * sqrt(7), cosine = 2/sqrt(1/3 -
    #                 2/pi^2), optcosine = 2/sqrt(1 - 8/pi^2))
    #             bw <- width/fac
    #         }
    #         if (is.character(width))
    #             bw <- width
    #     }
    #     if (is.character(bw)) {
    #         if (nx < 2)
    #             stop("need at least 2 points to select a bandwidth automatically")
    #         bw <- switch(tolower(bw), nrd0 = bw.nrd0(x), nrd = bw.nrd(x),
    #             ucv = bw.ucv(x), bcv = bw.bcv(x), sj = , `sj-ste` = bw.SJ(x,
    #                 method = "ste"), `sj-dpi` = bw.SJ(x, method = "dpi"),
    #             stop("unknown bandwidth rule"))
    #     }
    bw = nrd0(x)
    #     if (!is.finite(bw))
    #         stop("non-finite 'bw'")
    #     bw <- adjust * bw
    bw *= adjust

    # for some reason I have to multiply bw by 2
    #bw *= 2.

    #     if (bw <= 0)
    #         stop("'bw' is not positive.")
    #     if (missing(from))
    #         from <- min(x) - cut * bw
    #     if (missing(to))
    #         to <- max(x) + cut * bw
    if extrema is None:
        extrema = (np.min(x) - cut * bw, np.max(x) + cut * bw)
    #     if (!is.finite(from))
    #         stop("non-finite 'from'")
    #     if (!is.finite(to))
    #         stop("non-finite 'to'")
    #     lo <- from - 4 * bw
    #     up <- to + 4 * bw
    lo = np.squeeze(extrema[0] - 4 * bw)
    up = np.squeeze(extrema[1] + 4 * bw)
    # print extrema,lo,up
    #     y <- .Call(C_BinDist, x, weights, lo, up, n) * totMass
    #y = np.histogram(x, nbins=nbins, weights=weights, range=(lo,up))*totMass
    xi = x.copy()
    xi -= lo
    xi /= (up - lo) / (nbins - 1)
    xi = np.floor(xi)

    # Next, make a histogram of x
    # Avoiding np.histogram2d due to excessive memory usage with many points
    y = sp.sparse.coo_matrix((weights, np.vstack([xi, np.zeros_like(xi)])),
                             shape=(nbins, 1)).toarray()[:, 0] * totMass

    #     kords <- seq.int(0, 2 * (up - lo), length.out = 2L * n)
    kords = np.linspace(0, 2 * (up - lo), 2 * nbins)
    #     kords[(n + 2):(2 * n)] <- -kords[n:2]
    kords[nbins + 1:] = -kords[nbins - 1:0:-1]
    #     kords <- switch(kernel, gaussian = dnorm(kords, sd = bw),
    #         rectangular = {
    #             a <- bw * sqrt(3)
    #             ifelse(abs(kords) < a, 0.5/a, 0)
    #         }, triangular = {
    #             a <- bw * sqrt(6)
    #             ax <- abs(kords)
    #             ifelse(ax < a, (1 - ax/a)/a, 0)
    #         }, epanechnikov = {
    #             a <- bw * sqrt(5)
    #             ax <- abs(kords)
    #             ifelse(ax < a, 3/4 * (1 - (ax/a)^2)/a, 0)
    #         }, biweight = {
    #             a <- bw * sqrt(7)
    #             ax <- abs(kords)
    #             ifelse(ax < a, 15/16 * (1 - (ax/a)^2)^2/a, 0)
    #         }, cosine = {
    #             a <- bw/sqrt(1/3 - 2/pi^2)
    #             ifelse(abs(kords) < a, (1 + cos(pi * kords/a))/(2 *
    #                 a), 0)
    #         }, optcosine = {
    #             a <- bw/sqrt(1 - 8/pi^2)
    #             ifelse(abs(kords) < a, pi/4 * cos(pi * kords/(2 *
    #                 a))/a, 0)
    #         })

    # NOTE: bw is doubled here to match doubled width of kernel
    bw2 = bw * 2.
    if kernel == 'gaussian':
        kords = normal(std=bw2).pdf(kords)
    elif kernel == 'epanechnikov':
        a = bw2 * np.sqrt(5)
        ax = np.abs(kords)
        ind = ax < a
        ax[ind] = .75 * (1 - (ax[ind] / a)**2) / a
        ax[~ind] = 0.0
        kords = ax
    else:
        raise ValueError("Unknown kernel type.")

    # squeeze to ensure 1d
    kords = np.squeeze(kords)
    
    #     kords <- fft(fft(y) * Conj(fft(kords)), inverse = TRUE)
    kords = np.fft.ifft(np.concatenate([np.fft.fft(y)] * 2) *
                        np.conj(np.fft.fft(kords)))
    #     kords <- pmax.int(0, Re(kords)[1L:n]/length(y))
    #kords = (np.real(kords)[:nbins]/float(len(y))).clip(0,np.inf)
    #kords = (np.real(kords)[::2]/float(len(y))).clip(0,np.inf)
    #kords = (np.real(kords)/float(len(y))).clip(0,np.inf)
    #kords = (np.real(kords)).clip(0,np.inf)*2.
    kords = (np.real(kords)[::2]).clip(0, np.inf) * 2.
    #     xords <- seq.int(lo, up, length.out = n)
    xords = np.linspace(lo, up, nbins)
    #     x <- seq.int(from, to, length.out = n.user)
    if xx is None:
        xx = np.linspace(extrema[0], extrema[1], nbins_user)
    pdf = np.interp(xx, xords, kords)
    #     structure(list(x = x, y = approx(xords, kords, x)$y, bw = bw,
    #         n = N, call = match.call(), data.name = name, has.na = FALSE),
    #         class = "density")
    # }

    return pdf, xx


def nrd0(x):
    """
    """
    # <bytecode: 0x9939994>
    # <environment: namespace:stats>
    # >
    # > getAnywhere('bw.nrd0')
    # A single object matching 'bw.nrd0' was found
    # It was found in the following places
    #   package:stats
    #   namespace:stats
    # with value

    # function (x)
    # {
    #     if (length(x) < 2L)
    #         stop("need at least 2 data points")
    #     hi <- sd(x)
    hi = np.std(x)
    #     if (!(lo <- min(hi, IQR(x)/1.34)))
    #         (lo <- hi) || (lo <- abs(x[1L])) || (lo <- 1)
    lo = min(hi, IQR(x) / 1.34)
    if lo == 0:
        lo = hi
        if lo == 0:
            lo = np.abs(x[0])
            if lo == 0:
                lo = 1
    #     0.9 * lo * length(x)^(-0.2)
    return 0.9 * lo * len(x)**(-0.2)


from scipy.stats.mstats import mquantiles


def IQR(x):
    # }
    # <bytecode: 0x9958d18>
    # <environment: namespace:stats>
    # > getAnywhere('IQR')
    # A single object matching 'IQR' was found
    # It was found in the following places
    #   package:stats
    #   namespace:stats
    # with value

    # function (x, na.rm = FALSE, type = 7)
    # diff(quantile(as.numeric(x), c(0.25, 0.75), na.rm = na.rm, names = FALSE,
    #     type = type))
    return np.diff(scipy.stats.mstats.mquantiles(x, [.25, .75]))


if __name__ == "__main__":

    sdata = """
.15 .09 .18 .10 .05 .12 .08
.05 .08 .10 .07 .02 .01 .10
.10 .10 .02 .10 .01 .40 .10
.05 .03 .05 .15 .10 .15 .09
.08 .18 .10 .20 .11 .30 .02
.20 .20 .30 .30 .40 .30 .05
"""
    X = np.asarray([float(x) for x in sdata.split()])
    best_lambdax = best_boxcox_lambdax(X, lambdax=0.0)
    print((best_lambdax, boxcox_loglike(X, best_lambdax)))
