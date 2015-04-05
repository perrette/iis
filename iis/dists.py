""" Helper to define distributions classes
"""
import copy
import logging
import numpy as np
from scipy.stats import norm, uniform, lognorm, multivariate_normal
from scipy.stats.kde import gaussian_kde  # estimate kernel density distribution

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class KDE(object):
    """ Wrapper class to build a distribution from empirical data
    with the same methods as scipy.stats' parametric distributions

    Use scipy.stats.kde to obtain a smooth representation of the 
    distribution

    Thanks
    ------
    http://nbviewer.ipython.org/github/nicolasfauchereau/NIWA_Python_seminars/blob/master/4_Statistical_modelling.ipynb

    Examples
    --------
    >>> # truth = lognorm(1) # works ok, but get < 0 values
    >>> truth = norm()
    >>> samples = truth.rvs(100)
    >>> dist = KDE(samples)
    >>> xgrid = np.linspace(-10, 10, 100)
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(samples, histtype='step', normed=True, label='sampled')
    >>> plt.plot(xgrid, dist.pdf(xgrid), label='fitted')
    >>> plt.plot(xgrid, truth.pdf(xgrid), label='truth')
    >>> plt.legend(frameon=True)
    """
    def __init__(self, dataset, *args, **kwargs):
        self.fit(dataset, *args, **kwargs)

    def fit(self, dataset, *args, **kwargs):
        """
        dataset : array_like
            Datapoints to estimate from. In case of univariate data this is a 1-D
            array, otherwise a 2-D array with shape (# of samples, # of dims)
            (Its transpose is passed to scipy.stats.gaussian_kde)
        """
        self.kde = gaussian_kde(dataset.T, *args, **kwargs)

    @property
    def dataset(self):
        return self.kde.dataset.T

    def pdf(self, x):
        return self.kde.pdf(x)

    def logpdf(self, x):
        return self.kde.logpdf(x)

    def var(self):
        return np.var(self.dataset, axis=0)

    def std(self):
        return np.std(self.dataset, axis=0)

    def mean(self):
        return np.mean(self.dataset, axis=0)

    def median(self):
        return np.median(self.dataset, axis=0)

    def ppf(self, q):
        return np.percentiles(self.dataset, np.asarray(q)/100., axis=0)

    def rvs(self, size=None, raw=False):
        """ Random variable sample
        size : number of variables to draw
        raw : if True, use discrete draw of initial dataset
                instead of kde.resample (default to False)
        """
        if raw:
            ids = np.random.randint(self.dataset.shape[0], size=size)
            samples = self.dataset[ids]
        else:
            samples = self.kde.resample(size)
        return samples


#
# Combine several distributions into a multivariate distribution
#

class MultivariateComposite(object):
    """ Multivariate distribution constructed from multiple, uncorrelated scipy.dists' like distributions

    Examples
    --------
    >>> dist = MultivariateComposite()
    >>> dist.append_dist(norm(loc=0, scale=1))
    >>> dist.append_dist(uniform(loc=0, scale=1))
    >>> dist.mean() # mean value
    array([ 0. ,  0.5])
    >>> dist.pdf([0, 0]) 
    0.3989422804014327
    >>> dist.pdf([[0, 0],[-1, -1]]) 
    array([ 0.39894228,  0.        ])
    >>> dist.rvs() # random sample
    array([-0.12203606,  0.95917321])
    >>> dist.rvs(4)
    array([[-0.04185913,  0.81625594],
           [ 0.11530888,  0.73923087],
           [-1.62012524,  0.89062211],
           [-0.11473132,  0.2810124 ]])
    >>> dist.ppf(0.5) # median
    array([ 0. ,  0.5])
    >>> dist.ppf([0.05, 0.95]) # 90% range
    array([[-0.04185913,  0.81625594],
           [ 0.11530888,  0.73923087],
           [-1.62012524,  0.89062211],
           [-0.11473132,  0.2810124 ]])

    Can also append multivariate distributions
    >>> dist.append_dist(multivariate_normal(mean=[0, 0])) # size 2+2 !
    >>> dist.rvs(3)
    array([[-1.63025465,  0.64894417, -0.09001673,  0.28240077],
           [-1.10006775,  0.37548807, -0.04465086, -1.84934054],
           [-0.2026251 ,  0.41626579,  0.74605497,  0.87420546]])
    >>> dist.logpdf([-1.10006775,  0.37548807, -0.04465086, -1.84934054])
    -5.0729171930021648

    Note: only rvs and logpdf are available in multivariate_normal
    """
    def __init__(self, dists=None, required_methods=('rvs','logpdf')):
        self.dists = []
        self.ndims = [] # number of dimension for each distribution

        self.required_methods = required_methods
        if dists is not None:
            if isinstance(dists, MultivariateComposite):
                dists = dists.dists
            elif not isinstance(dists, list) or isinstance(dists, tuple):
                dists = [dists]
            for dist in dists:
                self.append_dist(dist)

    def copy(self): 
        dist = MultivariateComposite()
        dist.dists = copy.copy(self.dists)
        dist.ndims = copy.copy(self.ndims)
        return dist

    @property
    def ndim(self):
        return sum(self.ndims)

    def append_dist(self, dist):
        """ append a new distribution, can be mono- or multi-variate
        """
        if isinstance(dist, MultivariateComposite):
            for d in dist.dists:
                self.append_dist(d)
            return

        for m in self.required_methods:
            assert hasattr(dist, m) and callable(getattr(dist,m)), '{}-dim distribution invalid: {} method missing'.format(self.ndim, m)

        self.dists.append(dist)
        if hasattr(dist, 'rvs'):
            self.ndims.append(np.size(dist.rvs())) # draw one random sample to get the size
        else:
            logging.warning('rvs method missing, assume monovariate distribution')
            self.ndims.append(1) # assume size 1

    def __iter__(self):
        """ iterate on the distribution, yield dist and index slice
        """
        start = 0
        for i, dist in enumerate(self.dists):
            count = self.ndims[i]
            if count == 1:
                idx = start
            else:
                idx = slice(start,start+count)
            yield dist, idx
            start += count

    def _concat_scalar_results(self, method, *args, **kwargs):
        res = np.empty(self.ndim)
        for dist, slice_ in self:

            # special cases in case of a multivariate normal 
            is_mvn = dist.__class__.__name__ in ('multivariate_normal_gen','multivariate_normal_frozen')
            if is_mvn and method == 'mean':
                res[slice_] = dist.mean
                continue
            elif is_mvn and method == 'var':
                res[slice_] = np.diag(dist.cov)
                continue
            elif is_mvn and method == 'std':
                res[slice_] = np.sqrt(np.diag(dist.cov))
                continue

            res[slice_] = getattr(dist, method)(*args, **kwargs)
        return res

    def _concat_array_results(self, method, result_size, *args, **kwargs):
        res = np.empty((result_size, self.ndim))
        for dist, slice_ in self:
            res[:, slice_] = getattr(dist, method)(*args, **kwargs)
        return res

    def logpdf(self, x):
        """ non-normalized version of pdf
        """
        logpdf = 0
        x = np.asarray(x)
        for dist, slice_ in self:
            if np.ndim(x) == 2:
                assert np.shape(x)[1] == self.ndim, 'x must be samples x dims' 
                logpdf += dist.logpdf(x[:, slice_])
            else:
                assert np.size(x) == self.ndim, 'invalid size for x, expected {}, got {}'.format(self.ndim, np.size(x))
                logpdf += dist.logpdf(x[slice_])
        return logpdf

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=None):
        # return self._concat_results('rvs', size)
        if size is None:
            return self._concat_scalar_results('rvs', size=size)
        else:
            return self._concat_array_results('rvs', size, size=size)

    def ppf(self, q):
        if np.isscalar(q):
            return self._concat_scalar_results('ppf', q)
        else:
            return self._concat_array_results('ppf', np.size(q), q)

    def mean(self):
        return self._concat_scalar_results('mean')
    def var(self):
        return self._concat_scalar_results('var')
    def std(self):
        return self._concat_scalar_results('std')
    def median(self):
        return self._concat_scalar_results('median')


def introduce_correlation(x, r, factor=False, normalize=True):
    """ Introduce correlation in a sequence of sampled variables

    Parameters
    ----------
    x: 2-d ndarray (samples x variables)
        Uncorrelated samples
    r: 2-d ndarray
        Target correlation matrix (variables x variables)
        or factor (variables x subspace) if factor=True
    factor : bool, optional
        if True, r is a factor for the correlation matrix
        so that correlation_matrix = r*r.T
        if False (the default), Choleski decomposition of 
        r is used.
    normalize : bool, optional
        if True, normalize the variables prior to matrix
        multiplication, to enhance numerical precision but
        at a slightly higher computational cost.
        Default to True.

    Returns
    -------
    correlated_x : 2-d ndarray (samples x variables)
        correlated samples

    Reference
    ---------
    After scipy's cookbook:
    http://wiki.scipy.org/Cookbook/CorrelatedRandomSamples
    
    but normalize sample before multiplying by C 

    See Also:
    ---------
    numpy.random.multivariate_normal
    """
    from scipy.linalg import eigh, cholesky, LinAlgError
    x = np.asarray(x).T
    p, num_samples = np.shape(x)

    assert np.ndim(r) == 2 and r.shape[0] == p, 'shapes do not match'

    # We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
    # the Cholesky decomposition, or the we can construct `c` from the
    # eigenvectors and eigenvalues.
    if factor:
        c = r
    else:
        try:
            c = cholesky(r, lower=True)
        except LinAlgError as error:
            logging.warn(error.message+' ==> problem in Choleski decomposition compute eigenvalues instead' )
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(r)
            # Construct c, so c*c^T = r.
            c = np.dot(evecs, np.diag(np.sqrt(evals)))

    # y = c.x
    # cov(y,y) = 1/(n-1) y.y* = c 1/(n-1) x.x* c* = c cov(x,x) c*
    # if x is i.i.d, cov(x,x) = I and cov(y,y) = c c * = r
    # ==> needs to normalize the variables, and transform r from
    # correlation to covariance matrix
    xm = np.mean(x, axis=1)
    xs = np.std(x, axis=1)
    xiid = (x-xm[:,None])/xs[:,None]

    print 'initial correl', np.corrcoef(x)
    print 'initial covariance', np.cov(x)

    print 'x iid'
    print 'mean', np.mean(xiid, 1)
    print 'cov', np.cov(xiid)
    print 'corr', np.corrcoef(xiid)

    # introduce covariance (since cov = diag(sigma) corr diag(sigma) = c2 c2* where c2= diag(sigma) c)
    c2 = np.dot(np.diag(xs), c)

    print "target covariance", np.dot(c2, c2.T)/(num_samples-1)

    # Convert the data to correlated random variables. 
    y = np.dot(c2, xiid) + xm[:,None]

    print "actual covariance", np.cov(y)
    print "actual correlation", np.corrcoef(y)

    return y

