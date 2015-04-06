""" Core classes and functions for the Iterative Importance Sampling after Annan and Hargreaves

References
----------
Annan, J. D., & Hargreaves, J. C. (2010). Efficient identification of 
ocean thermodynamics in a physical/biogeochemical ocean model with an iterative 
Importance Sampling method. Ocean Modelling, 32(3-4), 205-215. 
doi:10.1016/j.ocemod.2010.02.003
"""
from __future__ import division, print_function
import logging
import inspect
import copy
import numpy as np
from .resampling import multinomial_resampling, residual_resampling
from .dists import MultivariateComposite
from .utils import formatdoc

# logging.basicConfig(filename='example.log',level=logging.INFO)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig()
# logging.basicConfig(level=logging.DEBUG)


# DEFAULT PARAMETERS
# bounds for scaling the log-likehood
# SAMPLING_METHOD = "random_walk"
# SAMPLING_METHOD = "residual"
SAMPLING_METHOD = "residual"
EPSILON = 0.05  # start value for adaptive_posterior_exponent (kept if NEFF in NEFF_BOUNDS)
EPSILON_BOUNDS = (1e-3, 0.1)  # has priority over NEFF_BOUNDS
NEFF_BOUNDS = (0.5, 0.9)
DEFAULT_SIZE = 500
DEFAULT_ALPHA_TARGET = 0.95

# NOTE: There is a tradeoff in the choice of the epsilon value.
# The smaller the epsilon value, the slower the convergence, and 
# the more precision is required (bigger ensemble size). 
# Annan and Hargreave use 0.05, which we also find to yield
# good result for reasonable convergence properties (about 60 
# iteration to reach alpha = 0.95) with an ensemble size of 
# 50 (lower limit, quite noisy, not very stable convergence) 
# to 500 (much smoother). 
# Higher epsilon values will lead to more rapid and more stable convergence 
# but relies more on the Gaussian nature of the spread (since the 
# jitter is then larger)
# For a value of 0.01, 50 particles are not sufficient and may lead to 
# ensemble collapse. 500 particles or more are OK.
#
# Maybe the answer is to be found in the estimator of the sample variance
#  Var(s^2) = 2*sig^4/(n-1) where sig^2 is the true variance.
# and to errors introduced during resampling (full mathematical analysis
# remains to be done...)

#========================================
# Functions
#========================================

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions to Determine the exponent to weight the likelihood
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _get_Neff(weights):
    """ Return an estimate of the effective ensemble size
    """
    weightssum = np.sum(weights)
    weights = weights/weightssum
    Neff = 1./np.sum(weights**2)
    return Neff

def adaptive_posterior_exponent(loglik, epsilon=None, neff_bounds = NEFF_BOUNDS ):
    """ Compute likelihood exponents to avoid ensemble collapse

    Resampling weights are computed as:

        likelihood ~ exp(loglik)
        weights ~ likelihood ** epsilon

    where epsilon is an exponent (between 0 and 1) chosen so that the effective
    ensemble size of the resampled ensemble remains reasonable, thereby 
    avoiding ensemble collapse (where only very few of the original members 
    are resampled, due to large differences in the likelihood).

    If epsilon is not provided, it will be estimated dynamically to yields 
    an effective ensemble size between 0.5 and 0.9 of the original ensemble.

    Parameters
    ----------
    loglik : 1-D array of log-likehoods
    epsilon : initial value for epsilon
    neff_bounds : acceptable effective ensemble ratio

    Returns
    -------
    epsilon : exponent such that weights = likelihood**epsilon

    Notes
    -----
    Small epsilon value means flatter likelihood, more homogeneous resampling.
    and vice versa for large epsilon value. 

    References
    ----------
    Annan and Hargreaves, 2010, Ocean Modelling
    """
    # compute appropriate weights
    likelihood = np.exp(loglik)
    if np.sum(likelihood) == 0:
        raise RuntimeError('No ensemble member has a likelihood greater than zero: consider using less constraints')
    N = np.size(likelihood)

    # CHECK FOR CONVERGENCE: effective ensemble size of the model is equal to 90% of that of a uniform distribution
    Neff_weighted_obs = _get_Neff(likelihood)
    ratio_prior = Neff_weighted_obs/N

    logging.info("Epsilon tuning:")
    logging.info("...no epsilon (eps=1): Neff/N = {}".format(ratio_prior))

    # Now adjust the likelihood function so as to have an effective size 
    # between 50% and 90% that of the previous ensemble (that is, because of 
    # the resampling, always between 50% and 90% of a uniform distribution)
    eps_min, eps_max = EPSILON_BOUNDS
    epsilon = epsilon or EPSILON
    eps_prec = 1e-3 
    niter = 0
    while True:
        niter += 1
        logging.debug('niter: {}, epsilon: {}'.format(niter, epsilon))
        if niter > 100: 
            logging.warning("too many iterations when estimating exponent")
            break
            # raise RuntimeError("too many iterations when estimating exponent")

        ratio_eps = _get_Neff(likelihood**epsilon) / N

        if epsilon < eps_min:
            logging.info('epsilon = {} < {} = eps_min. Set back to eps_min. Effective ensemble size too low : Neff/N = {}'.format(epsilon,eps_min,ratio_eps))
            epsilon = eps_min
            break
        if epsilon > eps_max:
            logging.info('epsilon = {} > {} = eps_max. Set back to eps_max. Effective ensemble size too high : Neff/N = {}'.format(epsilon,eps_max,ratio_eps))
            epsilon = eps_max
            break

        # neff_bounds = [0.5, 0.9]
        if ratio_eps > neff_bounds[1]:
            # Effective ensemble size too high, increase epsilon
            eps_incr = max(eps_prec, (eps_max - epsilon)/2)
            epsilon += eps_incr
        elif ratio_eps < neff_bounds[0]:
            # Effective ensemble size too low, decrease epsilon
            eps_incr = max(eps_prec, (epsilon - eps_min)/2)
            epsilon -= eps_incr
        else:
            break

    logging.info("...epsilon={} : Neff/N = {}".format(epsilon, ratio_eps))

    return epsilon


def sample_with_bounds_check(params, covjitter, bounds):
    """ Sample from covariance matrix and update parameters

    Parameters
    ----------
    params : 1-D numpy array (p)
    covjitter : covariance matrix p * p
    bounds : 2*p array (2 x p)
        parameter bounds: array([min1,min2,min3,...],[max1,max2,max3,...])

    Returns
    -------
    newparams : 1-D numpy array of resampled parameters
    """
    assert params.ndim == 1

    # prepare the jitter
    tries = 0
    maxtries = 100
    while True:
        tries += 1
        newparams = np.random.multivariate_normal(params, covjitter)
        params_within_bounds = not np.any((newparams < bounds[0]) | (newparams > bounds[1]), axis=0)
        if params_within_bounds:
            logging.debug("Required {} time(s) sampling jitter to match bounds".format(tries, i))
            break
        if tries > maxtries : 
            logging.warning("Could not add jitter within parameter bounds")
            newparams = params
            break
    return newparams

#
# Maybe useful at some point to normalize parameter variance, 
# for numerical accuracy when computing covariance matrix and 
# so on. Alternative : just choose appropriately normalized 
# parameters beforehand...
#

def _normalize_params(params):
    raise NotImplementedError()
    return normalized_params, loc, scale

def _restore_params(normalized_params, loc, scale):
    raise NotImplementedError()
    return params

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main classes
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Model(object):
    """ A class representing a model, to be provided by the user
    """
    def __init__(self, func, likelihood, proposal=None, prior=None, 
                 initial_state=None, default_params=None, params_bounds=None, 
                 labels_state=None, labels_params=None, args=(), **kwargs):
        """ Define Model

        Parameters
        ----------
        func : functional form that takes a 1-d ndarray `params` (Np) argument, 
            an optional 1-d ndarray `initial_state` (Ns) as 2nd argument, and returns a 
            1-d ndarray `state` (Ns). It represents for example a physical model.
        likelihood : (list of) `scipy.dists`-like distribution(s), including `multivariate_normal` (dimension : Ns)
            It represents observations of state variables. Internally, it will 
            be combined using `iis.stats.MultivariateComposite` distribution.
            It has to match the size of `state` array returned by `func`.
            According to Bayes rule, the likelihood pdf is multiplied by the prior 
            to obtain the seeked-for posterior distribution.
        prior : idem, optional (dimension : Np)
            Prior distribution for the parameters, will be multiplied by the
            likelihood to obtain the posterior. If not provided, the `prior` 
            will be considered ignorant, thus model `params` and `state` will 
            converge toward the `likelihood`. In that case, `proposal` must be 
            provided to draw the initial `params` sample.
        proposal : idem, optional (dimension : Np)
            proposal distribution, from which to sample parameters in the first place
            (independently from whether or not `prior` is provided). Default to `prior`.
            The main difference with `prior` is that this distribution is used only 
            to initialize the ensemble, but is "forgotten" in the course of the iterative
            importance sampling procedure (since it is not used in the weights calculation)
        initial_state : 1-D ndarray, optional (size : Ns)
            Passed to func, or simply used to get problem dimensions Default to likelihood.mean()
        default_params : 1-D ndarray, optional (size : Np)
            Passed to func, or simply used to get problem dimensions Default to prior.mean()
        params_bounds : 2-D ndarray (shape: 2 x Np)
            Minimum and Maximum params' bounds
            Resample or scale jitter to remain within these bounds. 
            Rather than providing params_bounds, a better approach is to transform 
            the parameters to that no hard bounds are required. 
        labels_state : sequence of str, optional (len : Ns)
            label states, for plotting 
        labels_params : sequence of str, optional (len : Np)
            label params, for plotting 
        args : tuple, optional
            variable-length arguments for func
        kwargs : keyword arguments for func, optional
        """
        # Check function
        assert callable(func), 'model `func` must be callable'

        # ==============================
        # Check user-input distribution
        # ==============================
        if proposal is None and prior is None:
            raise ValueError("Need to provide either proposal or prior distribution")
        elif proposal is None:
            proposal = prior

        # Check the minimum requirement for the input distributions
        # likelihood need logpdf to compute posterior probability
        try:
            likelihood = MultivariateComposite(likelihood, required_methods=['logpdf'])
        except Exception as error:
            raise ValueError('likelihood: '+error.message)

        # prior need logpdf to compute posterior probability
        if prior is not None:
            try:
                prior = MultivariateComposite(prior, required_methods=['logpdf'])
            except Exception as error:
                raise ValueError('prior: '+error.message)
        # proposal needs 'rvs' to sample from random samples
        try:
            proposal = MultivariateComposite(proposal, required_methods=['rvs'])
        except Exception as error:
            raise ValueError('proposal: '+error.message)

        if prior is not None:
            assert prior.ndim == proposal.ndim, 'prior and proposal distributions are not consistent'

        # ======================================================
        # Check user-input initial state and default parameters
        # ======================================================
        # initial state
        if initial_state is not None:
            initial_state = np.asarray(initial_state)
        else:
            try:
                initial_state = likelihood.mean()
            except:
                try:
                    initial_state = likelihood.rvs(1000).mean(axis=0)
                except:
                    raise ValueError("Could not determine initial model state from mean() or rvs() methods. Please provide initial_state input parameter or define 'mean' or 'rvs' methods")
        # default parameters
        if default_params is not None:
            default_params = np.asarray(default_params)
        else:
            try:
                default_params = proposal.mean()
            except:
                try:
                    default_params = proposal.rvs(1000).mean(axis=0)
                except:
                    raise ValueError("Could not determine default model params from mean() or rvs() methods. Please provide default_params input parameter or define 'mean' or 'rvs' methods")

        if initial_state.ndim != 1:
            raise TypeError('model state must be 1-D, not {}-D'.format(initial_state.ndim))
        if initial_state.size != likelihood.ndim:
            raise ValueError('model state size {} and likelihood function dimension {} do not match'.format(initial_state.size, likelihood.ndim))
        if initial_state.dtype.kind == 'o':
            raise TypeError('needs numerical model state, got object type')

        if default_params.ndim != 1:
            raise TypeError('model params must be 1-D, not {}-D'.format(default_params.ndim))
        if default_params.size != proposal.ndim:
            raise ValueError('model params size {} and proposal function dimension {} do not match'.format(default_params.size, proposal.ndim))
        if default_params.dtype.kind == 'o':
            raise TypeError('needs numerical model params, got object type')

        # args = inspect.getargspec(f)[0]

        # ======================================================
        # Check labels
        # ======================================================
        # labels (remove?)
        if labels_state is None:
            labels_state = ["v{}".format(i) for i in xrange(initial_state.size)]
        if labels_params is None:
            labels_params =["p{}".format(i) for i in xrange(default_params.size)]
        assert len(labels_params) == default_params.size, 'labels_params: invalid size'
        assert len(labels_state) == initial_state.size, 'labels_model: invalid size'

        # ======================================================
        # Assign attributes
        # ======================================================
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self.prior = prior
        self.likelihood = likelihood
        self.proposal = proposal 

        self.state = initial_state
        self.params = default_params
        self.params_bounds = params_bounds

        self.labels_state = labels_state 
        self.labels_params = labels_params


    def integrate(self, params, initial_state=None):
        """ Integrate the model to update state based on parameters

        Parameters
        ----------
        params : parameter array (1d)
        initial_state : state array (1d), optional

        Returns
        -------
        state : model state conditional on parameters and initial state
        """
        kwargs = self.kwargs.copy()
        # if initial_state is not None:
        kwargs['initial_state'] = initial_state
        try:
            return self.func(params, *self.args, **kwargs) # include initial_state argument
        except:
            return self.func(params, *self.args, **self.kwargs) # no 'initial_state'

    def sample_proposal_params(self, size=1):
        """ Sample prior parameter sets
        """
        return self.proposal.rvs(size=size)

    def get_loglik(self, state):
        """ log-likelihood from state variables
        """
        return self.likelihood.logpdf(state)

    def get_logprior(self, params):
        """ prior probability of the params
        """
        if self.prior is None:
            return 0
        else:
            return self.prior.logpdf(params)

class Ensemble(object):
    """ Machinery to help deal with an ensemble of particles
    """
    def __init__(self, model):
        """ Model instance
        """
        assert isinstance(model, Model), 'model must be a `iss.Model` instance'
        self.model = model  # contains methods like integrate_model, etc.
        self._state = None
        self._params = None
        self.weights = None # could be useful at some point (e.g. calculate percentiles, means)

        self.ready = False # integration over
        self.analysis = None  # analysis diagnostic
        self.alpha = 0 # describe convergence state
        self.ancestor_ids = None # ids of ancestors in a previous ensemble, after resampling

    @property
    def size(self):
        assert self.params is not None, "ensemble is not initialized"
        return self.params.shape[0]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        if not isinstance(params, np.ndarray):
            raise TypeError('params should be ndarray !')
        if not params.ndim == 2:
            raise TypeError('params should be 2-D !')
        self._params = params

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if not isinstance(state, np.ndarray):
            raise TypeError('state should be ndarray !')
        if not state.ndim == 2:
            raise TypeError('state should be 2-D !')
        self._state = state

    def sample_proposal_params(self, size):
        self.params = self.model.sample_proposal_params(size)
        self.state = np.repeat(self.model.state[np.newaxis, :], size, axis=0)
        self.ready = False

    def integrate(self, **kwargs):
        assert self.params is not None, 'params uninitialized'

        for i in xrange(self.size):
            try:
                # first try passing state as second argument
                self.state[i] = self.model.integrate(params=self.params[i], initial_state=self.state[i], **kwargs)
            except TypeError:
                self.state[i] = self.model.integrate(params=self.params[i], **kwargs)

        self.ready = True

    def get_loglik(self):
        """ Return log-likelihood based on model state
        """
        loglik = np.empty(self.size)
        for i in xrange(self.size):
            loglik[i] = self.model.get_loglik(state=self.state[i])
        return loglik

    def get_logprior(self):
        """ Return the prior log-probability based on model param
        """
        logprior = np.empty(self.size)
        for i in xrange(self.size):
            logprior[i] = self.model.get_logprior(params=self.params[i])
        return logprior

    def resample(self, weights, size=None, method=None):
        """ resample ensemble according to weights

        Parameters
        ----------
        weights : 1-D array of weights (need not be normalized)
        method : str, optional
            "multinomial" : purely random
            "residual" : partly deterministic (default)
            "stratified" (NOT IMPLEMENTED) : partly deterministic
            "systematic" (NOT IMPLEMENTED) : determinisitc

        ..seealso : iis.resampling

        Returns
        -------
        newensemble : resampled ensemble with updated ancestor_ids attribute
        """
        method = method or SAMPLING_METHOD
        size = size or self.size

        weights /= weights.sum() # normalize weights

        # Sample the model versions according to their weight
        if method == "multinomial":
            ids = multinomial_resampling(weights, size)
        elif method == "residual":
            ids = residual_resampling(weights, size)
        elif method in ("stratified", "deterministic"):
            raise NotImplementedError(method)
        else:
            raise ValueError("Unknown resampling method: "+method)

        ids = np.sort(ids)  # sort indices (has no effect on the results)

        # create new ensemble
        newensemble = Ensemble(self.model)
        newensemble.params = self.params[ids]
        newensemble.state = self.state[ids]
        newensemble.ancestor_ids = ids

        return newensemble

    def add_jitter(self, epsilon, check_bounds=True):
        """ Add noise with variance equal to epsilon times ensemble variance
        (in-place operation)
        """
        covjitter = np.cov(self.params.T)*epsilon
        if covjitter.ndim == 0: 
            covjitter = covjitter.reshape([1,1]) # make it 2-D
        jitter = np.random.multivariate_normal(np.zeros(self.params.shape[1]), covjitter, self.size)
        newparams = self.params + jitter

        # Check that params remain within physically-motivated "hard" bounds:
        if check_bounds and self.model.params_bounds is not None:
            bounds = self.model.params_bounds
            bad = np.any((newparams < bounds[0][np.newaxis, :]) | (newparams > bounds[1][np.newaxis, :]), axis=0)
            ibad = np.where(bad)[0]
            if ibad.size > 0:
                logging.warning("{} particles are out-of-bound after jittering: resample within bounds".format(len(ibad)))
                # newparams[ibad] = resampled_params[ibad]
                for i in ibad:
                    newparams[i] = sample_with_bounds_check(self.params[i], covjitter, bounds)

        self.params = newparams
        self.ready = False

        return covjitter # for the record only

class IIS(object):
    """ Solver for Iterative Importance Sampling
    """
    # @formatdoc(Model.__init__.__doc__)
    def __init__(self, model):
        """ model : `iis.Model` instance
        """
        self.ensemble = Ensemble(model) # define Ensemble
        self.history = []

    def initialize(self, size):
        """ first sampling and model integration
        """
        self.ensemble.sample_proposal_params(size)
        self.ensemble.integrate()

    def iterate(self, iterations=1, **kwargs):
        """ Perform a number of iteratation

        Parameters
        ----------
        iterations : number of iterations to perform
        **kwargs : keyword arguments passed to `analyze` method
        """
        for iteration in xrange(iterations):

            if not self.ensemble.ready:
                logging.warning("Ensemble is not ready, use integrate model before \
doing the analysis to get meaningful results")

            # analysis step
            newensemble, analysis = self.analyze(self.ensemble, **kwargs)

            # run models
            newensemble.integrate()

            # for the record, plotting, etc...
            self.ensemble.analysis = analysis  # update analysis field of the ensemble
            self.history.append(self.ensemble)

            # ensemble ready for analysis
            self.ensemble = newensemble

        return self.ensemble

    @staticmethod
    def analyze(ensemble, epsilon=None, resampling_method=None):
        """ Analyze a model ensemble

        Parameters
        ----------
        ensemble : Ensemble instance to analyze
        epsilon : float, optional
            Exponent to scale the likelihood and avoids ensemble collapse.
            If not provided, will be computed automatically
            ..seealso: `adaptive_posterior_exponent`
        resampling_method : str, optional
            passed to `Ensemble.resample` as "method"

        Returns
        -------
        newensemble : new Ensemble (updated params, not integrated)
        analysis : dict of diagnostics from the analysis
        """

        # Compute the weights as a power law of the posterior
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        loglik = ensemble.get_loglik() # based on model state
        logprior = ensemble.get_logprior() # based on model params
        logposterior = loglik + logprior  # bayes rule

        if epsilon is None:
            epsilon = adaptive_posterior_exponent(logposterior)

        weights = np.exp(logposterior*epsilon)

        # check effective ensemble size
        Neff = _get_Neff(weights)
        if Neff/weights.size < 0.5:
            logging.warning("Effective ensemble < 50%, risk of collapse {}".format(Neff/weights.size))

        newensemble = ensemble.resample(weights, method=resampling_method)

        covjitter = newensemble.add_jitter(epsilon=epsilon)

        # update exponent: ensemble now sample the distribution proposal**(1-alpha) * posterior**alpha
        newensemble.alpha = (ensemble.alpha + epsilon) / (1 + epsilon)

        # make some diagnostics
        analysis = {'weights':weights,'epsilon':epsilon, 'covjitter':covjitter,
                    'Neff':Neff, 'Nsample':len(np.unique(newensemble.ancestor_ids))} # just to keep track of things

        return newensemble, analysis

    @formatdoc(default_size=DEFAULT_SIZE, alpha_target=DEFAULT_ALPHA_TARGET)
    def estimate(self, size=DEFAULT_SIZE, alpha_target=DEFAULT_ALPHA_TARGET, maxiter=None, reset=True, **kwargs):
        """ Estimate ensemble parameters

        Parameters
        ----------
        size : int, optional
            ensemble size to use for the estimation (default: {default_size})
        alpha_target : float, optional
            target value for convergence parameter alpha
            At any time, the ensemble should sample the distribution
            phi = proposal**(1-alpha) * posterior**alpha
            where alpha starts at 0, converges toward 1
            (default : {alpha_target})
        maxiter : int, optional
            max number of iterations
            Alternative criterion to end iterations.
            For epsilon = 0.05, ~65 iterations are needed to reach 0.95
            Set alpha_target to 1 to only rely on maxiter criterion.
            (default : None)
        reset : bool, optional 
            call `IIS.initialize`, this will reset ensemble and alpha value
            (default : True)
        **kwargs : keyword arguments passed to `IIS.analyze`

        Returns
        -------
        ensemble : Ensemble instance of updated parameters and state variables
        """
        if reset:
            self.initialize(size)

        assert alpha_target >= 0, 'alpha_target must be between 0 and 1'
        assert alpha_target < 1 or maxiter is not None, 'if maxiter is not provided, requires alpha_target < 1 to end iteration'

        i = 0
        while True:
            # alpha = 0 if self.ensemble.analysis is None else self.ensemble.analysis['alpha']
            if self.ensemble.alpha > alpha_target:
                logging.info("Estimation over (alpha_target reached).")
                break
            if maxiter is not None and i >= maxiter:
                logging.info("Estimation over (maxiter reached).")
                break
            i += 1
            self.iterate(**kwargs)

        return self.ensemble


# def main():
#     pass
#
# if __name__ == "__main__":
#     main()
