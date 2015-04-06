IIS
===
Iterative Importance Sampling (IIS) for parameter estimation of physical models
in a Bayesian sense. The current version of this package is based on Annan and 
Hargreave (2010), DOI: 10.1016/j.ocemod.2010.02.003

This module has some bearing with pymc but focus on relatively computationally-
expensive physical models. It is based on Gaussian approximation of parameter
distribution and resulting model state, but is more robust than the Ensemble 
Kalman Filter with respect to deviations from non-linearity (Annan and Hargreave, 2010)
In the best cases, it allows convergence of a 50 to 500-member model ensemble
within a few 10s of iterations (see notebook examples). 
This is much faster than classical Monte Carlo Markov Chains, 
which are more general (not limited to Gaussian cases) but require 10,000s of 
iterations. Even without perfect, steady convergence of the posterior PDF 
(say, non-linear model with ensemble limited to 50 members), 
the IIS method can help "tuning" the model ensemble to within the range of 
observations, which is not always a trivial task "by hand".

Define some model to estimate.

- a forward function to integrate the model, here 2-param into scalar:

        def mymodel(params):
            """User-defined model with two parameters

            Parameters
            ----------
            params : 1-D numpy.ndarray of size 2

            Returns
            -------
            state : float
                return value
            """
            return params[0] + params[1]*2

- distributions that represent prior knowledge on model parameter
  and likelihood functions, using `scipy.stats` distributions:

        from scipy.stats import norm, uniform
        likelihood = norm(loc=1, scale=1)  # normal, univariate distribution mean 1, s.d. 1
        prior = [norm(loc=0, scale=10), uniform(loc=-10, scale=20)] 

- Use `iis` to estimate parameters and state:

        from iis import IIS, Model
        model = Model(mymodel, likelihood, prior=prior)  # define the model 
        solver = IIS(model)
        ensemble = solver.estimate(size=500)
    
- Plotting functions to check convergence

        from iis.diagnostic import Diagnostic, scatter_matrix

        # distribution and correlation of results
        scatter_matrix(ensemble)

        # check convergence history
        diag = Diagnostic(solver.history)
        diag.plot_series_state()

Check in-line help for more option on `iis.IIS.estimate`, `iis.Model` and so on.


Install
=======
From within the cloned repository:

        python setup.py install
