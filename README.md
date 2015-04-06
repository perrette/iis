IIS
===
Iterative Importance Sampling (IIS) for bayesian parameter estimation of physical models

Background
----------
This module has some bearing with [pymc](https://github.com/pymc-devs/pymc) 
but focuses on relatively computationally-
expensive physical models. It is based on Gaussian approximation of parameter
distribution and resulting model state, but is more robust than the Ensemble 
Kalman Filter with respect to deviations from non-linearity ([Annan and Hargreave, 2010](http://doi.org/10.1016/j.ocemod.2010.02.003))
In the best cases, it allows convergence of a 50 to 500-member model ensemble
within a few 10s of iterations (e.g. notebooks  [here](http://nbviewer.ipython.org/github/perrette/iis/blob/master/notebooks/iis_concept.ipynb)
and [there](http://nbviewer.ipython.org/github/perrette/iis/blob/master/notebooks/examples.ipynb)). 
This is much faster than classical Monte Carlo Markov Chains, 
which are more general (not limited to Gaussian cases) but require 10,000s of 
iterations. Even without perfect, steady convergence of the posterior PDF 
(say, non-linear model with ensemble limited to 50 members), 
the IIS method can help "tuning" the model ensemble to within the range of 
observations, which is not always a trivial task "by hand".

The concept is explained in more details [as a notebook](http://nbviewer.ipython.org/github/perrette/iis/blob/master/notebooks/iis_concept.ipynb) and in the paper from [Annan and Hargreaves (2010)](http://doi.org/10.1016/j.ocemod.2010.02.003), from which this package took its inspiration from.

Getting started
---------------

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
    
- A number of methods to analyze the results

        ensemble.to_dataframe()     # convert to pandas dataframe
        solver.to_panel()           # convert to pandas panel
        solver.plot_history()
        ensemble.scatter_matrix()


Check in-line help for more option on `iis.IIS.estimate`, `iis.Model` and so on as well as [notebook examples](http://nbviewer.ipython.org/github/perrette/iis/blob/master/notebooks/examples.ipynb)


Dependencies
------------

- Required:
 
        - numpy (tested with 1.9.2) 
        - scipy (tested with 0.15.1)

- Optional:

        - pandas (plotting only) (tested with 0.15.2)

Install
-------

python setup.py install
