""" Make various diagnostic of model convergence and so on
"""
from collections import OrderedDict as odict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # some plots are based on pandas
from pandas import DataFrame, Series
import pandas.tools.plotting as pdplt
# from pandas.tools.plotting import scatter_matrix, andrews_curves, parallel_coordinates, radviz

##########################################
# Wrapper around Pandas plotting functions
##########################################
def ensemble_to_dataframe(ensemble, params_ids=None, state_ids=None, categories=None, class_column='_CatName'):
    """ Transform an Ensemble instance into pandas' DataFrame

    Parameters
    ----------
    ensemble : `iis.Ensemble` instance
    params_ids : list of int, optional
        parameters to include in the df (default to all)
    state_ids : list of int, optional
        model state to include in the df (default to all)
    categories : list of str, optional
        class names to cluster the data (must match ensemble size)
        Will be added as a column according to class_column parameter
    class_column : str, optional
        Column name containing class names, if categories if provided by default '_CatName'

    Returns
    -------
    `pandas.DataFrame` instance
    """
    if params_ids is None:
        params_ids = range(ensemble.model.params.size)
    if state_ids is None:
        state_ids = range(ensemble.model.state.size)
    # aggregate all data
    datasets = []
    labels = []
    for i in params_ids:
        datasets.append(ensemble.params[:, i])
        labels.append(ensemble.model.labels_params[i])
    for i in state_ids:
        datasets.append(ensemble.state[:, i])
        labels.append(ensemble.model.labels_state[i])

    df = DataFrame(np.asarray(datasets).T, columns=labels)

    # add categories
    if categories is not None:
        assert len(categories) == ensemble.size, 'categories must match ensemble size'
        # idx_check = np.arange(len(datasets))
        cat = DataFrame({class_column:Series(categories)})
        df = df.join(cat)

    return df

##########################################
# Plot convergence of the ensemble
##########################################
def history_to_panel(ensembles, params_ids=None, state_ids=None, quantiles=None, trace_ids=None):
    """Convert a list of ensembles into a `pandas.panel` object
    
    Parameters
    ----------
    ensembles : list of Ensemble instances (iterations)
    params_ids : indexing array, optional
        model parameters to include (default to all)
    state_ids : indexing array, optional
        model state variables to include (default to all)
    quantiles : list of floats in [0,1], optional 
        compute quantiles instead of full ensemble
    trace_ids : indices or bool, optional 
        Trace-back final parameters and model variables following their ancestor 
        particle. This is the most straightforward way of understanding the role
        of jittering on a given parameter or state variable.
        ..seealso :: `traceback_ids` (to build the matrix of indices)

    Returns
    -------
    panel : `pandas.Panel` instance (iterations x {samples or quantiles} x variables)
    """
    if trace_ids is not None and trace_ids is not False:
        trace_ids = traceback_ids(ensembles, None if trace_ids is True else trace_ids)

    n = len(ensembles)
    snapshots = []
    for i, ensemble in enumerate(ensembles):
        df = ensemble_to_dataframe(ensemble, params_ids, state_ids)
        if trace_ids is not None:
            df = df.iloc[trace_ids[i]]
        if quantiles is not None:
            df = df.quantile(quantiles)
            df.index.name = "quantile"
        snapshots.append(df)
    return pd.Panel({i:k for i, k in enumerate(snapshots)})

def traceback_ids(ensembles, ids=None):
    """ Create 2-D array of indices (iteration x samples)
    to trace-back particles, starting from the last iteration.

    ensembles : list of Ensemble instances
    ids : indices, optional
        restrict trace back to those indices
    """
    n = len(ensembles)
    # start from somewhere (all or a few ids)
    if ids is None:
        size = ensembles[-1].size
        ids = np.arange(size, dtype=int)
    else:
        size = np.size(ids)
    # build 2-D array of ids
    trace_ids = np.empty((n, size), dtype=int)
    trace_ids[-1] = ids
    i = n - 1
    while i > 0:
        trace_ids[i-1] = ensembles[i].ancestor_ids[trace_ids[i]]
        i -= 1
    return trace_ids

def plot_history(ensembles, params_ids=None, state_ids=None, quantiles=(0.5, 0.95, 0.05, 0.16, 0.84), overlay_dists=True, **kwargs):
    """ Plot convergence of the ensemble

    Parameters
    ----------
    ensembles : list of Ensemble instances (iterations)
    params_ids : indexing array, optional
        model parameters to include (default to all)
    state_ids : indexing array, optional
        model state variables to include (default to all)
    quantiles : list of floats in [0,1], optional 
        (default is [0.5, 0.95, 0.05, 0.16, 0.84] )
    overlay_dists : bool, optional
        if True, overlay prior and likelihood distributions onto the convergence plots
        Default to False.
    """
    panel = history_to_panel(ensembles, params_ids, state_ids, quantiles)

    # add distributions?
    if overlay_dists:
        dists = dist_quantiles(ensembles[-1].model, params_ids, state_ids, quantiles)
    else:
        dists = None

    # over = self.model.likelihood.ppf(np.array(pct)/100.) # quantiles
    return _plot_history(panel, dists=dists, **kwargs)

def _plot_history(panel, dists=None, alpha=0.3, color='blue', linestyle='-', linewidth=2, **kwargs):
    """ Plot convergence of a series

    Parameters
    ----------
    panel : pandas.Panel obtained via `history_to_panel` or `IIS.to_panel`
        Shape iterations x quantiles x variables
        where quantiles if has an odd number of elements, starting with 
        the median and followed by pairs of concentric confidence interfals
        e.g. [0.5, 0.05, 0.95, 0.16, 0.84]
    dists : list of quantiles to overlay onto the figure (e.g. prior for params, likelihood for model state)
    """
    assert isinstance(panel, pd.Panel), 'must be pandas Panel'
    ni, nq, nv = panel.shape
    assert np.mod(nq, 2) == 1, 'must be an odd number of quantiles starting with the median'

    # iterations
    x = panel.axes[0].values
    q = panel.axes[1].values
    v = panel.axes[2].values

    assert q[0] == 0.5, 'first quantile must be the median'

    fig, axes = plt.subplots(nv, 1, sharex=True)
    for i, ax in enumerate(axes):
        data = panel.iloc[:,:,i].values.T

        # plot the intervals
        for j in range(1, nq, 2):
            key = "{}%".format((q[j+1]-q[j])*100)
            ax.fill_between(x, data[:, j], data[:, j+1], alpha=alpha, color=color, **kwargs)
        # plot the median
        ax.plot(x, data[:, 0], color=color, linestyle=linestyle, linewidth=linewidth)

        # overlay target distribution
        if dists is not None and dists[i] is not None:
            mi, ma = 0, ni-1
            over = dists[i]
            ax.hlines(over[0], mi, ma, linestyle='-', color='red')
            ax.hlines(over[[1,2]], mi, ma, linestyle=':', color='red')
            ax.hlines(over[[3,4]], mi, ma, linestyle='--', color='red')
            # plt.legend(frameon=False)

        # add labels
        ax.set_ylabel(v[i])
        ax.grid()

    ax.set_xlabel("Iteration number")
    ax.set_xlim([0, ni-1])
    # plt.ylabel("")

    fig.tight_layout()

    return axes #, handles

def dist_quantiles(model, params_ids, state_ids, quantiles):
    """ target quantiles to over lay on convergence plots

    # parameters : prior distribution
    # state variables : likelihood
    """
    if params_ids is None:
        params_ids = range(model.params.size)
    if state_ids is None:
        state_ids = range(model.state.size)

    dists = []
    if model.prior is None:
        dists.extend( [None] * len(params_id) )
    else:
        params_q = model.prior.ppf(quantiles)
        for sq in params_q.T:
            dists.append(sq)

    state_q = model.likelihood.ppf(quantiles)
    for sq in state_q.T:
        dists.append(sq)

    return dists

def plot_history_diag(ensembles):
    """ plot things like epsilon and so on
    """
    epsilon = np.array([e.analysis['epsilon'] for e in ensembles])
    neff = np.array([e.analysis['Neff'] for e in ensembles])
    nsample = np.array([e.analysis['Nsample'] for e in ensembles])
    alpha = np.array([e.alpha for e in ensembles])

    fig, axes = plt.subplots(3,1,sharex=True)
    ax = axes[0]
    ax.plot(neff, label="Neff")
    ax.plot(nsample, label="Nsample")
    ax.legend(frameon=False)
    ax.set_title("Convergence diagnostics")
    ax = axes[1]
    ax.plot(epsilon, label='epsilon')
    ax.legend(frameon=False)
    ax.set_ylabel('scaling factor epsilon')
    ax = axes[2]
    ax.plot(alpha, label='alpha')
    # ax.plot(alpha, label='beta')
    ax.set_ylabel('exponents')
    ax.set_xlabel('iterations')
    fig.tight_layout()
    return axes

def plot_particles_trace(ensembles, ids=None):
    """ Plot particle history
    """
    traceids = traceback_ids(ensembles, ids=None)
    plt.figure()
    lines = plt.plot(traceids)
    plt.title("Particle history")
    plt.xlabel("iteration number")
    plt.ylabel("id in the ensemble")
    return lines

########################################################
# Plot PDF and CDF with overlaids likelihood and prior
########################################################
def plot_distribution(ensemble, field="state", dim=0):
    """ Plot ensemble distributions

    Parameters
    ----------
    ensemble : Ensemble instance
    field : str, optional
        "state" (default) or "params" 
    dim : int, optional
        parameter to plot, can be > 0 for multivariate distribution
        by default 0
    """
    fig, axes = plt.subplots(1, 2, sharex=True)
    bins = 10

    variable = getattr(ensemble, field)[:, dim]
    model = ensemble.model

    mi, ma = np.min(variable), np.max(variable)
    x = np.linspace(mi, ma, 1000)

    if field == 'state':
        dist = model.likelihood.dists[dim]
        label = 'observations'
    else:
        if model.prior is not None:
            dist = model.prior.dists[dim]
            label = 'prior'
        else:
            dist = None

    # PDF
    ax = axes[0]
    ax.hist(variable, bins, histtype='step', label='result', normed=True)
    if dist is not None:
        ax.plot(x, dist.pdf(x), label=label)

    ax.legend(frameon=False)
    ax.set_title("Histogram (empirical PDF)")

    # CDF
    ax = axes[1]
    ax.hist(variable, bins, histtype='step', label='result', normed=True, cumulative=True)
    if dist is not None:
        ax.plot(x, dist.cdf(x), label=label)
    ax.legend(frameon=False)
    ax.set_title("Empirical cumulative distribution function (CDF))")

    # fig.set_figwidth(17)
    ax.set_xlim([mi, ma])


# ##########################################
# # A class as wrapper for all that
# ##########################################
# class Diagnostic(object):
#     """ Class to make diagnostics from the historiy field for the Solver class
#     """
#     def __init__(self, history):
#         assert len(history) > 0, 'no iterations'
#         self.history = history
#
#         # contains distributions
#         self.model = history[0].model
#
#         # labels for parameters and model state variables
#         self.labels_state = self.model.labels_state
#         self.labels_params = self.model.labels_params
#
#     # plot distribution of final state
#     def plot_distribution(self, field="state", dim=0, i=-1):
#         return plot_distribution(self.history[i], field, dim)
#
#     def scatter_matrix(self, i=-1, **kwargs):
#         return scatter_matrix(self.history[i], **kwargs)
#
#     def parallel_coordinates(self, categories, i=-1, **kwargs):
#         return parallel_coordinates(self.history[i], categories, **kwargs)
#
#     def radviz(self, categories, i=-1, **kwargs):
#         return radviz(self.history[i], categories, **kwargs)
#
#     def andrews_curves(self, categories, i=-1, **kwargs):
#         return andrews_curves(self.history[i], categories, **kwargs)
#
#     # plot time series
#     def get_dimensions(self):
#         """ dimensions of the problem (i, n, p, m)
#         i: number of iterations
#         n: ensemble size (number of runs)
#         p: params size (number of model parameters)
#         m: model size (number of model's state variables)
#         """
#         ens0 = self.history[0]
#         return (len(self.history), ens0.size, self.model.params.size, self.model.state.size)
#
#     def get_params_q(self, q):
#         " series of params' quantiles: param x iter x quantiles "
#         ni, n, p, m = self.get_dimensions()
#         pcthist = np.empty((p, ni, len(q))) 
#         for i, ens in enumerate(self.history):
#             pcthist[:,i] = np.percentile(ens.params, np.asarray(q)*100, axis=0, keepdims=True).T
#         return pcthist 
#
#     def get_state_q(self, q):
#         " series of state variables' quantiles: var x iter x quantiles "
#         ni, n, p, m = self.get_dimensions()
#         pcthist = np.empty((m, ni, len(q))) 
#         for i, ens in enumerate(self.history):
#             pcthist[:,i,:] = np.percentile(ens.state, np.asarray(q)*100, axis=0, keepdims=True).T
#         return pcthist 
#
#     def plot_series_quantiles(self, field="state", dim=0, ax=None):
#         """ field : "state" or "params"
#         """
#         ax = ax or plt.gca()
#         ensembles = self.history
#
#         pct = [50, 5, 95, 16, 84]
#         n = len(ensembles)
#         data = np.empty((n, len(pct)))
#
#         for i, ens in enumerate(ensembles):
#             data[i] = np.percentile(getattr(ens, field)[:, dim], pct)
#             
#         x = np.arange(n)
#         f90 = ax.fill_between(x, data[:, 1], data[:, 2], alpha=0.2)
#         f67 = ax.fill_between(x, data[:, 3], data[:, 4], alpha=0.5)
#         l50 = ax.plot(x, data[:, 0], label="model")
#
#         ax.set_xlabel("Iteration number")
#         ax.set_ylabel("{}[{}]".format(field, dim))
#         # plt.ylabel("")
#
#         return {'median':l50, '67%':f67, '90%':f90, 'pct':pct}
#
#     def plot_series_state(self):
#         ni, n, p, m = self.get_dimensions()
#         fig, axes = plt.subplots(m, 1, sharex=True, squeeze=False)
#
#         for i, ax in enumerate(axes.flatten()):
#             # extract and plot quantiles
#             h = self.plot_series_quantiles('state', dim=i, ax=ax)
#             pct = h['pct']
#
#             # overlay likelihood
#             mi, ma = ax.get_xlim()
#             over = self.model.likelihood.ppf(np.array(pct)/100.) # quantiles
#             plt.hlines(over[0, i], mi, ma, linestyle='-', label='likelihood', color='red')
#             plt.hlines(over[[1,2], i], mi, ma, linestyle=':', color='red')
#             plt.hlines(over[[3,4], i], mi, ma, linestyle='--', color='red')
#             plt.xlim([mi, ma])
#             plt.legend(frameon=False)
#
#     def plot_series_params(self):
#         ni, n, p, m = self.get_dimensions()
#         fig, axes = plt.subplots(p, 1, sharex=True, squeeze=False)
#
#         for i, ax in enumerate(axes.flatten()):
#             # extract and plot quantiles
#             h = self.plot_series_quantiles('params', dim=i, ax=ax)
#             pct = h['pct']
#
#             # overlay prior, if any
#             if self.model.prior:
#                 # mi, ma = ax.get_xlim()
#                 mi, ma = 0, len(self.history)
#                 over = self.model.prior.ppf(np.array(pct)/100.) # quantiles
#                 ax.hlines(over[0, i], mi, ma, linestyle='-', label='prior', color='red')
#                 ax.hlines(over[[1,2], i], mi, ma, linestyle=':', color='red')
#                 ax.hlines(over[[3,4], i], mi, ma, linestyle='--', color='red')
#                 ax.set_xlim([mi, ma])
#             ax.legend(frameon=False)
#
#     def traceback_ids(self, target_iter=-1):
#         """ For a given iteration step, trace back the history 
#         of each particle in the ensemble. Since a cloud of particles
#         is created at each iteration step (from jittering) from resampled
#         particles, this makes a kind of branching which only works backward
#         (e.g. from a leave to the trunk)
#
#         Parameters
#         ----------
#         target_iter : iteration number to start from (default to last)
#
#         Returns
#         -------
#         ids : 2-d integer numpy array (iter x ensemble)
#             e.g. ids[0] refers to the first particle at iteration
#             number i=target_iter. It is an array that indicates its index
#             in each previous ensemble.
#         """
#         ni, n, p, m = self.get_dimensions()
#         try:
#             target_iter = np.arange(ni)[target_iter]
#         except:
#             raise ValueError("{} is outside iteration bounds [{}, {}]".format(target_iter, 0, ni-1))
#         i = target_iter
#         ids = np.empty((i+1, n), dtype=int)
#         ids[i] = np.arange(n, dtype=int)
#         while i > 0:
#             ids[i-1] = self.history[i].ancestor_ids[ids[i]]
#             i -= 1
#         return ids
#
#     def plot_traceback(self, target_iter=-1, **kwargs):
#         """ Plot particle history, starting at target_iter
#         """
#         ids = self.traceback_ids(target_iter)
#         plt.figure()
#         _ = plt.plot(ids)
#         plt.title("Particle history from target_iter={}".format(target_iter))
#         plt.xlabel("iteration number")
#         plt.ylabel("id in the ensemble")
#         return _
#
    # def get_params(self, param_ids=None, trace_ids=None):
    #     """Series of params
    #
    #     Parameters
    #     ----------
    #     param_ids: 1-D int array, optional
    #         subset of parameters to retrieve (default: all)
    #     trace_ids: 2-D int array, optional ( iter x ids )
    #         mapping of the particle at the last iteration (or before) 
    #         onto the ensembles of previous iterations.
    #         See traceback_ids.
    #
    #     Returns
    #     -------
    #     3-D ndarray of params ( param x iter x ensemble)
    #     """
    #     ni, n, p, m = self.get_dimensions()
    #     if param_ids is None: # params' index
    #         param_ids = np.arange(p, dtype=int)
    #     if trace_ids is None: # ensemble's index
    #         # by default just assumes the particles remain the same...
    #         trace_ids = np.arange(n, dtype=int)[np.newaxis, :].repeat(ni, axis=1)
    #     params = np.empty((len(param_ids), trace_ids.shape[0], trace_ids.shape[1])) 
    #     for i in xrange(trace_ids.shape[0]):
    #         params[:,i,:] = self.history[i].params[trace_ids[i], param_ids].T
    #     return params
    #
    #
    # def plot_traceback_param(self, param_id=None, model_id=None, target_iter=-1, **kwargs):
    #     ni, n, p, m = self.get_dimensions()
    #     # param(s) to trace back
    #     if param_id is None:
    #         p_ids = np.arange(p)
    #     else:
    #         p_ids = [param_id]
    #     # model(s) to trace back
    #     if model_id is None:
    #         m_ids = np.arange(n)
    #     elif np.iterable(model_id):
    #         m_ids = model_id
    #     else:
    #         m_ids = [model_id]
    #     ids = self.traceback_ids(target_iter)
    #     params = self.get_params(trace_ids=ids, param_ids=p_ids)
    #     # 3-D ndarray of params ( param x iter x ensemble)
    #
    #     figlines = []
    #     
    #     for pid in p_ids:
    #         plt.figure()
    #         lines = []
    #         for m in m_ids:
    #             line = plt.plot(params[pid][:, m], label=m, **kwargs)
    #             lines.append(line)
    #         figlines.append(lines)
    #         plt.title("Particle history w.r.t to param {}".format(self.labels_params[pid]))
    #         plt.xlabel("Iteration number")
    #         plt.ylabel("Param value (?)")
    #         if len(m_ids) <= 10:
    #             plt.legend(frameon=False)
    #
    #     return figlines

