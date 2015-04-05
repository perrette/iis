""" Make various diagnostic of model convergence and so on
"""
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_distribution', 'Diagnostic']

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

class Diagnostic(object):
    """ Class to make diagnostics from the historiy field for the Solver class
    """
    def __init__(self, history):
        assert len(history) > 0, 'no iterations'
        self.history = history

        # contains distributions
        self.model = history[0].model

        # labels for parameters and model state variables
        self.labels_state = self.model.labels_state
        self.labels_params = self.model.labels_params

    def get_dimensions(self):
        """ dimensions of the problem (i, n, p, m)
        i: number of iterations
        n: ensemble size (number of runs)
        p: params size (number of model parameters)
        m: model size (number of model's state variables)
        """
        ens0 = self.history[0]
        return (len(self.history), ens0.size, self.model.params.size, self.model.state.size)

    def get_params(self, param_ids=None, trace_ids=None):
        """Series of params

        Parameters
        ----------
        param_ids: 1-D int array, optional
            subset of parameters to retrieve (default: all)
        trace_ids: 2-D int array, optional ( iter x ids )
            mapping of the particle at the last iteration (or before) 
            onto the ensembles of previous iterations.
            See traceback_ids.

        Returns
        -------
        3-D ndarray of params ( param x iter x ensemble)
        """
        ni, n, p, m = self.get_dimensions()
        if param_ids is None: # params' index
            param_ids = np.arange(p, dtype=int)
        if trace_ids is None: # ensemble's index
            # by default just assumes the particles remain the same...
            trace_ids = np.arange(n, dtype=int)[np.newaxis, :].repeat(ni, axis=1)
        params = np.empty((len(param_ids), trace_ids.shape[0], trace_ids.shape[1])) 
        for i in xrange(trace_ids.shape[0]):
            params[:,i,:] = self.history[i].params[trace_ids[i], param_ids].T
        return params

    def get_params_q(self, q):
        " series of params' quantiles: param x iter x quantiles "
        ni, n, p, m = self.get_dimensions()
        pcthist = np.empty((p, ni, len(q))) 
        for i, ens in enumerate(self.history):
            pcthist[:,i] = np.percentile(ens.params, np.asarray(q)*100, axis=0, keepdims=True).T
        return pcthist 

    def get_state_q(self, q):
        " series of state variables' quantiles: var x iter x quantiles "
        ni, n, p, m = self.get_dimensions()
        pcthist = np.empty((m, ni, len(q))) 
        for i, ens in enumerate(self.history):
            pcthist[:,i,:] = np.percentile(ens.state, np.asarray(q)*100, axis=0, keepdims=True).T
        return pcthist 

    def plot_series_quantiles(self, field="state", dim=0, ax=None):
        """ field : "state" or "params"
        """
        ax = ax or plt.gca()
        ensembles = self.history

        pct = [50, 5, 95, 16, 84]
        n = len(ensembles)
        data = np.empty((n, len(pct)))

        for i, ens in enumerate(ensembles):
            data[i] = np.percentile(getattr(ens, field)[:, dim], pct)
            
        x = np.arange(n)
        f90 = plt.fill_between(x, data[:, 1], data[:, 2], alpha=0.2)
        f67 = plt.fill_between(x, data[:, 3], data[:, 4], alpha=0.5)
        l50 = plt.plot(x, data[:, 0], label="model")

        plt.xlabel("Iteration number")
        plt.ylabel("{}[{}]".format(field, dim))
        # plt.ylabel("")

        return {'median':l50, '67%':f67, '90%':f90, 'pct':pct}

    def plot_series_state(self):
        ni, n, p, m = self.get_dimensions()
        fig, axes = plt.subplots(m, 1, sharex=True, squeeze=False)

        for i, ax in enumerate(axes.flatten()):
            # extract and plot quantiles
            h = self.plot_series_quantiles('state', dim=i, ax=ax)
            pct = h['pct']

            # overlay likelihood
            mi, ma = ax.get_xlim()
            over = self.model.likelihood.ppf(np.array(pct)/100.) # quantiles
            plt.hlines(over[0, i], mi, ma, linestyle='-', label='likelihood', color='red')
            plt.hlines(over[[1,2], i], mi, ma, linestyle=':', color='red')
            plt.hlines(over[[3,4], i], mi, ma, linestyle='--', color='red')
            plt.xlim([mi, ma])
            plt.legend(frameon=False)

    def plot_series_params(self):
        ni, n, p, m = self.get_dimensions()
        fig, axes = plt.subplots(p, 1, sharex=True, squeeze=False)

        for i, ax in enumerate(axes.flatten()):
            # extract and plot quantiles
            h = self.plot_series_quantiles('params', dim=i, ax=ax)
            pct = h['pct']

            # overlay prior, if any
            if self.model.prior:
                mi, ma = ax.get_xlim()
                over = self.model.prior.ppf(np.array(pct)/100.) # quantiles
                plt.hlines(over[0, i], mi, ma, linestyle='-', label='prior', color='red')
                plt.hlines(over[[1,2], i], mi, ma, linestyle=':', color='red')
                plt.hlines(over[[3,4], i], mi, ma, linestyle='--', color='red')
                plt.xlim([mi, ma])
            plt.legend(frameon=False)

    def plot_distribution(self, field="state", dim=0, i=-1):
        return plot_distribution(self.history[i], field, dim)

    def plot_series_analysis(self):
        """ plot things like epsilon and so on
        """
        epsilon = np.array([e.analysis['epsilon'] for e in self.history])
        neff = np.array([e.analysis['Neff'] for e in self.history])
        nsample = np.array([e.analysis['Nsample'] for e in self.history])
        alpha = np.array([e.alpha for e in self.history])

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

    def traceback_ids(self, target_iter=-1):
        """ For a given iteration step, trace back the history 
        of each particle in the ensemble. Since a cloud of particles
        is created at each iteration step (from jittering) from resampled
        particles, this makes a kind of branching which only works backward
        (e.g. from a leave to the trunk)

        Parameters
        ----------
        target_iter : iteration number to start from (default to last)

        Returns
        -------
        ids : 2-d integer numpy array (iter x ensemble)
            e.g. ids[0] refers to the first particle at iteration
            number i=target_iter. It is an array that indicates its index
            in each previous ensemble.
        """
        ni, n, p, m = self.get_dimensions()
        try:
            target_iter = np.arange(ni)[target_iter]
        except:
            raise ValueError("{} is outside iteration bounds [{}, {}]".format(target_iter, 0, ni-1))
        i = target_iter
        ids = np.empty((i+1, n), dtype=int)
        ids[i] = np.arange(n, dtype=int)
        while i > 0:
            ids[i-1] = self.history[i].ancestor_ids[ids[i]]
            i -= 1
        return ids

    def plot_traceback(self, target_iter=-1, **kwargs):
        """ Plot particle history, starting at target_iter
        """
        ids = self.traceback_ids(target_iter)
        plt.figure()
        _ = plt.plot(ids)
        plt.title("Particle history from target_iter={}".format(target_iter))
        plt.xlabel("iteration number")
        plt.ylabel("id in the ensemble")
        return _

    def plot_traceback_param(self, param_id=None, model_id=None, target_iter=-1, **kwargs):
        ni, n, p, m = self.get_dimensions()
        # param(s) to trace back
        if param_id is None:
            p_ids = np.arange(p)
        else:
            p_ids = [param_id]
        # model(s) to trace back
        if model_id is None:
            m_ids = np.arange(n)
        elif np.iterable(model_id):
            m_ids = model_id
        else:
            m_ids = [model_id]
        ids = self.traceback_ids(target_iter)
        params = self.get_params(trace_ids=ids, param_ids=p_ids)
        # 3-D ndarray of params ( param x iter x ensemble)

        figlines = []
        
        for pid in p_ids:
            plt.figure()
            lines = []
            for m in m_ids:
                line = plt.plot(params[pid][:, m], label=m, **kwargs)
                lines.append(line)
            figlines.append(lines)
            plt.title("Particle history w.r.t to param {}".format(self.labels_params[pid]))
            plt.xlabel("Iteration number")
            plt.ylabel("Param value (?)")
            if len(m_ids) <= 10:
                plt.legend(frameon=False)

        return figlines
