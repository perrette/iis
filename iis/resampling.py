# -*- coding: utf-8 -*-
""" Resample particles according to their weight

References
----------
Douc and Cappe. 2005. Comparison of resampling schemes for particle filtering.
    ISPA2005, Proceedings of the 4th Symposium on Image and Signal Processing.

Hol, Jeroen D., Thomas B. Schön, and Fredrik Gustafsson, 
    "On Resampling Algorithms for Particle Filters", 
    in NSSPW - Nonlinear Statistical Signal Processing Workshop 2006, 
    2006 <http://dx.doi.org/10.1109/NSSPW.2006.4378824>

"""
from __future__ import division
import numpy as np
# from iis.lib.walkerrandomwalk import Walkerrandom

def _build_ids(counts):
    """ make an array of ids from counts, e.g. [3, 0, 1] will returns [0, 0, 0, 2]
    """
    ids = np.empty(counts.sum(), dtype=int)
    start = 0
    for i, count in enumerate(counts):
        ids[start:start+count] = i
        start += count
    return ids

def multinomial_resampling(weights, size):
    """
    weights : (normalized) weights 
    size : sample size to draw from the weights
    """
    counts = np.random.multinomial(size, weights)
    return _build_ids(counts)

def residual_resampling(weights, size):
    """
    Deterministic resampling of the particles for the integer part of the counts
    Random sampling of the residual.
    Each particle (index) is copied int(weights[i]*size) times
    """
    # copy particles
    counts_decimal = weights * size
    counts_copy = np.floor(counts_decimal)
    # sample randomly from residual weights
    weights_resid = counts_decimal - counts_copy
    weights_resid /= weights_resid.sum()
    counts_resid = np.random.multinomial(size - counts_copy.sum(), weights_resid)
    # make the ids
    return _build_ids(counts_copy + counts_resid)

def stratified_resampling(weights, size):
    raise NotImplementedError()

def systematic_resampling(weights, size):
    raise NotImplementedError()

def deterministic_resampling(weights, size):
    """
    Li et al. (2012)
    "Deterministic Resampling: Unbiased Sampling to Avoid Sample Impoverishment in Particle Filters"
    <http://dx.doi.org/10.1016/j.sigpro.2011.12.019>
    """
    raise NotImplementedError()

