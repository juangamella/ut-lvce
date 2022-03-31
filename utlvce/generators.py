# Copyright 2020 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
The :mod:`utlvce.generators` module contains functions to generate random UT-LVCE models from given or random DAG adjacencies.
"""


import numpy as np
from utlvce.model import Model
import utlvce.utils as utils

# --------------------------------------------------------------------
# Functions to generate random models


def chain_graph_model(p,
                      I,
                      num_latent,
                      e,
                      var_lo,
                      var_hi,
                      int_var_lo,
                      int_var_hi,
                      psi_lo,
                      psi_hi,
                      int_psi_lo,
                      int_psi_hi,
                      B_lo,
                      B_hi,
                      sparse_latents=False,
                      obs=True,
                      random_state=42,
                      verbose=0):
    """Generate a random model from a chain graph with `p` nodes.

    Parameters
    ----------
    p : int
        The number of observed variables in the model.
    I : set
        The set of intervention targets.
    num_latent : int
        The number of latent variables in the model.
    e : int
        The number of environments.
    var_lo : float
        The lower bound for the variances of the noise terms of the
        observed variables.
    var_hi : float
        The upper bound for the variances of the noise terms of the
        observed variables.
    int_var_lo : float
        The lower bound for the intervention variances on the observed
        variables.
    int_var_hi : float
        The upper bound for the intervention variances on the observed
        variables.
    psi_lo : float
        The lower bound for the variances of the latent variables.
    psi_hi : float
        The upper bound for the variances of the latent variables.
    int_psi_lo : float
        The lower bound for the intervention variances on the latent
        variables.
    int_psi_hi : float
        The upper bound for the intervention variances on the latent
        variables.
    B_lo : float
        The lower bound for the edge weights between observed
        variables.
    B_hi : float
        The upper bound for the edge weights between observed
        variables.
    sparse_latents : bool, default=False
        If the gamma matrix of latent effects should be sparse (see
        source).
    obs : bool, default=True
        Whether the first environment should be "observational",
        i.e. that the variances of the noise terms and latents are
        lower (variable-wise) than the other environments. With
        `obs=True`, the variances for first environment are sampled
        from `[var_lo, var_hi]` and, from `[var_lo + int_var_lo,
        var_hi + int_var_hi]` for the remaining environments; the same
        holds for the sampling of `psi`. If `obs=False`, the latter
        interval is used for all environments. Note that is not a
        necessary assumption for the UT-LVCE estimator, but makes the
        actual intervention strength less sensitive to the random
        sampling of parameters.
    random_state : int, default=42
        To set the random state for reproducibility. Successive calls
        with the same random state will return the same model.
    verbose: int, default = 0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    model : utlvce.model.Model
        An instance of the model with the sampled parameters.

    Raises
    ------
    ValueError :
        If the intervention targets are not a subset of the variable
        indices, i.e. `[0,...,p-1]`.

    Examples
    --------
    >>> chain_graph_model(20,{2},2,5,0.5,0.6,3,6,0.2,0.4,1,5,0.7,0.8,False,True,42,0) #doctest: +ELLIPSIS
    <utlvce.model.Model object at 0x...>
    """
    A = utils.chain_graph(p)
    return sample_parameters(A,
                             I,
                             num_latent,
                             e,
                             var_lo,
                             var_hi,
                             int_var_lo,
                             int_var_hi,
                             psi_lo,
                             psi_hi,
                             int_psi_lo,
                             int_psi_hi,
                             B_lo,
                             B_hi,
                             sparse_latents=sparse_latents,
                             obs=True,
                             random_state=42,
                             verbose=0)


def random_graph_model(p,
                       k,
                       I,
                       num_latent,
                       e,
                       var_lo,
                       var_hi,
                       int_var_lo,
                       int_var_hi,
                       psi_lo,
                       psi_hi,
                       int_psi_lo,
                       int_psi_hi,
                       B_lo,
                       B_hi,
                       sparse_latents=False,
                       obs=True,
                       random_state=42,
                       verbose=0):
    """Generate a random model from a random Erdős–Rényi graph with `p`
    nodes and average degree `k`.

    Parameters
    ----------
    p : int
        The number of observed variables in the model.
    k : float
        The average degree of the underlying Erdős–Rényi graph.
    I : set
        The set of intervention targets.
    num_latent : int
        The number of latent variables in the model.
    e : int
        The number of environments.
    var_lo : float
        The lower bound for the variances of the noise terms of the
        observed variables.
    var_hi : float
        The upper bound for the variances of the noise terms of the
        observed variables.
    int_var_lo : float
        The lower bound for the intervention variances on the observed
        variables.
    int_var_hi : float
        The upper bound for the intervention variances on the observed
        variables.
    psi_lo : float
        The lower bound for the variances of the latent variables.
    psi_hi : float
        The upper bound for the variances of the latent variables.
    int_psi_lo : float
        The lower bound for the intervention variances on the latent
        variables.
    int_psi_hi : float
        The upper bound for the intervention variances on the latent
        variables.
    B_lo : float
        The lower bound for the edge weights between observed
        variables.
    B_hi : float
        The upper bound for the edge weights between observed
        variables.
    sparse_latents : bool, default=False
        If the gamma matrix of latent effects should be sparse (see
        source).
    obs : bool, default=True
        Whether the first environment should be "observational",
        i.e. that the variances of the noise terms and latents are
        lower (variable-wise) than the other environments. With
        `obs=True`, the variances for first environment are sampled
        from `[var_lo, var_hi]` and, from `[var_lo + int_var_lo,
        var_hi + int_var_hi]` for the remaining environments; the same
        holds for the sampling of `psi`. If `obs=False`, the latter
        interval is used for all environments. Note that is not a
        necessary assumption for the UT-LVCE estimator, but makes the
        actual intervention strength less sensitive to the random
        sampling of parameters.
    random_state : int, default=42
        To set the random state for reproducibility. Successive calls
        with the same random state will return the same model.
    verbose: int, default = 0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    model : utlvce.model.Model
        An instance of the model with the sampled parameters.

    Raises
    ------
    ValueError :
        If the intervention targets are not a subset of the variable
        indices, i.e. `[0,...,p-1]`.

    Examples
    --------
    >>> random_graph_model(20,2.1,{2},2,5,0.5,0.6,3,6,0.2,0.4,1,5,0.7,0.8,False,True,42,0) #doctest: +ELLIPSIS
    <utlvce.model.Model object at 0x...>
    """
    A = _dag_avg_deg(p, k, random_state=random_state)
    return sample_parameters(A,
                             I,
                             num_latent,
                             e,
                             var_lo,
                             var_hi,
                             int_var_lo,
                             int_var_hi,
                             psi_lo,
                             psi_hi,
                             int_psi_lo,
                             int_psi_hi,
                             B_lo,
                             B_hi,
                             obs=obs,
                             random_state=42,
                             sparse_latents=sparse_latents,
                             verbose=verbose)


def sample_parameters(A,
                      I,
                      num_latent,
                      e,
                      var_lo,
                      var_hi,
                      int_var_lo,
                      int_var_hi,
                      psi_lo,
                      psi_hi,
                      int_psi_lo,
                      int_psi_hi,
                      B_lo,
                      B_hi,
                      sparse_latents=False,
                      obs=True,
                      random_state=42,
                      verbose=0):
    """Generate a random model given an adjacency matrix `A` and
    intervention targets `I`.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix of the DAG underlying the model, where
        `A[i,j] != 0` implies  `i -> j`.
    I : set
        The set of intervention targets.
    num_latent : int
        The number of latent variables in the model.
    e : int
        The number of environments.
    var_lo : float
        The lower bound for the variances of the noise terms of the
        observed variables.
    var_hi : float
        The upper bound for the variances of the noise terms of the
        observed variables.
    int_var_lo : float
        The lower bound for the intervention variances on the observed
        variables.
    int_var_hi : float
        The upper bound for the intervention variances on the observed
        variables.
    psi_lo : float
        The lower bound for the variances of the latent variables.
    psi_hi : float
        The upper bound for the variances of the latent variables.
    int_psi_lo : float
        The lower bound for the intervention variances on the latent
        variables.
    int_psi_hi : float
        The upper bound for the intervention variances on the latent
        variables.
    B_lo : float
        The lower bound for the edge weights between observed
        variables.
    B_hi : float
        The upper bound for the edge weights between observed
        variables.
    sparse_latents : bool, default=False
        If the gamma matrix of latent effects should be sparse (see
        source).
    obs : bool, default=True
        Whether the first environment should be "observational",
        i.e. that the variances of the noise terms and latents are
        lower (variable-wise) than the other environments. With
        `obs=True`, the variances for first environment are sampled
        from `[var_lo, var_hi]` and, from `[var_lo + int_var_lo,
        var_hi + int_var_hi]` for the remaining environments; the same
        holds for the sampling of `psi`. If `obs=False`, the latter
        interval is used for all environments. Note that is not a
        necessary assumption for the UT-LVCE estimator, but makes the
        actual intervention strength less sensitive to the random
        sampling of parameters.
    random_state : int, default=42
        To set the random state for reproducibility. Successive calls
        with the same random state will return the same model.
    verbose: int, default = 0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    model : utlvce.model.Model
        An instance of the model with the sampled parameters.

    Raises
    ------
    ValueError :
        If the given adjacency is not a DAG or the intervention
        targets are not a subset of the variable indices,
        i.e. `[0,...,p-1]`.

    Examples
    --------
    >>> A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    >>> sample_parameters(A,{2},2,5,0.5,0.6,3,6,0.2,0.4,1,5,0.7,0.8,False,True,42,0) #doctest: +ELLIPSIS
    <utlvce.model.Model object at 0x...>

    Requesting an inappropriate (>p) number of targets yields a `ValueError`:

    >>> sample_parameters(A,{3},2,5,0.5,0.6,3,6,0.2,0.4,1,5,0.7,0.8,False,True,42,0)
    Traceback (most recent call last):
    ...
    ValueError: The intervention targets must be a subset of [0,...,p-1].

    A `ValueError` is raised if the given adjacency does not correspond to a DAG (e.g. it contains cycles):

    >>> A = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])    
    >>> sample_parameters(A,{2},2,5,0.5,0.6,3,6,0.2,0.4,1,5,0.7,0.8,False,True,42,0)
    Traceback (most recent call last):
    ...
    ValueError: The given adjacency does not correspond to a DAG.

    """
    # Check inputs
    p = len(A)
    if not utils.is_dag(A):
        raise ValueError("The given adjacency does not correspond to a DAG.")
    if not I <= set(range(p)):
        raise ValueError(
            "The intervention targets must be a subset of [0,...,p-1].")
    rng = np.random.default_rng(random_state)
    B = A * rng.uniform(B_lo, B_hi, size=A.shape)
    # Sample gamma
    if sparse_latents:
        n_latent_edges = round(p / 5)
        gamma = np.zeros((num_latent, p))
        for i in range(num_latent):
            non_zeros = rng.choice(range(p), size=n_latent_edges)
            gamma[i, non_zeros] = rng.normal(
                0, 1 / np.sqrt(n_latent_edges), size=n_latent_edges)
    else:
        gamma = rng.normal(0, 1 / np.sqrt(p), size=(num_latent, p))
        # Sample omegas, with or without an observational environment
    if obs and e > 1:
        omegas = np.tile(rng.uniform(var_lo, var_hi, size=p), (e, 1))
        interventions = np.zeros_like(omegas)
        for j in I:
            interventions[1:, j] = rng.uniform(
                int_var_lo, int_var_hi, size=(e - 1))
            omegas += interventions
    else:
        omegas = rng.uniform(var_lo, var_hi, size=(e, p))
        for j in range(p):
            if j not in I:
                omegas[:, j] = omegas[:, j].mean()
                # Sample psis
    if obs and e > 1 and num_latent > 0:
        psis = np.zeros((e, num_latent))
        psis[0, :] = rng.uniform(psi_lo, psi_hi, size=num_latent)
        for k in range(1, e):
            psis[k, :] = psis[0, :]
            psi_interventions = rng.uniform(
                int_psi_lo, int_psi_hi, size=(e, num_latent))
            psi_interventions[0, :] = 0
            psis += psi_interventions
    else:
        psis = rng.uniform(psi_lo, psi_hi, size=(e, num_latent))
        # Construct model
    model = Model(A, B, gamma, omegas, psis)
    # Debugging output
    if verbose:
        print("  Spec. norm gamma @ psi @ gamma:", spectral_norm(
            gamma.T @ np.diag(psis[0]) @ gamma))
        print("  Spec. norm omegas:", spectral_norm(
            np.diag(omegas[:, 0])))
        print(gamma.T @ np.diag(psis[0]) @ gamma)
        print(np.diag(omegas[:, 0]))
    return model


def intervention_targets(p, num_targets, random_state=42):
    """Sample a set of intervention targets.

    Parameters
    ----------
    p : int
        The number of variables, i.e. targets will be sampled from
        `[0,p-1]`.
    num_targets : int or tuple
        Specifies the number of targets. If a two-element tuple,
        the number of targets is sampled uniformly at random from
        `[size[0], size[1]]`
    random_state : int
        To set the random state for reproducibility.

    Returns
    -------
    targets : set
        A set with the indices of the intervention targets.

    Raises
    ------
    ValueError :
        If the given number of targets is invalid.

    Examples
    --------
    >>> intervention_targets(20, 3)
    {1, 13, 14}

    >>> intervention_targets(20, (1,10), random_state=1)
    {0, 2, 8, 12, 17}

    >>> intervention_targets(20, (1,10), random_state=2)
    {1, 3, 4, 6, 8, 13, 18, 19}

    >>> intervention_targets(10, 10)
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    >>> intervention_targets(10, 0)
    set()

    Requesting an inappropriate (>p) number of targets yields a `ValueError`:

    >>> intervention_targets(10, 11)
    Traceback (most recent call last):
    ...
    ValueError: Invalid number of targets.

    >>> intervention_targets(10, (0,11))
    Traceback (most recent call last):
    ...
    ValueError: Invalid number of targets.

    """
    # Check inputs
    invalid_input = (isinstance(num_targets, int) and num_targets > p)
    invalid_input = invalid_input or (isinstance(num_targets, tuple) and (
        len(num_targets) != 2 or max(num_targets) > p))
    if invalid_input:
        raise ValueError("Invalid number of targets.")
    # Generate targets
    rng = np.random.default_rng(random_state)
    if isinstance(num_targets, tuple):
        num_targets = rng.integers(num_targets[0], num_targets[1], size=1)
    targets = rng.choice(range(p), replace=False, size=num_targets)
    return set(targets)


def spectral_norm(X):
    return np.linalg.norm(X, ord=2)


def _dag_avg_deg(p, k, w_min=1, w_max=1, return_ordering=False, random_state=None, debug=False):
    """Generate an Erdos-Renyi graph with p nodes and average degree k,
    and orient edges according to a random ordering. Sample the edge
    weights from a uniform distribution. (NOTE: this function is a
    copy from the sempler package, i.e. sempler.generators.dag_avg_deg)

    Parameters
    ----------
    p : int
        The number of nodes in the graph.
    k : float
        The desired average degree.
    w_min : float, optional
        The lower bound on the sampled weights. Defaults to 1.
    w_max : float, optional
        The upper bound on the sampled weights. Defaults to 1.
    return_ordering: bool, optional
        If the topological ordering used to orient the edges should be
        returned.
    random_state : int,optional
        To set the random state for reproducibility.
    debug : bool, optional
        If debug traces should be printed.

    Returns
    -------
    W : numpy.ndarray
       The connectivity (weights) matrix of the generated DAG.
    ordering : numpy.ndarray, optional
       If return_ordering = True, a topological ordering of the graph.

    Example
    -------
    >>> _dag_avg_deg(5, 2, random_state = 42)
    array([[0., 0., 0., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    Optionally, the ordering used to orient the edges can be returned

    >>> _dag_avg_deg(5, 2, return_ordering = True, random_state = 42)
    (array([[0., 0., 0., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]]), array([0, 4, 1, 2, 3]))

    """
    np.random.seed(random_state) if random_state is not None else None
    # Generate adjacency matrix as if top. ordering is 1..p
    prob = k / (p-1)
    print("p = %d, k = %0.2f, P = %0.4f" % (p, k, prob)) if debug else None
    A = np.random.uniform(size=(p, p))
    A = (A <= prob).astype(float)
    A = np.triu(A, k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights

    # Permute rows/columns according to random topological ordering
    permutation = np.random.permutation(p)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    print("avg degree = %0.2f" % (np.sum(A) * 2 / len(A))) if debug else None
    if return_ordering:
        return (W[permutation, :][:, permutation], np.argsort(permutation))
    else:
        return W[permutation, :][:, permutation]


# --------------------------------------------------------------------
# Doctests
if __name__ == '__main__':
    import doctest
    doctest.testmod(extraglobs={}, verbose=True, optionflags=doctest.ELLIPSIS)
