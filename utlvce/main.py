# Copyright 2020 Juan Luis Gamella Martin

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
"""

import numpy as np
import utlvce.utils as utils
from utlvce.score import Score
import ges
import ges.scores
import gc  # Garbage collector
import time

# TODO
#   - Move functions to utils
#   - Score verbosity is passed as init
#   - Remove lambdas_graph,I parameters
#   - Remove method= parameter

# ---------------------------------------------------------------------
# Module API


def equivalence_class(candidate_dags, data, prune_edges=False,
                      nums_latent=[1, 2, 3],
                      folds=[.5, .25, .25],
                      psi_max=None,
                      psi_fixed=False,
                      max_iter=1000,
                      threshold_dist_B=1e-3,
                      threshold_fluctuation=1e-5,
                      max_fluctuations=5,
                      threshold_score=1e-5,
                      learning_rate=1,
                      B_solver='grad',
                      random_state=42,
                      verbose=0):
    """Estimate the equivalence class of the best scoring graph from an
    initial set of candidate DAGs.

    Parameters
    ----------
    candidate_dags : list of numpy.ndarray
        A list of the candidate DAG adjacencies, where for each
        adjacency A, `A[i, j] != 0 implies i -> j`.
    data : list of numpy.ndarray
        A list with the sample from each environment, where each
        sample is an array with columns corresponding to variables and
        rows to observations.
    prune_edges : bool, default=False
        If we also consider the pruned versions of the candidate
        graphs.
    nums_latent : list of ints, default=[1,2,3]
        The candidates for the number of hidden variables, from which
        one is selected via cross-validation.
    folds : list of float, default=[0.5, 0.25, 0.25]
        The size of the different splits of the data, where the first
        corresponds to the training data (used to fit the models), the
        second is used to select the number of latent variables and best
        DAG, and the third is used to select the intervention targets.
    psi_max: float, default = None
       The maximum allowed change in variance between environments for
       the hidden variables. If None, psis are unconstrained.
    psi_fixed: bool, default = False
       If `True`, impose the additional constraint that the hidden
       variables have all the same variance.
    max_iter: int, default=1000
       The maximum number of iterations allowed for the alternating
       minimization procedure.
    threshold_dist_B: float, default=1e-3
       If the change in B between successive iterations of the
       alternating optimization procedure is lower than this
       threshold, stop the procedure and return the estimate.
    threshold_fluctuation: float, default=1e-5
       For the alternating optimization routine, if the score worsens by
       more than this value, consider the iteration a fluctuation.
    max_fluctuations: int, default=5
       For the alternating optimization routine, the maximum number of
       fluctuations(see above) allowed before stopping the
       subroutine.
    threshold_score: float, default=1e-5
       For the gradient descent subroutines, if change in score
       between successive iterations is below this threshold, stop.
    learning_rate: float, default=1
        The initial learning rate(factor by which gradient is
        multiplied) for the gradient descent subroutines.
    B_solver : {'grad', 'adaptive', 'cvx'}, default='grad'
        Sets the solver for the connectivity matrix B, where the
        options are (ordered by decreasing speed and increasing stability)
        `grad`, `adaptive` and `cvx`.
    random_state : int, default=42
        To set the random state for reproducibility when randomly
        splitting the data. Successive calls with the same random state
        will have the same result.
    verbose: int, default = 0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    estimated_icpdag : numpy.ndarray
        The I-CPDAG representing the estimated equivalence class.
    estimated_I : set of ints
        The estimated set of intervention targets.
    estimated_model : utlvce.model.Model
        The estimated model. Its underlying DAG adjacency can be
        accessed by `estimated_model.A`. The fitted parameters and
        assumption deviation metrics can be seen by calling
        `print(estimated_model)` (see `utlvce.model` module).

    Raises
    ------
    ValueError :
        If (1) the given data is not valid, i.e. (different number of
        variables per sample, or one sample with a single
        observation), (2) if the given `B_solver` is not valid or (3)
        if one of the candidate adjacencies does not correspond to a
        DAG.
    TypeError :
        If the given data is not a list of `numpy.ndarray`.

    """
    candidate_dags = np.array(candidate_dags)
    result = _fit(data=data,
                  initial_graphs=candidate_dags,
                  nums_latent=nums_latent,
                  prune_graph=prune_edges,
                  psi_max=psi_max,
                  psi_fixed=psi_fixed,
                  max_iter=max_iter,
                  threshold_dist_B=threshold_dist_B,
                  threshold_fluctuation=threshold_fluctuation,
                  max_fluctuations=max_fluctuations,
                  threshold_score=threshold_score,
                  learning_rate=learning_rate,
                  B_solver=B_solver,
                  verbose=verbose,
                  folds=folds,
                  random_state=42,
                  store_history=0)
    (estimated_model, estimated_I, _test_score), exec_history = result
    # Compute I-CPDAG
    estimated_icpdag = utils.dag_to_icpdag(estimated_model.A, estimated_I)
    return estimated_icpdag, estimated_I, estimated_model


def equivalence_class_w_ges(data,
                            nums_latent=[1, 2, 3],
                            folds=[.5, .25, .25],
                            psi_max=None,
                            psi_fixed=False,
                            max_iter=1000,
                            threshold_dist_B=1e-3,
                            threshold_fluctuation=1e-5,
                            max_fluctuations=5,
                            threshold_score=1e-5,
                            learning_rate=1,
                            B_solver='grad',
                            random_state=42,
                            verbose=0):
    """Estimate the equivalence class of the data-generating model, using
    the output of GES on the pooled data as an initial set of
    candidate DAGs.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list with the sample from each environment, where each
        sample is an array with columns corresponding to variables and
        rows to observations.
    nums_latent : list of ints, default=[1,2,3]
        The candidates for the number of hidden variables, from which
        one is selected via cross-validation.
    folds : list of float, default=[0.5, 0.25, 0.25]
        The size of the different splits of the data, where the first
        corresponds to the training data (used to fit the models), the
        second is used to select the number of latent variables and best
        DAG, and the third is used to select the intervention targets.
    psi_max: float, default = None
       The maximum allowed change in variance between environments for
       the hidden variables. If None, psis are unconstrained.
    psi_fixed: bool, default = False
       If `True`, impose the additional constraint that the hidden
       variables have all the same variance.
    max_iter: int, default=1000
       The maximum number of iterations allowed for the alternating
       minimization procedure.
    threshold_dist_B: float, default=1e-3
       If the change in B between successive iterations of the
       alternating optimization procedure is lower than this
       threshold, stop the procedure and return the estimate.
    threshold_fluctuation: float, default=1e-5
       For the alternating optimization routine, if the score worsens by
       more than this value, consider the iteration a fluctuation.
    max_fluctuations: int, default=5
       For the alternating optimization routine, the maximum number of
       fluctuations(see above) allowed before stopping the
       subroutine.
    threshold_score: float, default=1e-5
       For the gradient descent subroutines, if change in score
       between successive iterations is below this threshold, stop.
    learning_rate: float, default=1
        The initial learning rate(factor by which gradient is
        multiplied) for the gradient descent subroutines.
    B_solver : {'grad', 'adaptive', 'cvx'}, default='grad'
        Sets the solver for the connectivity matrix B, where the
        options are (ordered by decreasing speed and increasing stability)
        `grad`, `adaptive` and `cvx`.
    random_state : int, default=42
        To set the random state for reproducibility when randomly
        splitting the data. Successive calls with the same random state
        will have the same result.
    verbose: int, default = 0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    estimated_icpdag : numpy.ndarray
        The I-CPDAG representing the estimated equivalence class.
    estimated_I : set of ints
        The estimated set of intervention targets.
    estimated_model : utlvce.model.Model
        The estimated model. Its underlying DAG adjacency can be
        accessed by `estimated_model.A`. The fitted parameters and
        assumption deviation metrics can be seen by calling
        `print(estimated_model)` (see `utlvce.model` module).

    Raises
    ------
    ValueError :
        If (1) the given data is not valid, i.e. (different number of
        variables per sample, or one sample with a single
        observation), (2) if the given `B_solver` is not valid.
    TypeError :
        If the given data is not a list of `numpy.ndarray`.

    """
    result = _fit(data=data,
                  initial_graphs=None,  # This results in _fit running GES with default settings
                  nums_latent=nums_latent,
                  prune_graph=True,
                  psi_max=psi_max,
                  psi_fixed=psi_fixed,
                  max_iter=max_iter,
                  threshold_dist_B=threshold_dist_B,
                  threshold_fluctuation=threshold_fluctuation,
                  max_fluctuations=max_fluctuations,
                  threshold_score=threshold_score,
                  learning_rate=learning_rate,
                  B_solver=B_solver,
                  verbose=verbose,
                  folds=folds,
                  random_state=42,
                  store_history=0)
    (estimated_model, estimated_I, _test_score), exec_history = result
    # Compute I-CPDAG
    estimated_icpdag = utils.dag_to_icpdag(estimated_model.A, estimated_I)
    return estimated_icpdag, estimated_I, estimated_model

# ---------------------------------------------------------------------
# Internal functions


def _fit(data,
         psi_max,
         psi_fixed,
         max_iter,
         threshold_dist_B,
         threshold_fluctuation,
         max_fluctuations,
         threshold_score,
         learning_rate,
         B_solver,
         score_verbose=0,
         nums_latent=None,
         prune_graph=True,
         folds=[.5, .25, .25],
         threshold_graph=None,
         threshold_I=None,
         initial_graphs=None,
         ges_env=None,
         ges_phases=None,
         ges_lambdas=None,
         random_state=42,
         store_history=1,
         verbose=0):
    """
    Monolith function of the UT-LVCE algorithm.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list with the sample from each environment, where each
        sample is an array with columns corresponding to variables and
        rows to observations.
    psi_max: float
       The maximum allowed change in variance between environments for
       the hidden variables. If None, psis are unconstrained.
    psi_fixed: bool
       If `True`, impose the additional constraint that the hidden
       variables have all the same variance.
    max_iter: int
       The maximum number of iterations allowed for the alternating
       minimization procedure.
    threshold_dist_B: float
       If the change in B between successive iterations of the
       alternating optimization procedure is lower than this
       threshold, stop the procedure and return the estimate.
    threshold_fluctuation: float
       For the alternating optimization routine, if the score worsens by
       more than this value, consider the iteration a fluctuation.
    max_fluctuations: int
       For the alternating optimization routine, the maximum number of
       fluctuations(see above) allowed before stopping the
       subroutine.
    threshold_score: float
       For the gradient descent subroutines, if change in score
       between successive iterations is below this threshold, stop.
    learning_rate: float
        The initial learning rate(factor by which gradient is
        multiplied) for the gradient descent subroutines.
    B_solver : {'grad', 'adaptive', 'cvx'}
        Sets the solver for the connectivity matrix B, where the
        options are (ordered by decreasing speed and increasing stability)
        `grad`, `adaptive` and `cvx`.
    score_verbose : int, default=0
        Controls the level of verbosity of the alternating minmization
        procedure.
    nums_latent : NoneType or list of ints, default=None
        The candidates for the number of hidden variables, from which
        one is selected via cross-validation. If `None`, the number of
        latents is selected using a scree-plot procedure (see
        `_scree_selection` in this module).
    prune_graph : bool, default=True
        If we also consider the pruned versions of the candidate
        graphs.
    folds : list of float, default=[0.5, 0.25, 0.25]
        The size of the different splits of the data, where the first
        corresponds to the training data (used to fit the models), the
        second is used to select the number of latent variables and best
        DAG, and the third is used to select the intervention targets.
    threshold_graph : NoneType or float, default=None
        If not `None`, the graphs are pruned by removing all edges
        with weight (in abs. value) below this threshold (see
        _prune_graphs).
    threshold_I : NoneType or float, default=None
        If not `None`, the estimated intervention targets are selected
        by taking variables whose variance of noise-term-variances is
        above this threshold (see _prune_I).
    initial_graphs: NoneType or numpy.ndarray
        If `None`, GES is run to obtain an initial set of candidate
        DAGs. The parameters `ges_{env,phases,lambdas}` control the
        behaviour of GES. Otherwise, initial graphs is an array
        containing the the candidate DAG adjacencies, where for each
        adjacency A, `A[i, j] != 0 implies i -> j`.
    ges_env : NoneType or int, default=None
        The index of the environment on whose sample GES should
        run. If `None`, GES runs on the data pooled across
        environments.
    ges_phases : NoneType or [{'forward', 'backward', 'turning'}*], default=None
        Specifies the phases of GES which should be run, and in which
        order. When `None` GES will run the forward, backward and
        turning phases.
    ges_lambdas : NoneType or list of float
        If `None`, GES runs with the default BIC score penalization
        (2). If specified, GES runs for the given penalization values
        and the resulting graphs are pooled.
    random_state : int, default=42
        To set the random state for reproducibility when randomly
        splitting the data. Successive calls with the same random state
        will have the same result.
    store_history : int, default=1
       If the execution history should be stored and returned, and in what level of detail, where
         - =0 : Do not store additional information
         - >0 : Store initial dags
         - >1 : Store all pruned graphs / Is and their scores
         - >2 : Store the cached score for computing metrics / debugging.
    verbose: int, default = 0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    estimated_cpdag : numpy.ndarray
        The CPDAG representing the estimated equivalence class.
    estimated_I : set of ints
        The estimated set of intervention targets.
    estimated_model : utlvce.model.Model
        The estimated model. Its underlying DAG adjacency can be
        accessed by `estimated_model.A`. The fitted parameters and
        assumption deviation metrics can be seen by calling
        `print(estimated_model)` (see `utlvce.model` module).

    Raises
    ------
    ValueError :
        If (1) the given data is not valid, i.e. (different number of
        variables per sample, or one sample with a single
        observation), (2) if the given `B_solver` is not valid or (3)
        if one of the candidate adjacencies does not correspond to a
        DAG.
    TypeError :
        If the given data is not a list of `numpy.ndarray`.

    """
    # Split the data
    [training_data, test_data_graph, test_data_I] = utils.split_data(
        data, ratios=folds, random_state=random_state)
    test_sample_covariances_graph = np.array(
        [np.cov(X, rowvar=False) for X in test_data_graph])
    test_n_obs_graph = [len(X) for X in test_data_graph]
    test_sample_covariances_I = np.array(
        [np.cov(X, rowvar=False) for X in test_data_I])
    test_n_obs_I = [len(X) for X in test_data_I]

    # Set the base score parameters, i.e. everything except the data
    # and number of latent variables
    score_params = {
        'psi_max': psi_max,
        'psi_fixed': psi_fixed,
        'max_iter': max_iter,
        'threshold_dist_B': threshold_dist_B,
        'threshold_fluctuation': threshold_fluctuation,
        'max_fluctuations': max_fluctuations,
        'threshold_score': threshold_score,
        'learning_rate': learning_rate,
        'B_solver': B_solver}

    if initial_graphs is None:
        # Run GES on pooled training data or only a particular environment
        ges_data = training_data if ges_env is None else [
            training_data[ges_env]]
        initial_graphs = _fit_ges(
            ges_data, phases=ges_phases, lambdas=ges_lambdas, verbose=verbose)

    if verbose:
        print("No. edges in initial graphs (%d)" % len(initial_graphs), [np.sum(A)
                                                                         for A in initial_graphs])  # if verbose else None

    # If no numbers of latents were given to select from by
    # cross-validation, pick the upper bound using the scree procedure
    if nums_latent is None:
        obs_covariance = np.cov(training_data[0], rowvar=False)
        scree_selection = _scree_selection(obs_covariance)
        nums_latent = list(range(1, scree_selection + 1))
        print("Selected h=%s via the scree plot." %
              nums_latent) if verbose else None

    # ----------------------------------
    # Graph-pruning phase

    print("\n\nCross-validation for the graph-pruning phase") if verbose else None
    results = dict()
    score_caches = dict()
    for h in nums_latent:
        print("\n  h = %d" % h) if verbose else None
        # Start the score class for this number of latents h
        train_cache = Score(training_data, h, **score_params)
        # Prune all graphs by iteratively removing the weakest edge
        # until the graph is empty (fitting on the train data)
        pruned_models = _prune_graphs(initial_graphs, train_cache,
                                      verbose, prune_graph, threshold_graph)
        # Compute the test score of all the resulting graphs
        test_scores = [(m.score(test_sample_covariances_graph, test_n_obs_graph), m.A)
                       for m in pruned_models]
        # Pick the graph with the highest test score
        (best_test_score, best_graph) = _best_scoring(test_scores)
        print("    Best test score = %s" %
              best_test_score) if verbose else None
        # Store the result and the score class for every h
        results[h] = (best_test_score, best_graph, test_scores)
        score_caches[h] = train_cache

    # Fix h to be the one which gave the best test score
    to_sort = [(test_score, h, best_graph, test_scores)
               for h, (test_score, best_graph, test_scores) in results.items()]
    (_, num_latent, best_graph, pruned_graphs) = sorted(to_sort)[0]

    # Reuse the score class of the selected h, and remove the others
    # from memory
    train_cache = score_caches[num_latent]
    del score_caches
    gc.collect()

    print("\n  Best parameter: h = %s" % num_latent) if verbose else None

    # ----------------------------------
    # I-pruning phase

    print("\n\nBeginning I-pruning phase\n") if verbose else None

    pruned_models_I = _prune_I(best_graph, train_cache, verbose, threshold_I)
    # Compute the test score of all the resulting models
    test_scores = [(m.score(test_sample_covariances_I, test_n_obs_I), m, I)
                   for (m, I) in pruned_models_I]
    # Pick the graph with the highest test score
    # TODO: Maybe have I be a parameter of the model
    (final_test_score, best_model, best_I) = _best_scoring(test_scores)
    print("\n  Best test score = %s" % final_test_score) if verbose else None

    # ----------------------------------
    # Store necessary parameters and return

    print("\nBest parameters: h = %s - graph with %d edges - Î = %s\n" %
          (num_latent, np.sum(best_model.A), best_I)) if verbose else None

    # Return the best model and its test score
    # Additionally, for store_history
    #  = 0 - Do not store additional information
    #  > 0 - Store initial dags
    #  > 1 - Store all pruned graphs / Is and their scores
    #  > 2 - Store the cached score for computing metrics / debugging.
    if store_history == 0:
        history = None
    elif store_history:
        history = {'num_latent': num_latent,
                   'initial_graphs': initial_graphs}
        if store_history > 1:
            # Store pruned graphs and pruned Is
            history['pruned_graphs'] = pruned_graphs
            history['pruned_Is'] = test_scores
        if store_history > 2:
            history['train_cache'] = train_cache

    return (best_model, best_I, final_test_score), history


def _gaussian_log_likelihood(sample, mean, var):
    """Return the log likelihood of a sample given a univariate gaussian
    distribution with given mean and variance."""
    log_likelihood = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var)
    log_likelihood -= ((sample - mean)**2).sum() / 2 / var
    return log_likelihood


def _scree_selection(cov):
    """Given a sample covariance, select the number of latents using
    the scree plot as described in _Automatic dimensionality selection
    from the scree plot via the use of profile likelihood_ by Mu Zhu
    and Ali Ghodsi: partition the eigenvalues into two sets and fit
    two gaussian distributions, picking the cutoff point with highest
    likelihood.
    """
    svds = np.linalg.svd(cov)[1]
    p = len(svds)
    likelihoods = []
    for h in range(1, p):
        s1, s2 = svds[0:h], svds[h:p]
        # The variance used for the likelihood is the pooled sample variance
        v1 = s1.var(ddof=1) if len(s1) > 1 else 0
        v2 = s2.var(ddof=1) if len(s2) > 1 else 0
        var = ((h - 1) * v1 + (p - h - 1) * v2) / (p - 2)
        likelihood = _gaussian_log_likelihood(
            s1, s1.mean(), var) + _gaussian_log_likelihood(s2, s2.mean(), var)
        likelihoods.append(likelihood)
    return np.argmax(likelihoods) + 1


def _best_scoring(tuples, loc=0):
    """
    Given a list of models (+ other params) and their scores, return
    the one with the smallest score.
    """
    scores = [t[loc] for t in tuples]
    i = np.argmin(scores)
    # if len(np.unique(scores)) < len(scores):
    #     print("WARNING: several models scored the same")
    return tuples[i]


def _prune_graphs(graphs, score_class, debug=0, prune=True, threshold=None):
    """Greedily prune the given graphs: for each one, remove the smallest edge (in
    absolute value) until the graph is empty.
    """
    full_I = set(range(graphs.shape[1]))
    print(' ' * 3, "Starting graph-pruning phase for %d graphs - prune: %s" %
          (len(graphs), prune)) if debug else None
    pruned_models = []
    for i, graph in enumerate(graphs):
        # Fit the given graphs (i.e. initial models)
        initial_model, likelihood = score_class.score_dag(
            graph, full_I, verbose=0)
        print(' ' * 5, "Graph %i - #edges = %d - likl. = %s" %
              (i, np.sum(graph), likelihood)) if debug else None
        # No pruning
        if not prune:
            pruned_models.append(initial_model)
        # Prune by thresholding
        elif threshold is not None:
            max_weight = abs(initial_model.B).max()
            pruned_A = (abs(initial_model.B) >
                        max_weight * threshold).astype(int)
            pruned_model, likelihood = score_class.score_dag(
                pruned_A, full_I, verbose=0)
            print(' ' * 7, "thresholding - removed %d edges - likl. = %s" %
                  (np.sum(graph) - np.sum(pruned_A), likelihood)) if debug else None
            pruned_models.append(pruned_model)
        # Prune by iteratively removing the weakest edge
        else:
            pruned_models.append(initial_model)
            current_dag, current_model = graph.copy(), initial_model
            while np.sum(current_dag) > 0:
                # Remove weakest edge
                weakest_edge = _weakest_edge(current_model.B)
                weight = current_model.B[weakest_edge]
                current_dag[weakest_edge] = 0
                # Fit pruned model and store it
                current_model, likelihood = score_class.score_dag(
                    current_dag, full_I, verbose=0)
                pruned_models.append(current_model)
                print(' ' * 7, "removing edge %s (%s) (%d edges) -> likl. = %s" %
                      (weakest_edge, weight, np.sum(current_dag), likelihood)) if debug else None
    return pruned_models


def _prune_I(graph, score_class, verbose=0, threshold=None):
    """Given an estimated model and intervention targets I, prune I until
    the model's score does not improve. We choose which I to remove by
    either:
      - scoring all resulting subsets of I, and taking the highest
        scoring one (i.e. `method="score"`), or
      - removing the variable in I for which its estimated noise-term
        variances have the smallest "variance" across environments
        (i.e. `method="rank"`).
    """
    # Obtain pruning order from fitting the graph with full I: remove
    # first variables for which the estimated noise term variances
    # vary the least accross environments: TODO: fix comment
    full_I = set(range(len(graph)))
    initial_model, likelihood = score_class.score_dag(graph, full_I)
    variances = np.var(initial_model.omegas, axis=0)
    # print(variances)
    # print(initial_model.omegas)
    # Select I by picking targes whose variance is above a threshold
    if threshold is not None:
        I = set(np.where(variances > variances.max() * threshold)[0])
        model, likelihood = score_class.score_dag(graph, I)
        print(' ' * 3, "thresholding - I = %s - likl. = %s" %
              (I, likelihood)) if verbose else None
        pruned_models = [(model, I)]
    # Select I by iteratively removing targets
    else:
        order = np.argsort(variances)
        print(' ' * 1, "Pruning I - order = %s - likl. = %s" %
              (order, likelihood)) if verbose else None
        # Remove following the order until score does not improve
        I = full_I
        pruned_models = [(initial_model, full_I)]
        for j in order:
            I = I - {j}
            current_model, likelihood = score_class.score_dag(graph, I)
            print(' ' * 3, "(likl. = %s): I - {%d} = %s" %
                  (likelihood, j, I)) if verbose else None
            pruned_models.append((current_model, I))
    # Return
    return pruned_models


def _weakest_edge(B):
    """Return the index of the weakest edge in B, i.e. non-zero with
    smallest absolute value."""
    B = B.copy()
    B[B == 0] = np.Inf
    return utils.argmin(abs(B))


def _fit_ges(data, verbose=1, lambdas=None, phases=None):
    """Pool the data and run GES on it, returning the estimated CPDAG.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list containing the sample from each environment.
    verbose : int, default=0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.
    lambdas : NoneType or list of float
        If `None`, GES runs with the default BIC score penalization
        (2). If specified, GES runs for the given penalization values
        and the resulting graphs are pooled.
    phases : NoneType or [{'forward', 'backward', 'turning'}*], default=None
        Specifies the phases of GES which should be run, and in which
        order. When `None` GES will run the forward, backward and
        turning phases.

    Returns
    -------
    cpdag : numpy.ndarray
        The adjacency matrix of the estimated CPDAG.
    """
    # Set phases & lambdas
    phases = ['forward', 'backward', 'turning'] if phases is None else phases
    lambdas = [2] if lambdas is None else lambdas
    # Pool data
    pooled_data = np.vstack(data)
    N = len(pooled_data)
    # Run GES
    graphs = []
    for pen in lambdas:
        start = time.time()
        print("  Running GES on pooled data for λ=%0.2f with phases=%s... " %
              (pen, phases), end="") if verbose > 0 else None
        # Set penalization
        lmbda = pen * 0.5 * np.log(N)
        score_class = ges.scores.GaussObsL0Pen(pooled_data, lmbda=lmbda)
        # Run GES
        cpdag = ges.fit(score_class, phases=phases, iterate=True)[0]
        graphs += list(utils.all_dags(cpdag))
        print("  done (%0.2f seconds)" %
              (time.time() - start)) if verbose > 0 else None
    return np.array(graphs)


# --------------------------------------------------------------------
# Doctests
if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True,
                    optionflags=doctest.ELLIPSIS)
