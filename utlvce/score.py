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

"""The :mod:`utlvce.score` module contains the implementation of the alternating
optimization procedure described in the paper, which is used to fit a
UT-LVCE model to the data and compute its likelihood score.

The procedure is accessed through the :class:`utlvce.score.Score`
class, which contains a caching mechanism (see class
:class:`utlvce.score._Cache`) to avoid re-running the procedure for
the same DAG / intervention targets. For more details on the
implementation please refer to section 4.1 of the `paper
<https://arxiv.org/abs/2101.06950>`_ and to the `source-code
<https://github.com/juangamella/ut-lvce/blob/master/utlvce/score.py>`_.

"""

import numpy as np
import utlvce.utils as utils
from utlvce.model import Model
import cvxpy as cp
import scipy.linalg
import copy
import time

# ---------------------------------------------------------------------
# Module API


class Score():
    """Contains the implementation of the alternating optimization
    procedure described in the paper; which fits a UT-LVC model given
    a DAG adjacency and intervention targets `I`.

    Parameters
    ----------
    sample_covariances: numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs: list of ints
       The number of observations available from each environment
       (i.e. the sample size).
    num_latent: int
       The assumed number of hidden variables.
    psi_max: float, default = None
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
    cache: _Cache or None
        The cache used to store results of calls to score_dag().
    """

    def __init__(self, data, num_latent, psi_max, psi_fixed, max_iter, threshold_dist_B, threshold_fluctuation, max_fluctuations, threshold_score, learning_rate, B_solver='grad', cache=True):
        """Create a new instance of a Score object.

        Parameters
        ----------
        data : list of numpy.ndarray or tuple or numpy.ndarray
            Can be either:
              1) A list with the samples from each environment, where
                 each sample is an array with columns corresponding to
                 variables and rows to observations.
              2) A tuple containing the precomputed sample covariances
                 and number of observations from each environment.
        num_latent: int
           The assumed number of hidden variables.
        psi_max: float, default = None
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
        B_solver : {'grad', 'adaptive', 'cvx'}, default='grad'
            Sets the solver for the connectivity matrix B, where the
            options are (ordered by decreasing speed and increasing stability)
            `grad`, `adaptive` and `cvx`.
        cache : bool, default=True
            If results from calling the `score_dag` function should be cached.

        Raises
        ------
        ValueError :
            If the given data is not valid, i.e. (different number of
            variables per sample, or one sample with a single
            observation). Also if the given `B_solver` is not valid.
        TypeError :
            If the given data is of invalid type.

        Examples
        --------

        Initializing the :class:`utlvce.score.Score` using the "raw" data:

        >>> score_params = {'psi_max': None,
        ...                 'psi_fixed': False,
        ...                 'max_iter': 1000,
        ...                 'threshold_dist_B': 1e-4,
        ...                 'threshold_fluctuation': 1,
        ...                 'max_fluctuations': 10,
        ...                 'threshold_score': 1e-5,
        ...                 'learning_rate': 1,
        ...                 'cache': True}
        >>> data = list(rng.uniform(size=(5,1000,20)))
        >>> Score(data, num_latent=2, **score_params) #doctest: +ELLIPSIS        
        <__main__.Score object at 0x...>

        Or passing pre-computed sample covariances:

        >>> n_obs = np.array([len(sample) for sample in data])
        >>> sample_covariances = np.array([np.cov(sample, rowvar=False) for sample in data])
        >>> Score((sample_covariances, n_obs), num_latent=2, **score_params) #doctest: +ELLIPSIS
        <__main__.Score object at 0x...>

        Errors are raised when not all samples have the same number of variables:

        >>> bad_data = data.copy()
        >>> bad_data[0] = bad_data[0][:,:-1]
        >>> Score(bad_data, num_latent=2, **score_params)
        Traceback (most recent call last):
        ...
        ValueError: All samples must have the same number of variables.

        >>> n_obs = np.array([len(sample) for sample in bad_data])
        >>> sample_covariances = np.array([np.cov(sample, rowvar=False) for sample in bad_data])
        >>> Score((sample_covariances, n_obs), num_latent=2, **score_params)
        Traceback (most recent call last):
        ...
        ValueError: All samples must have the same number of variables.

        Or when one sample has a single observation:

        >>> bad_data = data.copy()
        >>> bad_data[0] = bad_data[0][[0], :]
        >>> Score(bad_data, num_latent=2, **score_params)
        Traceback (most recent call last):
        ...
        ValueError: Each sample must contain at least two observations to estimate the covariance matrix.

        Error when wrongly selecting the solver for `B`:
        >>> Score(data, num_latent=2, B_solver='test', **score_params)
        Traceback (most recent call last):
        ...
        ValueError: Unrecognized value "test" for parameter B_solver.
        """
        # Check inputs and compute/store covariance matrices
        type_msg = "data should be a list or tuple of numpy.ndarray."
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray):
            sample_covariances, n_obs = data
            if len(sample_covariances) != len(n_obs):
                raise ValueError(
                    "Sample covariances and number of observations must have same length")
            ps = np.array([len(s) for s in sample_covariances])
            if len(np.unique(ps)) > 1:
                raise ValueError(
                    "All samples must have the same number of variables.")
            # Store the sample covariances
            self.sample_covariances = sample_covariances.copy()
            self.n_obs = n_obs.copy()

        elif isinstance(data, list):
            # Check that all samples are arrays
            for i, x in enumerate(data):
                if not isinstance(x, np.ndarray):
                    raise TypeError(type_msg)
            # Check that all samples have the same number of variables
            ps = np.array([x.shape[1] for x in data])
            if len(np.unique(ps)) > 1:
                raise ValueError(
                    "All samples must have the same number of variables.")
            # Check that all samples have at least one observation
            n_obs = np.array([len(sample) for sample in data])
            if (n_obs == 1).any():
                raise ValueError(
                    "Each sample must contain at least two observations to estimate the covariance matrix.")
            # Compute sample covariances
            self.n_obs = n_obs
            self.sample_covariances = np.array(
                [np.cov(sample, rowvar=False) for sample in data])
        else:
            raise TypeError(type_msg)

        # Compute the sample covariances

        # Start cache
        self.cache = _Cache() if cache else None

        # Set the other parameters
        if B_solver == 'cvx':
            self.B_solver = _solve_for_B_cvx
        elif B_solver == 'grad':
            self.B_solver = _solve_for_B_grad
        elif B_solver == 'adaptive':
            self.B_solver = _solve_for_B_adaptive
        else:
            raise ValueError(
                'Unrecognized value "%s" for parameter B_solver.' % B_solver)

        self.num_latent = num_latent
        self.psi_max = psi_max
        self.psi_fixed = psi_fixed
        self.max_iter = max_iter
        self.threshold_dist_B = threshold_dist_B
        self.threshold_fluctuation = threshold_fluctuation
        self.max_fluctuations = max_fluctuations
        self.threshold_score = threshold_score
        self.learning_rate = learning_rate

    def score_dag(self, A, I, init_model=None, verbose=0):
        """Score the given DAG while allowing interventions on certain variables.

        Parameters
        ----------
        A: numpy.ndarray
            The adjacency of the given DAG, where `A[i, j] != 0 implies i -> j`.
        I: set
            The observed variables for which interventions are allowed,
            i.e. for which the noise term distribution is allowed to change
            between environments.
        init_model: utlvce.Model, default = None
            To set a set of parameters, encoded by an instance of
            utlvce.model.Model as the starting point of the procedure.
        verbose: int, default = 0
            If debug and execution traces should be printed. `0`
            corresponds to no traces, higher values correspond to higher
            verbosity.

        Raises
        ------
        ValueError:
            If the given adjacency does not correspond to a DAG.

        Returns
        -------
        model: utlvce.Model
            The estimated model with all nuisance parameters. `model.B`
            returns the connectivity(weight) matrix found to maximize the
            score.
        score: float
            The score attained by the given DAG adjacency.

        """

        # Check in cache
        if init_model is None and self.cache is not None:
            cached = self.cache.read(A, I)
            if cached is not None:
                return cached

        # Check inputs: A
        if not utils.is_dag(A):
            raise ValueError("The given graph is not a DAG.")

        # ----------------------------------------------------------------
        # Initialization

        if init_model is not None and not (init_model.A == A).all():
            raise ValueError(
                'model_init.A does not correspond to the given adjacency matrix A.')
        elif init_model is not None:
            model = init_model
        else:
            # Initialize B (connectivity matrix)
            B = _initialize_B(A, self.sample_covariances, self.n_obs)
            # Initialize gamma (connectivity matrix latents -> observed)
            gamma = _initialize_gamma(
                B, self.num_latent, self.sample_covariances, self.n_obs)
            # Initialize omegas (noise term variances of observed variables)
            omegas = _initialize_omegas(
                B, I, self.sample_covariances, self.n_obs)
            # Initialize psis (variances of the latent variables)
            psis = _initialize_psis(
                len(self.sample_covariances), self.num_latent)
            model = Model(A, B, gamma, omegas, psis)

        if verbose > 0:
            print("-- Alternating Minimization Procedure --")
            print("Initial values:")
            print(model)

        # ----------------------------------------------------------------
        # Alternating optimization procedure

        previous_model = model
        previous_score = model.score(self.sample_covariances, self.n_obs)
        dist_B = np.Inf
        i = 0
        fluc_counter = 0

        # We stop the alternating procedure when
        #   1. The L-infinity distance between successive Bs drops below a threshold,
        #   2. we reach a maximum number of iterations, or
        #   3. we reach a maximum number of fluctuations in the score (measured by threshold_fluctuation).
        while (dist_B > self.threshold_dist_B) and (i < self.max_iter) and (fluc_counter < self.max_fluctuations):
            current_model = previous_model.copy()

            # Solve for B
            new_B = self.B_solver(
                current_model, self.sample_covariances, self.n_obs)
            current_model.B = new_B
            print(' ' * 4, 'Solved for B') if verbose > 0 else None
            print(current_model.B) if verbose > 1 else None

            # Solve for gamma
            current_model.gamma = _solve_for_gamma(current_model,
                                                   self.sample_covariances,
                                                   self.n_obs,
                                                   self.threshold_score,
                                                   self.learning_rate,
                                                   verbose=max(0, verbose - 1))
            print(' ' * 4, 'Solved for gamma') if verbose > 0 else None
            print(current_model.gamma) if verbose > 1 else None

            # Solve for the rest
            current_model.omegas, current_model.psis = _solve_for_rest(current_model,
                                                                       I,
                                                                       self.sample_covariances,
                                                                       self.n_obs,
                                                                       self.psi_max,
                                                                       self.psi_fixed,
                                                                       self.threshold_score,
                                                                       self.learning_rate,
                                                                       verbose=max(0, verbose - 1))
            print(' ' * 4, 'Solved for omegas, psis') if verbose > 0 else None
            print(current_model.omegas,
                  current_model.psis) if verbose > 1 else None

            # Compute stopping conditions
            i += 1
            dist_B = abs(previous_model.B -
                         current_model.B).max()  # L-infinity
            current_score = current_model.score(
                self.sample_covariances, self.n_obs)
            if current_score - previous_score > self.threshold_fluctuation:
                fluc_counter += 1

            print(' ' * 2, "Iter: %d - #flucs: %d - dist. B: %s - delta_score = %s\n" %
                  (i, fluc_counter, dist_B, current_score - previous_score)) if verbose > 0 else None
            print(' ' * 4, "B: %s" % new_B) if verbose > 1 else None

            previous_model = current_model
            previous_score = current_score

        # ----------------------------------------------------------------
        # Cache result and return model and its score

        if self.cache is not None:
            self.cache.write(A, I, current_model, current_score)

        return current_model, current_score

# --------------------------------------------------------------------
# Internal _Cache class used by the Score class


class _Cache():
    """Class to cache the calls to the score function `score_dag`, indexed by the
    given adjacency matrix `A` and intervention targets `I`.
    """

    def __init__(self):
        self._As = []
        self._Is = []
        self._scores = []
        self._models = []

    def _find(self, A, I):
        """Return the index of a the pair (A,I) in the cache, or None if it is not stored."""
        # Search first over graphs, as these will be the most heterogeneous
        if len(self._As) == 0:
            return None
        A_indices = np.where((A == self._As).all(axis=(1, 2)))[0]
        # No match
        if len(A_indices) == 0:
            return None
        # Single match
        elif len(A_indices) == 1:
            index = A_indices[0]
            if self._Is[index] == I:
                return index
            else:
                return None
        # Several matches
        else:
            possible_Is = np.array([self._Is[i] for i in A_indices])
            I_subindices = np.where(possible_Is == I)[0]
            if len(I_subindices) == 0:
                return None
            elif len(I_subindices) == 1:
                I_index = I_subindices[0]
                index = A_indices[I_index]
                return index
            else:
                raise Exception(
                    "Cache should contain only unique (A,I) entries")

    def read(self, A, I):
        """Return the cached score of an adjacency + intervention targets, or
        None if it is not stored."""
        index = self._find(A, I)
        if index is None:
            return None
        else:
            return self._models[index], self._scores[index]

    def write(self, A, I, model, score, check=True):
        """Save an element in the cache. If `check=True`, return an error if
        the element already exists in the cache."""
        if check and self._find(A, I) is not None:
            raise ValueError("The given pair is already stored in the cache")
        else:
            self._As.append(A.copy())
            self._Is.append(I.copy())
            self._scores.append(score)
            self._models.append(model.copy())

# --------------------------------------------------------------------
# Functions to solve for the different parameters of the model


def _solve_for_B_grad(model, sample_covariances, n_obs, debug=False):
    """Given all other parameters of the model, solve the convex
    optimization problem to estimate B using a fast gradient descent
    procedure.

    Parameters
    ----------
    model : Model()
        An instance of `model.Model` containing the parameters
        representing the model.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
        The number of observations available from each environment
        (i.e. the sample size).

    Returns
    -------
    B : numpy.ndarray
        The estimated connectivity matrix between observed variables, where
        B[i,j] != 0 => i -> j.

    """
    # Inverse noise-term covariance matrices entailed by the model,
    # for each environment
    Ms = model.inv_noise_term_covariances()
    # Build "N" matrix
    N = np.zeros((model.p**2, model.p**2))
    for k, sigma in enumerate(sample_covariances):
        S = np.kron(scipy.linalg.sqrtm(sigma), scipy.linalg.sqrtm(Ms[k]))
        N += S.T @ S
    # Solve gradient descent
    B = np.zeros((model.p**2, 1))
    # B_init = model.B.T.flatten(order='F')
    B_init = np.random.uniform(size=(model.p**2, 1))
    B_new = B_init
    learn_rate = 0.005
    # TODO: Check flattening is correct
    zero_elements = np.logical_not(model.A.T).flatten(order='F')
    one_elements = np.eye(model.p).flatten(order='F').astype(bool)
    i = 0
    while abs(B - B_new).max() > 1e-10:
        step = learn_rate * N @ B_new
        # If the step becomes infinite, reduce the learning rate and
        # restart the gradient descent
        if learn_rate == 0:
            break
        if not np.isfinite(step).all():
            print("WARNING: _solve_for_B_grad : after %d iterations gradient step not finite. learn_rate: %s" % (
                i, learn_rate))
            learn_rate *= 0.5
            B = np.zeros((model.p**2, 1))
            B_new = B_init
            i = 0
            continue
        i += 1
        B = B_new
        B_new = B_new - step
        B_new[zero_elements] = 0
        B_new[one_elements] = 1
    final_B = np.eye(model.p) - B_new.reshape((model.p, model.p), order='F')
    if np.isfinite(final_B).all() and learn_rate > 0:
        # print("Converged in %d iterations" % i)
        return final_B.T
    else:
        cond_numbers = [np.linalg.cond(m) for m in Ms]
        gradient_step = learn_rate * N @ B_new
        print("WARNING: non-convex solver failed, trying adaptive version; min omegas:",
              model.omegas.min(), "Ms cond. numbers:", cond_numbers, "gradient_step:", gradient_step.max())
        return _solve_for_B_adaptive(model, sample_covariances, n_obs, debug)


def _solve_for_B_adaptive(model, sample_covariances, n_obs, debug=False):
    """Given all other parameters of the model, solve the convex
    optimization problem to estimate B using an adaptive gradient
    descent procedure.

    Parameters
    ----------
    model : Model()
        An instance of `model.Model` containing the parameters
        representing the model.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
        The number of observations available from each environment
        (i.e. the sample size).

    Returns
    -------
    B : numpy.ndarray
        The estimated connectivity matrix between observed variables, where
        B[i,j] != 0 => i -> j.

    """

    # Loss (likelihood) function that we will use to adapt the learning rate
    def likelihood(B_flattened):
        B = np.eye(model.p) - \
            B_flattened.reshape((model.p, model.p), order='F')
        B = B.T
        new_model = Model(model.A, B, model.gamma, model.omegas, model.psis)
        return new_model.score(sample_covariances, n_obs)
    # Inverse noise-term covariance matrices entailed by the model,
    # for each environment
    Ms = model.inv_noise_term_covariances()
    # Build "N" matrix
    N = np.zeros((model.p**2, model.p**2))
    for k, sigma in enumerate(sample_covariances):
        S = np.kron(scipy.linalg.sqrtm(sigma), scipy.linalg.sqrtm(Ms[k]))
        N += S.T @ S
    # Sparsity and 1-diagonal patterns
    zero_elements = np.logical_not(model.A.T).flatten(order='F')
    one_elements = np.eye(model.p).flatten(order='F').astype(bool)
    # Solve gradient descent
    current_B = np.random.uniform(size=(model.p**2, 1))
    current_B[zero_elements] = 0
    current_B[one_elements] = 1
    current_score = likelihood(current_B)
    learn_rate = 1
    while True:
        next_B = current_B - learn_rate * N @ current_B
        next_B[zero_elements] = 0
        next_B[one_elements] = 1
        next_score = likelihood(next_B)
        # Adapt learning rate if new score is worse
        if next_score > current_score:
            learn_rate /= 2
            print("  Adjusting learning rate to %s" %
                  learn_rate) if debug else None
            continue
        # Compute stopping condition
        print("Dist=", abs(current_B - next_B).max(),
              "Score=", next_score) if debug else None
        if abs(current_B - next_B).max() < 1e-5:
            break
        else:
            current_B, current_score = next_B, next_score
    # Unflatten B and return
    final_B = np.eye(model.p) - \
        current_B.reshape((model.p, model.p), order='F')
    assert np.isfinite(final_B).all()
    return final_B.T


def _solve_for_B_cvx(model, sample_covariances, n_obs, debug=False):
    """Given all other parameters of the model, solve the convex
    optimization problem to estimate B.

    Parameters
    ----------
    model : Model()
        An instance of `model.Model` containing the parameters
        representing the model.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
        The number of observations available from each environment
        (i.e. the sample size).

    Returns
    -------
    B : numpy.ndarray
        The estimated connectivity matrix between observed variables, where
        B[i,j] != 0 => i -> j.

    """
    # Inverse noise-term covariance matrices entailed by the model,
    # for each environment
    Ms = model.inv_noise_term_covariances()
    # Optimization variable: B
    B = cp.Variable((model.p, model.p))
    # Build constraints
    mask = np.logical_not(model.A)
    sparsity_constraint = B[mask] == 0
    constraints = [sparsity_constraint]
    # Build objective: score
    #   Note: since the log term does not depend on B,
    #   we only optimize the trace term
    score = 0
    I_B = (np.eye(model.p) - B.T)
    for i, sigma in enumerate(sample_covariances):
        # Note: M and Sigma are positive semi-definite, and we can
        # re-write the trace term as the frobenius norm of the "aux"
        # matrix, see
        # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
        aux = scipy.linalg.sqrtm(Ms[i]) @ I_B @ scipy.linalg.sqrtm(sigma)
        score += cp.norm(aux, 'fro')**2 * n_obs[i]
    score /= sum(n_obs)
    # Construct problem and solve it
    objective = cp.Minimize(score)
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    # Return
    B = B.value
    if not np.allclose(B[mask], 0):
        for M in Ms:
            print(np.diag(M).min())
        print("eig. sam. covariances")
        for sigma in sample_covariances:
            print(np.linalg.eig(sigma)[0].min())
        print("B")
        print(B[mask].max())
    assert np.allclose(B[mask], 0)
    B[mask] = 0
    return B


def _solve_for_gamma(model, sample_covariances, n_obs, threshold_score, learning_rate, verbose=0):
    """Given all other parameters of the model, find a gamma that
    (locally) minimizes the score using gradient descent.

    Parameters
    ----------
    model : Model()
        An instance of the `model.Model` class containing the model's
        parameters. `model.gamma` will be used as the starting point
        for the gradient descent algorithm.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
        The number of observations available from each environment
       (i.e. the sample size).
    threshold_score : float
        For the gradient descent subroutines, if change in score
        between successive iterations is below this threshold, stop.
    learning_rate : float
        The initial learning rate (factor by which gradient is
        multiplied) of the gradient descent procedure.
    verbose : int, default=0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    gamma : numpy.ndarray
        The latent effects matrix that (locally) minimizes the score
        given all other parameters.

    """
    # Construct gradient function for current parameters
    I_B = (np.eye(model.p) - model.B.T)
    Ns = np.array([I_B @ sigma @ I_B.T for sigma in sample_covariances])

    def gradients(args):
        gamma = args[0]
        new_model = Model(model.A, model.B, gamma, model.omegas, model.psis)
        Ms = new_model.inv_noise_term_covariances()
        gradient = np.zeros_like(gamma)

        # Construct the gradient element by element
        for l in range(model.l):
            for k in range(model.p):
                gradient_lk = 0

                # Compute each environment's contribution to the
                # gradient
                for h, (M, N) in enumerate(zip(Ms, Ns)):
                    # 1. Construct T (see gradient equations)
                    T = np.zeros((model.p, model.p))
                    # T[k,k] = gamma[l,k] * model.psis[h] this line was not necessary; I leave it here for discussion
                    for j in range(model.p):
                        T[k, j] = gamma[l, j] * model.psis[h, l]  # TODO: Check!
                    # assert T[k,k] == gamma[l,k] * model.psis[h]
                    T = T + T.T
                    # Compute this environment's contribution
                    gradient_lk += n_obs[h] * \
                        (np.trace(M @ T) - np.trace(M @ N @ M @ T))
                gradient_lk /= sum(n_obs)
                # Finish by storing this component's gradient
                gradient[l, k] = gradient_lk
        return [gradient]

    # Construct loss function
    def loss(args):
        gamma = args[0]
        new_model = Model(model.A, model.B, gamma, model.omegas, model.psis)
        return new_model.score(sample_covariances, n_obs)

    # Run gradient descent
    new_gamma, _ = _gradient_descent(
        loss, gradients, [model.gamma], threshold_score, learning_rate, verbose=verbose)
    return new_gamma[0]


def _solve_for_rest(model, I, sample_covariances, n_obs, psi_max, psi_fixed, threshold_score, learning_rate, verbose=0):
    """Given all other parameters of the model, find the omegas/psis that
    (locally) minimize the score using gradient descent.

    Parameters
    ----------
    model : Model()
        An instance of the `model.Model` class containing the model's
        parameters. `model.psis` and `model.omegas` will be used as
        the starting point for the gradient descent algorithm.
    I : set()
        The observed variables for which interventions are
        allowed. This will affect the way in which the gradient for
        the noise term variances (omegas) is computed.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
        The number of observations available from each environment
       (i.e. the sample size).
    psi_max : float
       The maximum allowed change in variance between environments for
       the hidden variables.
    psi_fixed : bool, default=False
       If `True`, impose the additional constraint that the hidden
       variables have all the same variance.
    threshold_score : float
        For the gradient descent subroutines, if change in score
        between successive iterations is below this threshold, stop.
    learning_rate : float
        The initial learning rate (factor by which gradient is
        multiplied) of the gradient descent procedure.
    verbose : int, default=0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

    Returns
    -------
    omegas : numpy.ndarray
        The noise term covariances which (locally) minimize the score.
    psis : numpy.ndarray
        The latent variable variances which (locally) minimize the score.

    """

    # Construct gradient function for current parameters
    I_B = (np.eye(model.p) - model.B.T)
    Ns = np.array([I_B @ sigma @ I_B.T for sigma in sample_covariances])

    def gradients(args):
        omegas, psis = args[0], args[1]
        new_model = Model(model.A, model.B, model.gamma, omegas, psis)
        Ms = new_model.inv_noise_term_covariances()

        # Construct gradients for omegas and psis
        gradient_omegas = np.zeros_like(omegas)
        gradient_psis = np.zeros_like(psis)
        for h, (M, N) in enumerate(zip(Ms, Ns)):
            # Gradient for omegas
            gradient_omegas[h, :] = n_obs[h] / \
                sum(n_obs) * np.diag(M - M @ N @ M)
            # Gradient for psis (compute for each element)
            for k in range(model.l):
                T = np.zeros((model.p, model.p), dtype=float)
                for i in range(model.p):
                    for j in range(model.p):
                        T[i, j] = model.gamma[k, i] * model.gamma[k, j]
                gradient_psis[h, k] = n_obs[h] / \
                    sum(n_obs) * (np.trace(M @ T) - np.trace(M @ N @ M @ T))
        # Impose constraint set by I: if j is not in I, the variance
        # (and thus gradient) across environments is the same
        for j in set(range(model.p)) - I:
            gradient_omegas[:, j] = gradient_omegas[:, j].mean()
        # If psi_fixed = True, impose constraint that psis are all equal
        if psi_fixed:
            for j in range(model.l):
                gradient_psis[:, j] = gradient_psis[:, j].mean()
        return [gradient_omegas, gradient_psis]

    # Construct loss function
    def loss(args):
        omegas, psis = args[0], args[1]
        new_model = Model(model.A, model.B, model.gamma, omegas, psis)
        return new_model.score(sample_covariances, n_obs)

    # Set bounds for omegas and psis
    bounds = [(0, np.Inf)]
    bounds += [(0, np.Inf)] if psi_max is None else [(0, psi_max)]
    # Run gradient descent
    result, _ = _gradient_descent(loss, gradients, [
        model.omegas, model.psis], threshold_score, learning_rate, bounds, verbose=verbose)
    return result[0], result[1]


# --------------------------------------------------------------------
# Functions to initialize the different parameters of the model

def _initialize_B(A, sample_covariances, n_obs, init='avg'):
    """Initialize the connectivity matrix B according to the DAG adjacency A.

    PRECONDITION: `A` corresponds to a DAG adjacency (should be
    checked in calling function).

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency of the given DAG, where A0[i,j] != 0 implies i -> j.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
        The number of observations available from each environment
        (i.e. the sample size).
    init : {'obs', 'avg'}, default='obs'
        If the sample covariance from the observational environment
        (taken to be at index 0) ('obs') or the average sample
        covariance ('avg'), weighted by sample size, should be used for
        the initialization.

    Raises
    ------
    ValueError :
        If the value for init is not 'obs' or 'avg'.

    Returns
    -------
    B : numpy.ndarray
        The connectivity (weight) matrix.

    Example
    -------

    >>> _initialize_B(A, sample_covariances, n_obs, 'obs')
    array([[0.        , 0.        , 0.        , 0.53064129],
           [0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.89673092, 0.        , 0.        ]])

    >>> _initialize_B(A, sample_covariances, n_obs, 'avg')
    array([[0.        , 0.        , 0.        , 0.52838027],
           [0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.89501327, 0.        , 0.        ]])

    >>> _initialize_B(A, sample_covariances, n_obs, 'test')
    Traceback (most recent call last):
    ...
    ValueError: Unrecognized value "test" for parameter init

    """
    # Check input: init
    if init not in ['obs', 'avg']:
        raise ValueError('Unrecognized value "%s" for parameter init' % init)
    (e, p, _) = sample_covariances.shape  # num. of environments and variables
    B = np.zeros((p, p), dtype=float)
    # Decide if we compute the weighted, average sample covariance matrix
    if init == 'obs':
        init_covariance = sample_covariances[0]
    elif init == 'avg':
        scaled = sample_covariances * np.reshape(n_obs, (e, 1, 1))
        init_covariance = scaled.sum(axis=0) / sum(n_obs)
        # Regress B according to A
    for j in range(p):
        pa = list(np.where(A[:, j] != 0)[0])
        B[pa, j] = _regress(j, pa, init_covariance)
    return B


def _initialize_gamma(B, num_latent, sample_covariances, n_obs, init='avg'):
    """Given the connectivity matrix B and the assumed number of latent
    variables, initialize the matrix of latent effects.

    Parameters
    ----------
    B : numpy.ndarray
        The connectivity matrix between observed variables, where
        `B[i,j] != 0 => i -> j`.
    num_latent : int
       The assumed number of hidden variables.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
       The number of observations available from each environment
       (i.e. the sample size).
    init : {'obs', 'avg'}, default='obs'
       If the sample covariance from the observational environment
       (taken to be at index 0) ('obs') or the average sample
       covariance ('avg'), weighted by sample size, should be used for
       the initialization.

    Raises
    ------
    ValueError :
        If the value for init is not 'obs' or 'avg'.

    Returns
    -------
    gamma : numpy.ndarray
        The matrix of latent effects, i.e. connectivity matrix from
        latent to observed variables.

    Example
    -------

    >>> _initialize_gamma(B, 3, sample_covariances, n_obs, 'obs')
    array([[-0.00624498, -0.01088502, -0.00803778, -1.31568027],
           [ 0.00559699,  0.01425682,  1.22675101, -0.00763901],
           [ 0.0068256 ,  1.20568705, -0.01410495, -0.00992124]])


    >>> _initialize_gamma(B, 3, sample_covariances, n_obs, 'avg')
    array([[ 2.59412474e-05,  1.64447527e+00, -1.03597867e-03,
            -1.31391844e-03],
           [-2.36602469e-04,  1.14426317e-03, -1.85525853e-03,
             1.43359610e+00],
           [-1.09771216e-02, -7.73792530e-04, -1.22655061e+00,
            -1.58850898e-03]])


    >>> _initialize_gamma(B, 3, sample_covariances, n_obs, 'test')
    Traceback (most recent call last):
    ...
    ValueError: Unrecognized value "test" for parameter init

    """
    # Check input: init
    if init not in ['obs', 'avg']:
        raise ValueError('Unrecognized value "%s" for parameter init' % init)
    (e, p, _) = sample_covariances.shape  # num. of environments and variables
    # Decide if we compute the weighted, average sample covariance matrix
    if init == 'obs':
        init_covariance = sample_covariances[0]
    elif init == 'avg':
        scaled = sample_covariances * np.reshape(n_obs, (e, 1, 1))
        init_covariance = scaled.sum(axis=0) / sum(n_obs)
    # Initalize gamma via SVD
    I_B = np.eye(p) - B.T
    U, S, _ = np.linalg.svd(I_B @ init_covariance @ I_B.T)
    gamma = U[:, 0:num_latent] * (S[0:num_latent] ** 0.5)
    return gamma.T


def _initialize_omegas(B, I, sample_covariances, n_obs, init='avg'):
    """
    Given the connectivity matrix B and constraints on the
    intervention targets I, initialize the noise term variances.

    Parameters
    ----------
    B : numpy.ndarray
        The connectivity matrix between observed variables, where
        B[i,j] != 0 => i -> j.
    gamma : numpy.ndarrayg
        The matrix of latent effects, where gamma[i,j] != 0 => i -> j.
    I : list of sets
        The intervention targets in each environment, i.e. for which
        variables the noise term variance is allowed to vary between
        environments.
    sample_covariances : numpy.ndarray
        A 3-dimensional array containing the estimated sample
        covariances of the observed variables for each environment.
    n_obs : list of ints
       The number of observations available from each environment
       (i.e. the sample size).
    init : {'obs', 'avg'}, default='obs'
       If the sample covariance from the observational environment
       (taken to be at index 0) ('obs') or the average sample
       covariance ('avg'), weighted by sample size, should be used for
       the initialization.

    Raises
    ------
    ValueError :
        If the value for init is not 'obs' or 'avg'.

    Returns
    -------
    omegas : numpy.ndarray
        A `e x p` float array containing the estimated noise term variances.

    Example
    -------
    >>> _initialize_omegas(B, set(), sample_covariances, n_obs, 'obs')
    array([[1.43084781, 2.70430082, 1.50454551, 2.05520205],
           [1.43084781, 2.70430082, 1.50454551, 2.05520205],
           [1.43084781, 2.70430082, 1.50454551, 2.05520205],
           [1.43084781, 2.70430082, 1.50454551, 2.05520205]])

    >>> _initialize_omegas(B, {1}, sample_covariances, n_obs, 'avg')
    array([[1.43084781, 1.4540421 , 1.50454551, 2.05520205],
           [1.43084781, 1.45674425, 1.50454551, 2.05520205],
           [1.43084781, 6.45012445, 1.50454551, 2.05520205],
           [1.43084781, 1.45629248, 1.50454551, 2.05520205]])

    >>> _initialize_omegas(B, {1}, sample_covariances, n_obs, 'test')
    Traceback (most recent call last):
    ...
    ValueError: Unrecognized value "test" for parameter init

    """
    # Check input: init
    if init not in ['obs', 'avg']:
        raise ValueError('Unrecognized value "%s" for parameter init' % init)
    (e, p, _) = sample_covariances.shape  # num. of environments and variables
    # Decide if we compute the weighted, average sample covariance matrix
    if init == 'obs':
        sigma = sample_covariances[0]
    elif init == 'avg':
        scaled = sample_covariances * np.reshape(n_obs, (e, 1, 1))
        sigma = scaled.sum(axis=0) / sum(n_obs)
        # Then, first compute the noise term variances without constraints,
        # i.e. as if all variables are allowed to receive interventions in
        # all environments
    omegas_wo_constraints = np.zeros((e, p), dtype=float)
    I_B = np.eye(p) - B.T
    for i, sigma in enumerate(sample_covariances):
        omegas_wo_constraints[i, :] = np.diag(I_B @ sigma @ I_B.T)
        # Then set the variances as the weighted average (by sample size)
        # across environment for those variables not in I
    omegas = np.zeros_like(omegas_wo_constraints)
    for j in range(p):
        if j in I:
            omegas[:, j] = omegas_wo_constraints[:, j]
        else:
            avg = (omegas_wo_constraints[:, j] * n_obs).sum() / sum(n_obs)
            omegas[:, j] = avg
    return omegas


def _initialize_psis(num_envs, num_latent):
    """
    Initialize the variances of the latent variables (for now, assumed
    to be iid).

    Parameters
    ----------
    num_envs : int
        The number of environments.
    num_latent : int
        The number of latent variables.g

    Returns
    -------
    psis : numpy.ndarray
        An array with the initialized psis (one float per environment).

    Examples
    --------
    >>> _initialize_psis(2, 3)
    array([[1., 1., 1.],
           [1., 1., 1.]])
    """
    psis = np.ones((num_envs, num_latent), dtype=float)
    return psis

# --------------------------------------------------------------------
# Support functions


def _is_within_bounds(X, Bounds):
    """Checks that elements of the arrays in X are all (strictly) within the given bounds.

    Parameters
    ----------
    X : list of numpy.ndarray
    Bounds : list of tuples
        The bounds for each array; must be of equal length to X.

    Returns
    -------
    within_bounds : bool
        True if all the elements of the arrays in X are within their
        correspoding bounds.

    """
    within_bounds = True
    for x, bounds in zip(X, Bounds):
        if bounds[0] > bounds[1]:
            raise ValueError("Lower bound cannot be larger than upper bound.")
        if (x <= bounds[0]).any() or (x >= bounds[1]).any():
            within_bounds = False
            break
    return within_bounds


def _gradient_descent(loss_fun, gradient_fun, X_init, threshold_loss, learning_rate, bounds=None, verbose=0, kappa_zero_point=1e-50):
    """Perform gradient descent to minimize a given loss function.

    Parameters
    ----------
    loss_fun : function
        The loss or objective function of X which we aim to
        minimize. It must take a list of numpy.ndarray representing
        the arguments of the loss function, and return a float.
    gradient_fun : function
       The function to compute the gradient of the loss at a
       particular X. Must take and return a list of numpy.ndarray.
    X_init : list of numpy.ndarray
       The initial values for X.
    threshold_loss : float
       If the score improves below this threshold, stop the gradient
       descent procedure and return the current minimizer.
    learning_rate : float
       The starting learning rate, i.e. factor by which the gradient
       is scaled in each iteration.
    bounds : list of tuples, default=None
        The bounds for the elements of each array in X. If after
        applying the gradients elements land outside the bounds, we
        reduce kappa and repeat. If `bounds=None`, no checks are
        performed.
    verbose : int, default=0
       If debugging traces should be output showing the progress of
       the gradient descent. 0 means no traces, larger values
       correspond to increased verbosity.
    kappa_zero_point : float, default=1e-60
       The value under which kappa is considered zero; if kappa drops
       below or equal to this value, the gradient descent procedure
       considers the current solution as the maximizer.

    Returns
    -------
    X : list of numpy.ndarray
        The found minimizer.
    loss : float
        The value of the loss function at the minimizer.

    Raises
    ------
    ValueError:
        If enforce_non_negative is True and X_init contains negative
        elements.

    """

    # Check inputs: X_init
    if bounds is not None and not _is_within_bounds(X_init, bounds):
        raise ValueError(
            "Some elements of X_init are outside the required bounds.")

    current_X = X_init
    current_loss = loss_fun(current_X)
    kappa = 1.0
    i = 0

    print("  \nStarting gradient descent.") if verbose > 0 else None
    print("    X_init =%s\n" % X_init) if verbose > 1 else None

    moved = True  # To compute only gradients if we moved in the previous step
    while True:
        i += 1
        # Compute and apply gradient
        if moved:
            gradients = gradient_fun(current_X)
        # print(gradients)
        new_X = [x - kappa * learning_rate
                 * gradient for (x, gradient) in zip(current_X, gradients)]
        print("      gradients: %s" % gradients) if verbose > 2 else None

        # If we require the values of X to be bounded and some are
        # outside the bounds, reduce kappa by half and repeat
        # if bounds is not None:
        #     bounded_X = []
        #     for (x, bounds_x) in zip(new_X, bounds):
        #         lo_bound = np.ones_like(x) * bounds_x[0]
        #         hi_bound = np.ones_like(x) * bounds_x[1]
        #         bounded = np.minimum(x, hi_bound)
        #         bounded = np.maximum(bounded, lo_bound)
        #         bounded_X.append(bounded)
        #     new_X = bounded_X

        if bounds is not None and not _is_within_bounds(new_X, bounds):
            print(
                "WARNING: elements in X are outside their bounds") if verbose > 0 else None
            kappa /= 2
            moved = False
            continue

        # Compute new score and decide on next step
        new_loss = loss_fun(new_X)
        print("    Iter. %d: new - old: %s (thresh: %s, kappa: %s)" %
              (i, new_loss - current_loss, threshold_loss, kappa)) if verbose > 0 else None
        print("      %s" % new_X) if verbose > 2 else None
        # Option 1: Score improves below threshold; end gradient
        # descent
        if new_loss < current_loss and abs(new_loss - current_loss) < threshold_loss:
            print(
                "      Loss improved below threshold; stopping.") if verbose > 1 else None
            return new_X, new_loss
        # Option 2: Score improves above threshold; continue
        # gradient descent
        elif new_loss < current_loss:
            print("      Loss improved.") if verbose > 1 else None
            moved = True
            current_X, current_loss = new_X, new_loss
        # Option 3.1: Score increases and kappa is already zero (we
        # started at the highest scoring?)
        elif kappa * learning_rate <= kappa_zero_point:
            print("     WARNING: kappa * learning_rate reached zero (<= %s)" %
                  kappa_zero_point) if verbose > 0 else None
            return current_X, current_loss
        # Option 3.2: Score increases; reduce learning rate (through
        # kappa) and repeat
        else:
            moved = False
            print(
                "      Loss did not improve; reducing learning rate.") if verbose > 1 else None
            kappa /= 2


def _regress(j, pa, cov):
    # compute the regression coefficients from the
    # empirical covariance (scatter) matrix i.e. b =
    # Σ_{j,pa(j)} @ Σ_{pa(j), pa(j)}^-1
    return np.linalg.solve(cov[pa, :][:, pa], cov[j, pa])


# TODO
#    - decide if should use model.copy + set parameter or the constructor
#    - hierarchization of debug traces
#    - possibly refactor all functions into the Score class (must also adapt tests)


# --------------------------------------------------------------------
# Doctests
if __name__ == '__main__':
    import doctest
    A = np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0]])
    B = np.array([[0., 0., 0., 0.5284517],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0.89522034, 0., 0.]])
    sample_covariances = np.array([
        [[1.18154051e+00, 5.62861157e-01, 1.47446462e-03,
          6.26974179e-01],
         [5.62861157e-01, 3.11366562e+00, 2.52466553e-03,
            1.85075387e+00],
            [1.47446462e-03, 2.52466553e-03, 1.50520579e+00,
             2.14807856e-03],
            [6.26974179e-01, 1.85075387e+00, 2.14807856e-03,
             2.06388987e+00]],
        [[2.17597348e+00, 1.02758250e+00, 3.40364423e-05,
          1.14739485e+00],
         [1.02758250e+00, 3.32845211e+00, 3.17440652e-04,
            2.09187437e+00],
            [3.40364423e-05, 3.17440652e-04, 1.50305625e+00,
             -6.67585284e-04],
            [1.14739485e+00, 2.09187437e+00, -6.67585284e-04,
             2.33793813e+00]],
        [[1.18361276e+00, 5.56799188e-01, 3.51764554e-04,
          6.24536754e-01],
         [5.56799188e-01, 8.09468624e+00, -5.91543673e-03,
            1.84011457e+00],
            [3.51764554e-04, -5.91543673e-03, 1.50731571e+00,
             -2.54581781e-03],
            [6.24536754e-01, 1.84011457e+00, -2.54581781e-03,
             2.05891456e+00]],
        [[1.18226448e+00, 5.60098844e-01, 7.78629746e-04,
          6.25221215e-01],
         [5.60098844e-01, 4.14565413e+00, -1.25310329e-03,
            3.00512154e+00],
            [7.78629746e-04, -1.25310329e-03, 1.50260430e+00,
             -3.87781728e-04],
            [6.25221215e-01, 3.00512154e+00, -3.87781728e-04,
             3.35795466e+00]]])
    n_obs = [100] * len(sample_covariances)
    doctest.testmod(extraglobs={'A': A,
                                'B': B,
                                'sample_covariances': sample_covariances,
                                'rng': np.random.default_rng(42)},
                    verbose=True,
                    optionflags=doctest.ELLIPSIS)
