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

from utlvce.model import Model
from utlvce.score import Score, _Cache
import utlvce.score
import unittest
import numpy as np
import sempler.generators
import time
from termcolor import colored
import utlvce.utils as utils

NUM_GRAPHS = 3


# Tested functions


def spectral_norm(X):
    return np.linalg.norm(X, ord=2)


def sample_model(p, I, num_latent, e, var_lo, var_hi, B_lo, B_hi, random_state=42, verbose=0):
    rng = np.random.default_rng(random_state)
    B = sempler.generators.dag_avg_deg(
        p, 2.5, B_lo, B_hi, random_state=random_state)
    A = (B != 0).astype(int)
    gamma = rng.normal(0, 1 / np.sqrt(p), size=(num_latent, p))
    omegas = rng.uniform(var_hi * 1.5, var_hi * 3, size=(e, p))
    for j in range(p):
        if j not in I:
            omegas[:, j] = omegas[:, j].mean()
    psis = rng.uniform(var_lo, var_hi, size=(e, num_latent))
    model = Model(A, B, gamma, omegas, psis)
    print("  Spec. norm gamma:", spectral_norm(
        gamma.T @ np.diag(psis[0]) @ gamma)) if verbose else None
    print("  Spec. norm omegas:", spectral_norm(omegas)) if verbose else None
    return model


# def order_omegas(omegas):
#     variances = np.var(omegas, axis=0)
#     return list(np.argsort(variances)[::-1])


def order_omegas(omegas):
    variances = np.var(omegas, axis=0)
    idx = np.argsort(variances)[::-1]
    variances = np.sort(variances)[::-1]
    string = "[ "
    for (i, v) in zip(idx, variances):
        string += "%d(%0.2f)\t" % (i, v)
    string += "]"
    return string

# ---------------------------------------------------------------------
# Tests for the DAG scoring procedure


class ScoreDagTests(unittest.TestCase):
    """Test the whole alternating minimization procedure"""

    def test_population_init(self):
        """For random graphs, test the procedure on the true connectivity
        matrix with the population covariances, with population values
        as starting point.

        TEST PASSES IF:
          - estimated B respects sparsity in A
          - elements of estimated B are at most 1e-6 from population B
          - estimate score is <1e-15 away from score of population parameters

        WARNING IF:
          - score of estimate is worse than score of pop parameters
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing alternating procedure with population covariances + init")
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)
            pop_score = model.score(sample_covariances, n_obs)

            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=1)

            estimated_model, estimate_score = score_class.score_dag(
                model.A, set(range(model.p)), init_model=model, verbose=0)

            estimated_B = estimated_model.B
            print("  diff. wrt. population B",
                  abs(model.B - estimated_B).max())
            diff_scores = estimate_score - pop_score
            text = "  diff. wrt. to pop score = %s - pop. score = %s" % (
                diff_scores, pop_score)
            thresh = 0
            if diff_scores > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            mask = np.logical_not(model.A)
            self.assertTrue(np.allclose(estimated_B[mask], 0))
            self.assertLessEqual(estimate_score - pop_score, 1e-6)
            self.assertLess(abs(model.B - estimated_B).max(), 1e-6)

    def test_population(self):
        """For random graphs, test the procedure on the true connectivity
        matrix with the population covariances.

        TEST PASSES IF:
          - estimated B respects sparsity in A
          - score is <5e-2 from score of population parameters

        WARNING IF:
          - estimated B is more than 0.07 away from population B
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing alternating procedure with population covariances")
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            # print(model)
            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)
            pop_score = model.score(sample_covariances, n_obs)
            start = time.time()
            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=10)

            estimated_model, estimate_score = score_class.score_dag(
                model.A, set(range(model.p)), verbose=0)
            print("  Done in %0.2f seconds" % (time.time() - start))
            print("  Testing cache...", end=" ")
            start = time.time()
            score_class.score_dag(model.A, set(range(model.p)), verbose=0)
            print("  Done in %0.2f seconds" % (time.time() - start))
            start = time.time()
            estimated_B = estimated_model.B
            diff_wrt_pop = abs(model.B - estimated_B).max()
            text = "  diff. wrt. population B = %s" % diff_wrt_pop
            thresh = 0.07
            if diff_wrt_pop > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            print("  diff. wrt. to pop score =",
                  estimate_score - pop_score, pop_score)
            self.assertLess(estimate_score - pop_score, 5e-2)
            # print("Population B\n",model.B)
            # print("Estimated B\n", estimated_B)
            mask = np.logical_not(model.A)
            self.assertTrue(np.allclose(estimated_B[mask], 0))

    def test_wrong_dag_population(self):
        """For random graphs, test the procedure on wrong connectivity
        matrix with the population covariances.

        TEST PASSES IF:
          - estimated B respects sparsity in A

        WARNING IF:
          - score of estimate is > score of population parameters by less than 1
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print(
            "Testing alternating procedure with population covariances and wrong adjacency")
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            # print(model)
            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)
            pop_score = model.score(sample_covariances, n_obs)

            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-4, learning_rate=10)

            # Generate a different adjacency
            A = model.A
            while (A == model.A).all():
                A = sempler.generators.dag_avg_deg(p, 3, 1, 1)

            start = time.time()
            estimated_model, estimate_score = score_class.score_dag(
                A, set(range(model.p)), verbose=0)
            print("  Done in %0.2f seconds" % (time.time() - start))
            estimated_B = estimated_model.B
            mask = np.logical_not(A)
            self.assertTrue(np.allclose(estimated_B[mask], 0))
            diff_wrt_pop = abs(model.B - estimated_B).max()
            print("  diff. wrt. population B = %s" % diff_wrt_pop)
            score_diff = estimate_score - pop_score
            thresh = 1
            text = "  diff. wrt. to pop score = %s, %s" % (
                score_diff, pop_score)
            if score_diff < thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            # print("Population B\n",model.B)
            # print("Estimated B\n", estimated_B)

    def test_finite_sample_pop_init(self):
        """For random graphs, test the procedure on the true connectivity
        matrix, with a finite sample but setting the population
        parameters as starting point.

        TEST PASSES IF:
          - estimated B respects sparsity in A
          - score is better than score of population parameters

        WARNING IF:
          - estimated B is more than 0.07 away from population B
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        n = 1000
        print("\n" + "-" * 70)
        print("Testing alternating procedure with a finite sample (n_e = %d) but pop. params. as init" % n)
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            # print(model)
            _, sample_covariances, n_obs = model.sample(n)
            pop_score = model.score(sample_covariances, n_obs)
            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=10)

            start = time.time()
            estimated_model, estimate_score = score_class.score_dag(
                model.A, set(range(model.p)), init_model=model, verbose=0)
            print("  Done in %0.2f seconds" % (time.time() - start))
            # Check distance of estimated B and pop. B
            estimated_B = estimated_model.B
            diff_wrt_pop = abs(model.B - estimated_B).max()
            text = "  diff. wrt. population B = %s" % diff_wrt_pop
            thresh = 0.07
            if diff_wrt_pop > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            # Check distance of estimate's score and pop. score
            diff_scores = estimate_score - pop_score
            text = "  diff. wrt. to pop score = %s - pop. score = %s" % (
                diff_scores, pop_score)
            if diff_scores > 0:
                print("*** (> 0)" % thresh, colored(text, 'red'))
            else:
                print(text)
            self.assertLess(estimate_score - pop_score, 0)
            # print("Population B\n",model.B)
            # print("Estimated B\n", estimated_B)
            mask = np.logical_not(model.A)
            self.assertTrue(np.allclose(estimated_B[mask], 0))

    def test_finite_sample(self):
        """For random graphs, test the procedure on the true connectivity
        matrix, with a finite sample.

        TEST PASSES IF:
          - estimated B respects sparsity in A

        WARNING IF:
          - score is larger than score of population parameters
          - estimated B is more than 0.07 away from population B
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        n = 1000
        print("\n" + "-" * 70)
        print("Testing alternating procedure with a finite sample (n_e = %d)" % n)
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            # print(model)
            _, sample_covariances, n_obs = model.sample(n, random_state=i)
            pop_score = model.score(sample_covariances, n_obs)

            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=10)

            start = time.time()
            estimated_model, estimate_score = score_class.score_dag(
                model.A, set(range(model.p)), verbose=0)
            print("  Done in %0.2f seconds" % (time.time() - start))
            # Check distance of estimated B and pop. B
            estimated_B = estimated_model.B
            diff_wrt_pop = abs(model.B - estimated_B).max()
            text = "  diff. wrt. population B = %s" % diff_wrt_pop
            thresh = 0.07
            if diff_wrt_pop > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            # Check distance of estimate's score and pop. score
            diff_scores = estimate_score - pop_score
            text = "  diff. wrt. to pop score = %s - pop. score = %s" % (
                diff_scores, pop_score)
            thresh = 0
            if diff_scores > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)

            # self.assertLess(abs(estimate_score - pop_score), 1e-3)
            # print("Population B\n",model.B)
            # print("Estimated B\n", estimated_B)
            mask = np.logical_not(model.A)
            self.assertTrue(np.allclose(estimated_B[mask], 0))

    def test_finite_sample_latent_vs_not(self):
        """For random graphs, test the procedure on the true connectivity
        matrix, with a finite sample.

        TEST PASSES IF:
          - estimated B respects sparsity in A
          - score when using latents is better than without them

        WARNING IF:
          - L-infinity distance to population B is smaller using latents than without them
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        n = 1000
        print("\n" + "-" * 70)
        print(
            "Testing procedure with vs. without fixed-psi on a finite sample (n_e = %d)" % n)
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            # print(model)
            _, sample_covariances, n_obs = model.sample(n, random_state=i)
            pop_score = model.score(sample_covariances, n_obs)
            start = time.time()

            # Estimate with latent variables
            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=10)
            estimated_model_latents, estimate_score_latents = score_class.score_dag(
                model.A, set(range(model.p)), verbose=0)

            # Estimate without latent variables
            score_class = Score((sample_covariances, n_obs),
                                num_latent=0, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=10)
            estimated_model_no_latents, estimate_score_no_latents = score_class.score_dag(
                model.A, set(range(model.p)), verbose=0)

            print("  Done in %0.2f seconds" % (time.time() - start))
            # Check difference in scores
            diff_scores = estimate_score_latents - estimate_score_no_latents
            text = "  score w. latents < score wo. latents: %s - diff = %s - pop. score = %s" % (
                estimate_score_latents < estimate_score_no_latents, diff_scores, pop_score)
            thresh = 0
            if diff_scores > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            self.assertLess(estimate_score_latents, estimate_score_no_latents)
            # Check difference in distances to population B (estimate
            # with latents should be closer)
            dist_B_latents = abs(estimated_model_latents.B - model.B).max()
            dist_B_no_latents = abs(
                estimated_model_no_latents.B - model.B).max()
            diff_dists = dist_B_latents - dist_B_no_latents
            text = "  dist. to population B w. latents < wo. latents: %s - diff = %s" % (
                dist_B_latents < dist_B_no_latents, diff_dists)
            thresh = 0
            if diff_dists > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            # self.assertLess(dist_B_latents, dist_B_no_latents)
            # Check sparsity is respected
            mask = np.logical_not(model.A)
            self.assertTrue(np.allclose(estimated_model_latents.B[mask], 0))
            self.assertTrue(np.allclose(estimated_model_no_latents.B[mask], 0))

    def test_finite_sample_psi_fixed_vs_not(self):
        """For random graphs, test the procedure on the true connectivity
        matrix, with a finite sample.

        TEST PASSES IF:
          - estimated B respects sparsity in A
          - psis of model with fixed_psi = True are indeed all the same
          - score without constraint that hidden variances to be equal is better than with it

        WARNING IF:
          - L-infinity distance to population B is better without constraints than with fixed psi
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        n = 1000
        print("\n" + "-" * 70)
        print(
            "Testing procedure with vs. without latents on a finite sample (n_e = %d)" % n)
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            # print(model)
            _, sample_covariances, n_obs = model.sample(n, random_state=i)
            pop_score = model.score(sample_covariances, n_obs)
            start = time.time()

            # Estimate with latent variables
            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=False,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=10)
            estimated_model, estimate_score = score_class.score_dag(
                model.A, set(range(model.p)), verbose=0)

            # Estimate without latent variables
            score_class = Score((sample_covariances, n_obs),
                                num_latent=num_latent, psi_max=None, psi_fixed=True,
                                max_iter=1000, threshold_dist_B=1e-3,
                                threshold_fluctuation=1, max_fluctuations=10,
                                threshold_score=1e-6, learning_rate=10)
            estimated_model_fixed, estimate_score_fixed = score_class.score_dag(
                model.A, set(range(model.p)), verbose=0)

            print("  Done in %0.2f seconds" % (time.time() - start))
            # print(estimated_model.psis)
            # print(estimated_model_fixed.psis)
            self.assertTrue(
                (estimated_model_fixed.psis[0, :] == estimated_model_fixed.psis).all())
            # Check difference in scores
            diff_scores = estimate_score - estimate_score_fixed
            text = "  score unconstrained < score fixed psi: %s - diff = %s - pop. score = %s" % (
                estimate_score < estimate_score_fixed, diff_scores, pop_score)
            thresh = 0
            if diff_scores > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            self.assertLess(estimate_score, estimate_score_fixed)
            # Check difference in distances to population B (estimate
            # with latents should be closer)
            dist_B = abs(estimated_model.B - model.B).max()
            dist_B_fixed = abs(estimated_model_fixed.B - model.B).max()
            diff_dists = dist_B - dist_B_fixed
            text = "  dist. to population B unconstrained < fixed psi: %s - diff = %s" % (
                dist_B < dist_B_fixed, diff_dists)
            thresh = 0
            if diff_dists > thresh:
                print("*** (> %s)" % thresh, colored(text, 'red'))
            else:
                print(text)
            # self.assertLess(dist_B, dist_B_fixed)
            # Check that sparsity is respected
            mask = np.logical_not(model.A)
            self.assertTrue(np.allclose(estimated_model.B[mask], 0))
            self.assertTrue(np.allclose(estimated_model_fixed.B[mask], 0))

    def test_with_interventions_population(self):
        """Generate the population covariances from a random graph and some
        random intervention targets; check that the score remains
        close when applying the alternating procedure to I-equivalent
        DAGs, when setting I to I* and to the full set.

        TEST PASSES IF:
          - estimated B respects sparsity in A
          - score is <5e-2 from score of true population parameters
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        size_I = (0, 3)
        print("\n" + "-" * 70)
        print("Testing alternating procedure with population covariances and different interventions")
        for i in range(G):
            # Sample intervention targets
            true_I = set(
                sempler.generators.intervention_targets(p, 1, size_I)[0])
            # Build model
            model = sample_model(p, true_I, num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)
            pop_score = model.score(sample_covariances, n_obs)
            # Get equivalent DAGs
            icpdag = utils.dag_to_icpdag(model.A, true_I)
            equivalent_dags = utils.all_dags(icpdag)
            print("\nGraph %d - I* = %s - %d equivalent DAGs" %
                  (i, true_I, len(equivalent_dags)))
            for k, dag in enumerate(equivalent_dags):
                for I in [true_I, set(range(p))]:
                    print("  Eq. graph %d/%d; I = %s" %
                          (k + 1, len(equivalent_dags), I))
                    start = time.time()
                    score_class = Score((sample_covariances, n_obs),
                                        num_latent=num_latent, psi_max=None, psi_fixed=False,
                                        max_iter=1000, threshold_dist_B=1e-3,
                                        threshold_fluctuation=1, max_fluctuations=10,
                                        threshold_score=1e-6, learning_rate=10)
                    estimated_model, estimate_score = score_class.score_dag(
                        dag, I, verbose=0)
                    mask = np.logical_not(dag)
                    print("    Done in %0.2f seconds" % (time.time() - start))
                    print("    diff. wrt. to pop score =",
                          estimate_score - pop_score, pop_score)
                    self.assertLess(estimate_score - pop_score, 5e-2)
                    self.assertTrue(np.allclose(estimated_model.B[mask], 0))

    def test_with_interventions_finite_sample(self):
        """Generate data from a random graph and some random intervention
        targets; check that the score remains close when applying the
        alternating procedure to I-equivalent DAGs, when setting I to I* and
        to the full set.

        TEST PASSES IF:
          - For each estimated model, score is <5e-2 from score of true population parameters
          - Estimate score of true DAG + full_I is <5e-2 away from estimate score of true DAG + true_I
          - Estimate score of equivalent DAGs is <5e-2 away from estimate score of true DAG + true_I
        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        n = 1000
        size_I = (1, 4)
        print("\n" + "-" * 70)
        print(
            "Testing alternating procedure with finite sample and different interventions")
        for i in range(G):
            # Sample intervention targets
            true_I = set(sempler.generators.intervention_targets(
                p, 1, size_I, random_state=i)[0])
            # Build model
            model = sample_model(p, true_I, num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            _, sample_covariances, n_obs = model.sample(n, random_state=i)
            pop_score = model.score(sample_covariances, n_obs)
            # Get equivalent DAGs (put true DAG in first position)
            icpdag = utils.dag_to_icpdag(model.A, true_I)
            equivalent_dags = utils.all_dags(icpdag)
            idx = utils.member(equivalent_dags, model.A)
            equivalent_dags = [equivalent_dags[idx]] + [equivalent_dags[k]
                                                        for k in range(len(equivalent_dags)) if k != idx]
            print("\n\nGraph %d - I* = %s - %d equivalent DAGs" %
                  (i, true_I, len(equivalent_dags)))
            scores = []
            for k, dag in enumerate(equivalent_dags):
                # Build scores for all DAGs
                estimated_omegas = []
                estimated_gammas = []
                estimated_psis = []
                estimated_Bs = []
                # Score for true-I and all-I
                for I in [true_I, set(range(p))]:
                    print("    Scoring graph %d/%d with I = %s" %
                          (k + 1, len(equivalent_dags), I))
                    start = time.time()
                    score_class = Score((sample_covariances, n_obs),
                                        num_latent=num_latent, psi_max=2, psi_fixed=False,
                                        max_iter=1000, threshold_dist_B=1e-3,
                                        threshold_fluctuation=1, max_fluctuations=10,
                                        threshold_score=1e-6, learning_rate=10)
                    estimated_model, estimate_score = score_class.score_dag(
                        dag, I, verbose=0)
                    estimated_psis.append(estimated_model.psis)
                    estimated_gammas.append(estimated_model.gamma)
                    estimated_omegas.append(estimated_model.omegas)
                    estimated_Bs.append(estimated_model.B)
                    print("      Done in %0.2f seconds" %
                          (time.time() - start))
                    print("      diff. wrt. to score of pop. params=",
                          estimate_score - pop_score, pop_score)
                    self.assertLess(estimate_score - pop_score, 5e-2)
                    scores.append(estimate_score)
                # Show L1 distance between estimated omegas and Bs for
                # true-I and all-I
                # Print the variances of omegas and the corresponding ranking
                print("      Ranking of omega's variances for I* and I-full")
                print("        ", order_omegas(estimated_omegas[0]))
                print("        ", order_omegas(estimated_omegas[1]))
                # Print difference in estimates between I* and I-full
                estimated_Bs = np.array(estimated_Bs)
                estimated_gammas = np.array(estimated_gammas)
                estimated_omegas = np.array(estimated_omegas)
                estimated_psis = np.array(estimated_psis)
                print("      Max. difference between I* and full-I estimates for")
                print("        B = %s" %
                      abs(estimated_Bs[0] - estimated_Bs[1]).max())
                print("        gammas = %s" %
                      abs(estimated_gammas[0] - estimated_gammas[1]).max())
                print("        omegas %s" %
                      abs(estimated_omegas[0] - estimated_omegas[1]).max())
                # abs(estimated_psis[0] - estimated_psis[1]).max())
                print("      Estimated psis %s" % estimated_psis)
                print()
            # Check that all scores are close to true model
            scores = np.array(scores)
            diffs = abs(scores - scores[0])
            print("    DAG scores: %s" % scores)
            print("       diff. wrt. to true model's score: %s" % diffs)
            self.assertTrue((diffs < 5e-2).all())


# ---------------------------------------------------------------------
# Test the solver for the connectivity matrix B


class BSolverTests(unittest.TestCase):
    """Test that the solver for the connectivity matrix B works correctly"""

    debug = False

    def test_basic(self):
        """Test that a call to the solver does not fail"""
        rng = np.random.default_rng(42)
        A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0, 0, 0.5], [0, 0, 3], [0, 0, 0]])
        gamma = rng.uniform(size=(2, 3))
        omegas = rng.uniform(size=(5, 3))
        psis = rng.uniform(size=(5, 2))
        model = Model(A, B, gamma, omegas, psis)
        X, sample_covariances, n_obs = model.sample(100)
        B = utlvce.score._solve_for_B_grad(
            model, sample_covariances, n_obs, self.debug)
        mask = np.logical_not(A)
        self.assertTrue(np.allclose(B[mask], 0))

    def test_no_confounding(self):
        """Test that when there is no confounding and the population
        parameters are given to the solver, the returned B is close to the
        population B"""
        rng = np.random.default_rng(42)
        A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0, 0, 0.5], [0, 0, 3], [0, 0, 0]])
        gamma = np.zeros((2, 3))  # No latent effects
        omegas = rng.uniform(size=(5, 3))
        psis = rng.uniform(size=(5, 2))
        model = Model(A, B, gamma, omegas, psis)
        X, sample_covariances, n_obs = model.sample(
            round(1e6), random_state=42)
        # Solve for B
        estimated_B = utlvce.score._solve_for_B_grad(
            model, sample_covariances, n_obs, self.debug)
        mask = np.logical_not(A)
        self.assertTrue(np.allclose(estimated_B[mask], 0))
        self.assertLess(abs(B - estimated_B).max(), 0.001)

    def test_population(self):
        """For random graphs, test that when the population parameters +
        covariances are given to the solver, the returned B is close
        to the population B.
        """
        G = 100  # NUM_GRAPHS
        p = 5
        num_latent = 3
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing connectivity solver with population covariances")
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)
            #_, sample_covariances, n_obs = model.sample(10000)

            estimated_B = utlvce.score._solve_for_B_cvx(
                model, sample_covariances, n_obs, self.debug)
            print(estimated_B)
            mask = np.logical_not(model.A)
            self.assertTrue(np.allclose(estimated_B[mask], 0))
            print("  diff. wrt. population B =",
                  abs(model.B - estimated_B).max())
            self.assertLess(abs(model.B - estimated_B).max(), 1e-6)

    def test_vs_convex(self):
        return True

        # ---------------------------------------------------------------------
        # Test the gradient descent procedures for gamma, psis and omegas


class GradientDescentTests(unittest.TestCase):
    """Tests for the gradient descent procedure"""

    def test_least_squares_1d(self):
        """Check main._gradient_descent on a simple least-squares regression problem"""
        # Set up problem
        rng = np.random.default_rng(42)
        n = 1000
        X = rng.normal(0, 1.2, n)
        true_b = 2.1
        true_c = 3
        Y = X * true_b + true_c

        # Loss. theta[0] = b, theta[1] = c
        def obj_fun(theta):
            return np.sum((Y - X * theta[0] - theta[1])**2) / n

        # Gradient
        def gradient_fun(theta):
            gradient_b = -2 * np.sum((Y - theta[0] * X - theta[1]) * X)
            gradient_c = -1 * np.sum((Y - theta[0] * X - theta[1]))
            return [gradient_b, gradient_c]

        # Test
        estimate, _ = utlvce.score._gradient_descent(
            obj_fun, gradient_fun, np.ones(2) * -1, 1e-6, 1, None, 0)

        # Check that estimates are close
        self.assertLess(abs(true_b - estimate[0]), 1e-2)
        self.assertLess(abs(true_c - estimate[1]), 1e-2)

        # Test (enforcing positive X)
        estimate, _ = utlvce.score._gradient_descent(
            obj_fun, gradient_fun, np.ones(2) * 1, 1e-6, 1, [(0, np.Inf)] * 2, 0)

        # Check that estimates are close
        self.assertLess(abs(true_b - estimate[0]), 1e-2)
        self.assertLess(abs(true_c - estimate[1]), 1e-2)

    def test_solve_for_gamma_population_1(self):
        """When using the population covariances as sample covariances and
        setting the starting point at the population gamma, we expect
        the following behaviour: either the population gamma is the
        minimizer of the score (kappa reached zero and the returned
        gamma is the same as the starting point), or the procedure
        found one with a better score. The hypothesis is that this is
        due to the numerical precision of the computer and thus the
        difference in the population gamma and the minimizer
        """
        G = NUM_GRAPHS
        p = 5
        num_latent = 3
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing gradient descent with population gamma as starting point")
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)

            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)

            # Solve for gamma with the population gamma as starting point
            estimated_gamma = utlvce.score._solve_for_gamma(
                model, sample_covariances, n_obs, 1e-10, 1, verbose=0)
            estimated_model = model.copy()
            estimated_model.gamma = estimated_gamma

            # Either the population gamma is the minimizer of the
            # score (kappa reached zero, returned gamma is the same as
            # starting point), or the procedure found one with a
            # better score.
            score_pop = model.score(sample_covariances, n_obs)
            score_estimate = estimated_model.score(sample_covariances, n_obs)
            if (model.gamma == estimated_model.gamma).all():
                self.assertEqual(score_pop, score_estimate)
            else:
                max_diff_gammas = abs(
                    model.gamma - estimated_model.gamma).max()
                print("  Dif. gamma wrt. pop: %s" % max_diff_gammas)
                # If the population gamma is not the maximizer, and
                # this is due to precision issues, make sure the
                # difference between the gammas is very small
                self.assertLess(max_diff_gammas, 1e-10)
                self.assertLess(score_estimate, score_pop)

    def test_solve_for_gamma_population_2(self):
        """When using the population covariances as sample covariances and
        setting the starting point at a minimally PERTURBED population
        gamma, we expect the following behaviour: either the
        population gamma is the minimizer or the procedure found one
        with a better score; in any case the gradient descent
        procedure should make at least one move. The hypothesis is
        that this is due to the numerical precision of the computer
        and thus the difference in the population gamma and the
        minimizer.

        """
        G = NUM_GRAPHS
        p = 5
        num_latent = 3
        e = 5
        rng = np.random.default_rng(42)
        perturbance = 1e-4
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing gradient descent with perturbed (%s) gamma as starting point" % perturbance)
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)

            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)

            # Solve for gamma with a perturbed population gamma as starting point
            model_init = model.copy()
            model_init.gamma = model.gamma + \
                rng.uniform(-perturbance, perturbance, size=(num_latent, p))

            estimated_gamma = utlvce.score._solve_for_gamma(
                model_init, sample_covariances, n_obs, 1e-6, 1, verbose=0)
            estimated_model = model.copy()
            estimated_model.gamma = estimated_gamma

            # Check that the gradient descent procedure made at least one move
            self.assertFalse((model_init.gamma == estimated_model.gamma).all())

            # Check that the difference in population and estimated score is very small
            score_pop = model.score(sample_covariances, n_obs)
            score_estimate = estimated_model.score(sample_covariances, n_obs)
            diff_scores = abs(score_pop - score_estimate)
            print("  Dif. scores wrt. pop: %s" % diff_scores)
            # self.assertLess(diff_scores, 1e-14)

            # Check that the distance between estimated and population gamma is very small
            max_diff_gammas = abs(model.gamma - estimated_model.gamma).max()
            print("  Dif. gamma wrt. pop: %s" % max_diff_gammas)
            print("  Dif. gamma wrt. init: %s" %
                  abs(model_init.gamma - estimated_model.gamma).max())
            # self.assertLess(max_diff_gammas, 1e-10)

    def test_solve_for_rest_population_1(self):
        """TODO: When using the population covariances as sample covariances and
        setting the starting point at the population gamma, we expect
        the following behaviour: either the population gamma is the
        minimizer of the score (kappa reached zero and the returned
        gamma is the same as the starting point), or the procedure
        found one with a better score. The hypothesis is that this is
        due to the numerical precision of the computer and thus the
        difference in the population gamma and the minimizer
        """
        G = NUM_GRAPHS
        p = 5
        num_latent = 3
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing gradient descent with population omegas/psis as starting point")
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)

            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)

            # Solve for omegas/psis with their population values as starting point
            psi_max = 2
            estimated_omegas, estimated_psis = utlvce.score._solve_for_rest(
                model, set(range(model.p)), sample_covariances, n_obs, psi_max, False, 1e-10, 1, verbose=0)
            estimated_model = model.copy()
            estimated_model.omegas = estimated_omegas
            estimated_model.psis = estimated_psis

            # Assert that the bounds on omegas and psis are respected
            self.assertTrue((estimated_omegas > 0).all())
            self.assertTrue((estimated_psis > 0).all())
            self.assertTrue((estimated_psis < psi_max).all())

            # Either the population omegas/psis are the minimizer
            # of the score (kappa reached zero, returned gamma is the
            # same as starting point), or the procedure found one with
            # a better score.
            score_pop = model.score(sample_covariances, n_obs)
            score_estimate = estimated_model.score(sample_covariances, n_obs)
            if (model.psis == estimated_model.psis).all() and (model.omegas == estimated_model.omegas).all():
                self.assertEqual(score_pop, score_estimate)
            else:
                max_diff_omegas = abs(
                    model.omegas - estimated_model.omegas).max()
                max_diff_psis = abs(model.psis - estimated_model.psis).max()
                print("  Dif. omegas: %s" % max_diff_omegas)
                print("  Dif. psis: %s" % max_diff_psis)
                # If the population gamma is not the maximizer, and
                # this is due to precision issues, make sure the
                # difference between the gammas is very small
                self.assertLess(max_diff_psis, 1e-10)
                self.assertLess(max_diff_omegas, 1e-10)
                self.assertLess(score_estimate, score_pop)

    def test_solve_for_rest_population_2(self):
        """TODO: When using the population covariances as sample covariances and
        setting the starting point at a minimally PERTURBED population
        gamma, we expect the following behaviour: either the
        population gamma is the minimizer or the procedure found one
        with a better score; in any case the gradient descent
        procedure should make at least one move. The hypothesis is
        that this is due to the numerical precision of the computer
        and thus the difference in the population gamma and the
        minimizer.

        """
        G = NUM_GRAPHS
        p = 5
        num_latent = 3
        e = 5
        rng = np.random.default_rng(42)
        perturbance = 1e-4
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing gradient descent with perturbed (%s) omegas/psis as starting point" % perturbance)
        for i in range(G):
            print("\nGraph %d" % i)
            # Build model
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)

            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)

            # Solve for gamma with a perturbed population gamma as starting point
            model_init = model.copy()
            model_init.omegas = model.omegas + \
                rng.uniform(-perturbance, perturbance, size=(e, p))
            model_init.psis = model.psis + \
                rng.uniform(-perturbance, perturbance, size=(e, num_latent))

            psi_max = 2
            estimated_omegas, estimated_psis = utlvce.score._solve_for_rest(
                model_init, set(range(model.p)), sample_covariances, n_obs, psi_max, False, 1e-6, 1, verbose=0)
            estimated_model = model.copy()
            estimated_model.omegas = estimated_omegas
            estimated_model.psis = estimated_psis

            # Assert that the bounds on omegas and psis are respected
            self.assertTrue((estimated_omegas > 0).all())
            self.assertTrue((estimated_psis > 0).all())
            self.assertTrue((estimated_psis < psi_max).all())

            # Check that the gradient descent procedure made at least one move
            self.assertFalse(
                (model_init.omegas == estimated_model.omegas).all())
            # self.assertFalse((model_init.psis == estimated_model.psis).all())

            # Check that the difference in population and estimated score is very small
            score_pop = model.score(sample_covariances, n_obs)
            score_estimate = estimated_model.score(sample_covariances, n_obs)
            diff_scores = abs(score_pop - score_estimate)
            print("  Dif. scores wrt. pop: %s" % diff_scores)
            # self.assertLess(diff_scores, 1e-5)

            print("  Dif. omegas wrt. init: %s" %
                  abs(model_init.omegas - estimated_model.omegas).max())
            print("  Dif. psis wrt. init: %s" %
                  abs(model_init.psis - estimated_model.psis).max())

            # Check that the distance between estimated and population gamma is very small
            max_diff_omegas = abs(model.omegas - estimated_omegas).max()
            max_diff_psis = abs(model.psis - estimated_psis).max()
            print("  Dif. omegas wrt. pop: %s" % max_diff_omegas)
            print("  Dif. psis wrt.pop: %s" % max_diff_psis)
            # self.assertLess(max_diff_omegas, 1e-10)
            # self.assertLess(max_diff_psis, 1e-10)

    def test_within_limits_1(self):
        k = 10
        rng = np.random.default_rng(42)
        lower_bounds = rng.uniform(-2, 1, size=k)
        upper_bounds = lower_bounds + rng.uniform(0, 3, size=k)

        # Construct arrays within bounds
        sizes = [10, (10,)] + list(zip(rng.integers(1, 3, k - 2),
                                       rng.integers(1, 4, k - 2)))
        arrays = []
        for (l, h, size) in zip(lower_bounds, upper_bounds, sizes):
            arrays.append(rng.uniform(l, h, size=size))

        # Test that arrays within bounds always return true
        self.assertTrue(utlvce.score._is_within_bounds(
            arrays, zip(lower_bounds, upper_bounds)))

        # Test that an error is raised if a lower bound > upper bound
        try:
            utlvce.score._is_within_bounds(
                arrays, zip(upper_bounds, lower_bounds))
        except ValueError as e:
            print("OK:", e)
        except Exception:
            self.fail()

        # Test that arrays must be strictly within bounds
        arrays[0][0] = lower_bounds[0]
        self.assertFalse(utlvce.score._is_within_bounds(
            arrays, zip(lower_bounds, upper_bounds)))
        arrays[0][0] = upper_bounds[0]
        self.assertFalse(utlvce.score._is_within_bounds(
            arrays, zip(lower_bounds, upper_bounds)))


# ---------------------------------------------------------------------
# Tests for the initialization of the parameters


class IntializationTests(unittest.TestCase):

    def test_B_init_no_latent(self):
        """Test that when using the population covariances and the latent
        effect matrix is all zeros, the initialized B is very close to
        the pop. B

        """
        G = NUM_GRAPHS
        p = 5
        num_latent = 3
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print(
            "Testing initialization of B with population covariances and no latent effects")
        for i in range(G):
            print("\nGraph %d" % i)
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            model.gamma = np.zeros_like(model.gamma)
            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)

            estimated_B = utlvce.score._initialize_B(
                model.A, sample_covariances, n_obs, init='obs')
            print("Diff. wrt. pop. B:", abs(model.B - estimated_B).max())
            self.assertLess(abs(model.B - estimated_B).max(), 1e-10)

    def test_B_init_latent(self):
        """Test that when using the population covariances, the initialized B
        is very close to the pop. B

        """
        G = NUM_GRAPHS
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 0.6, 0.8
        print("\n" + "-" * 70)
        print("Testing initialization of B with population covariances")
        for i in range(G):
            print("\nGraph %d" % i)
            model = sample_model(p, set(range(p)), num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            sample_covariances = model.covariances()
            n_obs = np.array([1] * model.e)

            estimated_B = utlvce.score._initialize_B(
                model.A, sample_covariances, n_obs, init='obs')
            print(model.B)
            print(estimated_B)
            print("Diff. wrt. pop. B:", abs(model.B - estimated_B).max())
            # self.assertLess(abs(model.B - estimated_B).max(), 1e-10)

# ---------------------------------------------------------------------
# Tests for the Cache


class CacheTests(unittest.TestCase):

    def test_basic(self):
        n = 100
        p = 10
        cache = _Cache()
        models = []

        # Test that writing works
        for i in range(n):
            A = np.random.uniform(size=(p, p))
            I = set(sempler.generators.intervention_targets(p, 1, p)[0])
            model = sample_model(p, set(range(p)), 5, 2,
                                 0.5, 0.6, 0.7, 0.7, random_state=i)
            score = np.random.uniform()
            models.append((A, I, model, score))
            self.assertIsNone(cache.read(A, I))
            self.assertIsNone(cache._find(A, I))
            self.assertIsNone(cache.write(A, I, model, score, check=False))

        # Test reading, and that writing the same model with
        # check=True raises a ValueError
        for (A, I, model, score) in models:
            self.assertEqual(score, cache.read(A, I)[1])
            try:
                cache.write(A, I, model, score, check=True)
                self.fail()
            except ValueError:  # as e:
                # print("OK:", e)
                pass

        # Test that if we write a duplicate model (check=False) then
        # _find and read raise an Exception
        cache.write(A, I, model, score, check=False)
        try:
            cache._find(A, I)
            self.fail()
        except Exception:  # as e:
            # print("OK:", e)
            pass

        try:
            cache.read(A, I)
            self.fail()
        except Exception:  # as e:
            # print("OK:", e)
            pass
