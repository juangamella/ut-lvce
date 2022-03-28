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

import unittest
import numpy as np
import ut_lvcm.experiments as experiments
import ut_lvcm.metrics as metrics
import ut_lvcm.utils as utils
import sempler.generators
import time

# ---------------------------------------------------------------------
# Tests for the scree selection procedure

NUM_RND_TESTS = 50


class MetricsTests(unittest.TestCase):

    def test_dist_struct_properties(self):
        p = 10
        no_edges = 3
        for i in range(NUM_RND_TESTS):
            A = sempler.generators.dag_avg_deg(p, 2.1, w_min=1, w_max=2, random_state=i)
            # The distance from a graph to itself is 0
            self.assertEqual(0, metrics._dist_struct(A, A))
            # The distance from a graph to the empty graph is 1
            self.assertEqual(1, metrics._dist_struct(A, np.zeros_like(A)))
            # The distance from the empty graph to any graph is 0
            self.assertEqual(0, metrics._dist_struct(np.zeros_like(A), A))
            # The distance from a graph to a supergraph is always 0
            supergraph = utils.add_edges(A, no_edges, random_state=i)
            self.assertEqual(0, metrics._dist_struct(A, supergraph))
            # The distance from a graph to a subgraph is the number of
            # edges removed over the total number of edges
            self.assertEqual(no_edges / np.sum(supergraph),
                             metrics._dist_struct(supergraph, A))

    def test_dist_struct(self):
        A = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        B = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        C = np.array([[0, 0, 0, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        D = np.array([[0, 0, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertEqual(1 / 3, metrics._dist_struct(A, B))
        self.assertEqual(1 / 3, metrics._dist_struct(A, D))
        self.assertEqual(.5, metrics._dist_struct(B, A))
        self.assertEqual(.75, metrics._dist_struct(B, C))
        self.assertEqual(2 / 3, metrics._dist_struct(C, B))
        self.assertEqual(0, metrics._dist_struct(C, D))
        self.assertEqual(.5, metrics._dist_struct(D, A))
        self.assertEqual(0.25, metrics._dist_struct(D, C))
        self.assertEqual(1 / 3, metrics._dist_struct(A, C))
        self.assertEqual(1 / 3, metrics._dist_struct(C, A))

    def test_maxmin_1(self):
        # Test that for singleton classes the result is the same as calling _distr_struct
        p = 10
        for i in range(NUM_RND_TESTS):
            A = sempler.generators.dag_avg_deg(p, 2.1, w_min=1, w_max=2, random_state=i)
            B = sempler.generators.dag_avg_deg(p, 2.1, w_min=1, w_max=2, random_state=i + 100)
            self.assertEqual(metrics._dist_struct(A, B),
                             metrics._maxmin_metric([A], [B], metrics._dist_struct))
            self.assertEqual(metrics._dist_struct(A, B),
                             metrics.type_1_structc([A], [B]))

    def test_maxmin_2(self):
        A = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        B = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        C = np.array([[0, 0, 0, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        D = np.array([[0, 0, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        # Test 1
        class_1 = [A, C]
        class_2 = [B, D]
        self.assertEqual(1 / 3, metrics._maxmin_metric(class_1, class_2, metrics._dist_struct))
        self.assertEqual(1 / 2, metrics._maxmin_metric(class_2, class_1, metrics._dist_struct))
        self.assertEqual(1 / 3, metrics.type_1_structc(class_1, class_2))
        self.assertEqual(1 / 2, metrics.type_2_structc(class_1, class_2))
        # Test 2
        class_1 = [A, B]
        class_2 = [C, D]
        self.assertEqual(1 / 2, metrics._maxmin_metric(class_1, class_2, metrics._dist_struct))
        self.assertEqual(1 / 2, metrics._maxmin_metric(class_2, class_1, metrics._dist_struct))
        self.assertEqual(1 / 2, metrics.type_1_structc(class_1, class_2))
        self.assertEqual(1 / 2, metrics.type_2_structc(class_1, class_2))

    def test_imec_chain_graph(self):
        # Chain graph
        p = 20
        B_lo, B_hi = 0.7, 0.7
        h = 2
        e = 5
        var_lo, var_hi = 0.5, 0.6
        i_var_lo, i_var_hi = 6, 12
        psi_lo, psi_hi = 2.8, 2.8
        true_I = set()
        for i in range(1, p):
            true_I.add(p - i)

            # Build model
            true_model = experiments.chain_graph(
                p, true_I, h, e, var_lo, var_hi, i_var_lo, i_var_hi, psi_lo, psi_hi, B_lo, B_hi)

            # Test that call to _imec finishes fast enough
            start = time.time()
            imec = metrics._imec((true_model.A, true_I))
            print("%d members in I-MEC, done in %0.2f seconds" % (len(imec), time.time() - start))

    def test_dist_sets(self):
        p = 20
        k = 5
        rng = np.random.default_rng(42)

        # Distance from a set to a superset is always 0
        for i in range(NUM_RND_TESTS):
            I = set(rng.choice(range(p), size=k))
            sup = I | set(rng.choice(range(p), size=k))
            assert I <= sup
            self.assertEqual(0, metrics._dist_sets(set(), I))
            self.assertEqual(0, metrics._dist_sets(set(), sup))
            self.assertEqual(0, metrics._dist_sets(I, sup))

        # Distance from a set to a subset is the number of elements
        # removed over total number of elements
        for i in range(NUM_RND_TESTS):
            I = set(rng.choice(range(p), size=2 * k))
            rem = set(rng.choice(list(I), size=k))
            sub = I - rem
            self.assertEqual(len(rem) / len(I), metrics._dist_sets(I, sub))
            self.assertEqual(0, metrics._dist_sets(sub, I))

        # Some examples by hand
        A = {0, 1, 2}
        B = {2, 3, 4, 5}
        self.assertEqual(2 / 3, metrics._dist_sets(A, B))
        self.assertEqual(3 / 4, metrics._dist_sets(B, A))

    def test_type_x_I(self):
        p = 20
        k = 5
        rng = np.random.default_rng(42)

        for i in range(NUM_RND_TESTS):
            # Disjoint sets
            A = set(rng.choice(range(p), size=k))
            B = set(range(p)) - A
            # The scores of a set with itself should be 0
            self.assertEqual(0, metrics.type_1_I((None, A), (None, A)))
            self.assertEqual(0, metrics.type_2_I((None, A), (None, A)))
            self.assertEqual(0, metrics.type_1_I((None, B), (None, B)))
            self.assertEqual(0, metrics.type_2_I((None, B), (None, B)))
            # The scores of a set with a disjoint set should be 1
            self.assertEqual(1, metrics.type_1_I((None, A), (None, B)))
            self.assertEqual(1, metrics.type_2_I((None, A), (None, B)))
            self.assertEqual(1, metrics.type_1_I((None, B), (None, A)))
            self.assertEqual(1, metrics.type_2_I((None, B), (None, A)))

        # Some examples by hand
        A = {0, 1, 2}
        B = {2, 3, 4, 5}
        # type 1 : what ratio of the elements in the estimate (A) is wrong
        # type 2 : of the true elements, which ratio did A miss?
        self.assertEqual(2 / 3, metrics.type_1_I((None, A), (None, B)))
        self.assertEqual(3 / 4, metrics.type_2_I((None, A), (None, B)))

    def test_dist_sets_2(self):
        self.assertEqual(1, metrics._dist_sets({1, 2}, set()))
        self.assertEqual(1, metrics._dist_sets({3, 4}, set()))
        self.assertEqual(0, metrics._dist_sets({1, 2}, {1, 2, 3}))
        self.assertEqual(1 / 2, metrics._dist_sets({3, 4}, {1, 2, 3}))

        self.assertEqual(0, metrics._dist_sets(set(), {1, 2}))
        self.assertEqual(0, metrics._dist_sets(set(), {3, 4}))
        self.assertEqual(1 / 3, metrics._dist_sets({1, 2, 3}, {1, 2}))
        self.assertEqual(2 / 3, metrics._dist_sets({1, 2, 3}, {3, 4}))

        self.assertEqual(1, metrics._dist_sets({2}, set()))
        self.assertEqual(0, metrics._dist_sets({2}, {1, 2}))
        self.assertEqual(1, metrics._dist_sets({2}, {3, 4}))
        self.assertEqual(0, metrics._dist_sets({2}, {1, 2, 3}))

        self.assertEqual(0, metrics._dist_sets(set(), {2}))
        self.assertEqual(1 / 2, metrics._dist_sets({1, 2}, {2}))
        self.assertEqual(1, metrics._dist_sets({3, 4}, {2}))
        self.assertEqual(2 / 3, metrics._dist_sets({1, 2, 3}, {2}))

    def test_type_x_parents(self):
        # Test properties for random parent sets
        rng = np.random.default_rng(42)
        p = 10
        k = 10
        for i in range(NUM_RND_TESTS):
            size = 3
            parent_sets = [set(rng.choice(range(p), size=size, replace=False)) for i in range(k)]
            supersets = [s | {p} for s in parent_sets]
            # If two sets are the same the type-I and type-II errors are 0
            self.assertEqual(0, metrics.type_1_parents(parent_sets, parent_sets))
            self.assertEqual(0, metrics.type_2_parents(parent_sets, parent_sets))
            # if all are supersets
            # Test type 1 metric is 0
            self.assertEqual(0, metrics.type_1_parents(parent_sets, supersets))
            # Type II metric is 1 / (size + 1)
            self.assertEqual(1 / (size + 1), metrics.type_2_parents(parent_sets, supersets))
            # if all are subsets
            # Type 1 metric is 1 / (size + 1)
            self.assertEqual(1 / (size + 1), metrics.type_1_parents(supersets, parent_sets))
            # Type II metric is 0 if all are subsets
            self.assertEqual(0, metrics.type_2_parents(supersets, parent_sets))

        # Example by hand
        p1 = [{1, 2}, {3, 4}]
        p2 = [set(), {1, 2, 3}]
        self.assertEqual(1 / 2, metrics.type_1_parents(p1, p2))
        self.assertEqual(1 / 3, metrics.type_2_parents(p1, p2))
        self.assertEqual(1 / 3, metrics.type_1_parents(p2, p1))
        self.assertEqual(1 / 2, metrics.type_2_parents(p2, p1))

        # Example by hand
        p1 = [{1, 2}, {3, 4}]
        p2 = [{2}, {1, 2, 3}]
        self.assertEqual(1 / 2, metrics.type_1_parents(p1, p2))
        self.assertEqual(1 / 3, metrics.type_2_parents(p1, p2))
        self.assertEqual(1 / 3, metrics.type_1_parents(p2, p1))
        self.assertEqual(1 / 2, metrics.type_2_parents(p2, p1))

    def test_scores(self):
        """Test the basic properties of the structural metrics, i.e. that
             - recovering the truth yields type-I and type-II = 0
             - recovering a subset yields type-II > 0 but type-I = 0
             - recovering a superset yields type-II = 0 but type-I > 0
        """
        p = 20
        size_I = 3
        for i in range(NUM_RND_TESTS):
            targets = sempler.generators.intervention_targets(p, 1, size_I, random_state=i)[0]
            I = set(targets[0:-1])
            sup_I = set(targets)
            true_A = sempler.generators.dag_avg_deg(p=20, k=2.1, random_state=i)
            truth = (true_A, I)

            # Check that recovering the truth results in (0,0)
            self.assertEqual(0, metrics.type_1_struct(truth, truth))
            self.assertEqual(0, metrics.type_2_struct(truth, truth))

            # Check that recovering a subset of the truth results in type-II > 0 but type-I = 0
            estimate = (true_A, set(sup_I))
            if len(utils.imec(true_A, sup_I)) < len(utils.imec(true_A, I)):
                self.assertGreater(metrics.type_2_struct(estimate, truth), 0)
            else:
                self.assertEqual(0, metrics.type_2_struct(estimate, truth))
            self.assertEqual(0, metrics.type_1_struct(estimate, truth))

            # Check that recovering a superset of the truth results in type-II = 0 but type-I > 0
            truth = (true_A, sup_I)
            estimate = (true_A, I)
            if len(utils.imec(true_A, I)) > len(utils.imec(true_A, sup_I)):
                self.assertGreater(metrics.type_1_struct(estimate, truth), 0)
            else:
                self.assertEqual(0, metrics.type_1_struct(estimate, truth))
            self.assertEqual(0, metrics.type_2_struct(estimate, truth))

    def test_scores_2(self):
        """White box test: check that I-MECs are being computed correctly"""
        for i in range(NUM_RND_TESTS):
            for k in range(5):
                A1, I1 = A_I(i, size_I=k)
                A2, I2 = A_I(i + 1, size_I=k)
                imec1 = utils.imec(A1, I1)
                imec2 = utils.imec(A2, I2)
                self.assertEqual(metrics.type_1_struct((A1, I1), (A2, I2)),
                                 metrics.type_1_structc(imec1, imec2))
                self.assertEqual(metrics.type_2_struct((A1, I1), (A2, I2)),
                                 metrics.type_2_structc(imec1, imec2))
                self.assertEqual(metrics.type_1_struct((A2, I2), (A1, I1)),
                                 metrics.type_1_structc(imec2, imec1))
                self.assertEqual(metrics.type_2_struct((A2, I2), (A1, I1)),
                                 metrics.type_2_structc(imec2, imec1))

    def test_scores_armeen(self):
        """
        """
        for i in range(NUM_RND_TESTS):
            # Test 1: estimate and truth are the same singleton,
            # then type-I = type-II = 0
            (A, I) = A_I(i)
            self.assertEqual(0, metrics.type_1_structc([A], [A]))
            self.assertEqual(0, metrics.type_2_structc([A], [A]))

            # Test 2: Estimate is true DAG + extra edges
            # then type-I > 0, type-II = 0
            A1 = utils.add_edges(A, 5, random_state=i)
            self.assertGreater(metrics.type_1_structc([A1], [A]), 0)
            self.assertEqual(0, metrics.type_2_structc([A1], [A]))

            # Test 3: Reverse, i.e. estimate is true DAG - some edges
            # then type-I = 0, type-II > 0
            self.assertEqual(0, metrics.type_1_structc([A], [A1]))
            self.assertGreater(metrics.type_2_structc([A], [A1]), 0)

            # Test 4: Truth and estimate are same MEC,
            # then type-I = type-II = 0
            mec = utils.mec(A)
            self.assertEqual(0, metrics.type_1_structc(mec, mec))
            self.assertEqual(0, metrics.type_2_structc(mec, mec))

            # Test 5: Estimate is true MEC + extra edges on each graph
            # then type-I > 0, type-II = 0
            mec_extra = [utils.add_edges(D, 5, random_state=i) for D in mec]
            self.assertGreater(metrics.type_1_structc(mec_extra, mec), 0)
            self.assertEqual(0, metrics.type_2_structc(mec_extra, mec))

            # Test 6: Reversed, i.e. estimate is true MEC - some edges
            # on each graph, then type-I > 0, type-II = 0
            self.assertGreater(metrics.type_2_structc(mec, mec_extra), 0)
            self.assertEqual(0, metrics.type_1_structc(mec, mec_extra))

            # Test 7: If estimate is MEC(true DAG + extra edges) and
            # truth is MEC(true DAG), then
            #   - type-I >= number of added edges / total number of
            #     edges in supergraph
            #   - if every graph in the estimate has a subgraph in the
            #     truth, then type-I equals the bound
            #   - every graph in the truth has a supergraph in the estimate => type-II = 0
            #   - extra edges introduced v-structures => type-II >= 0
            no_edges = 20
            supergraph = utils.add_edges(A, no_edges, random_state=i)
            estimate = utils.mec(supergraph)
            print("\nCase %d : len(estimate) = %d vs. len(truth) = %d" %
                  (i, len(estimate), len(mec)))
            # Type-I tests
            type1 = metrics.type_1_structc(estimate, mec)
            bound = no_edges / np.sum(supergraph)
            # Check that the type-1 error respects lower bound
            self.assertGreaterEqual(type1, bound)
            # Check that if every graph in the estimate has a subgraph in the
            #     truth, then type-I equals the bound
            if utils.has_subgraph(estimate, mec):
                print(
                    "  every estimate graph has a subraph in the truth, i.e. type-I (%0.2f) == bound (%0.2f)" % (type1, bound))
                self.assertEqual(type1, bound)
            # And the reverse implication, i.e.
            if type1 > no_edges / np.sum(supergraph):
                print(
                    "  type-I = %0.2f > %0.2f = bound, i.e. there is at least one graph in the estimate without a subgraph in the truth." % (type1, bound))
                # NOTE: This can happen e.g. if true DAG is X -> Y <- Z and supergraph is X -> Y <- Z + (X -> Z)
                self.assertFalse(utils.has_subgraph(estimate, mec))
            # Type-II tests
            type2 = metrics.type_2_structc(estimate, mec)
            # Check that if every graph in the truth has a supergraph
            # in the estimate, then the type-II error is zero
            if utils.has_supergraph(mec, estimate):
                print("  type-II (%0.2f) should be zero" % type2)
                self.assertEqual(type2, 0)
            # Same but negated implication
            if type2 > 0:
                print(
                    "  type-II = %0.2f > 0, i.e. at least one graph in the truth does not have any supergraphs in estimate." % type2)
                self.assertFalse(utils.has_supergraph(mec, estimate))
            # Check that if the extra edges added v-structures which
            # oriented undirected edges in the true CPDAG, then
            # type-II >= 0
            true_cpdag = utils.dag_to_cpdag(A)
            estimated_cpdag = utils.dag_to_cpdag(supergraph)
            true_undirected = set(utils.undirected_edges(true_cpdag))
            estimate_undirected = set(utils.undirected_edges(estimated_cpdag))
            how_many = len(true_undirected - estimate_undirected)
            if how_many > 0:
                print(
                    "  new edges oriented previously undirected ones (%d), i.e. type-II (%0.2f) should be > 0" % (how_many, type2))
                self.assertGreater(type2, 0)

    def test_skeleton(self):
        p = 20
        no_edges = 7
        for i in range(NUM_RND_TESTS):
            print(".", end="")
            true_A = sempler.generators.dag_avg_deg(p=p, k=2.1, random_state=i)
            # Test that graph with itself has 0,0
            self.assertEqual(metrics.type_1_skeleton(true_A, true_A), 0)
            self.assertEqual(metrics.type_2_skeleton(true_A, true_A), 0)
            # Test that the graph with reversed edges has 0,0
            self.assertEqual(metrics.type_1_skeleton(true_A, true_A.T), 0)
            self.assertEqual(metrics.type_2_skeleton(true_A, true_A.T), 0)
            self.assertEqual(metrics.type_1_skeleton(true_A.T, true_A), 0)
            self.assertEqual(metrics.type_2_skeleton(true_A.T, true_A), 0)
            # Test supergraph vs. true A
            supergraph = utils.add_edges(true_A, no_edges)
            self.assertEqual(metrics.type_1_skeleton(supergraph, true_A),
                             no_edges / np.sum(supergraph))
            self.assertEqual(metrics.type_2_skeleton(supergraph, true_A), 0)
            self.assertEqual(metrics.type_1_skeleton(supergraph, true_A.T),
                             no_edges / np.sum(supergraph))
            self.assertEqual(metrics.type_2_skeleton(supergraph, true_A.T), 0)
            # Test true_A vs. supergraph
            supergraph = utils.add_edges(true_A, no_edges)
            self.assertEqual(metrics.type_1_skeleton(true_A, supergraph), 0)
            self.assertEqual(metrics.type_2_skeleton(true_A, supergraph),
                             no_edges / np.sum(supergraph))
            self.assertEqual(metrics.type_1_skeleton(true_A.T, supergraph), 0)
            self.assertEqual(metrics.type_2_skeleton(
                true_A.T, supergraph), no_edges / np.sum(supergraph))
        print()


def A_I(seed, p=20, size_I=3):
    I = set(sempler.generators.intervention_targets(p, 1, size_I, random_state=seed)[0])
    A = sempler.generators.dag_avg_deg(p=20, k=2.1, random_state=seed)
    return (A, I)
