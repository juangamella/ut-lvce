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

# Tested functions
from utlvce import Model


class TestModel(unittest.TestCase):
    rng = np.random.default_rng(42)
    A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    B = np.array([[0, 0, 0.5], [0, 0, 3], [0, 0, 0]])
    gamma = rng.uniform(size=(2, 3))
    omegas = rng.uniform(size=(5, 3))
    psis = rng.uniform(size=(5, 2))
    model = Model(A, B, gamma, omegas, psis)

    def test_memory(self):
        """Check that on creation objects are copied and not passed by reference"""
        rng = np.random.default_rng(42)
        A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0, 0, 0.5], [0, 0, 3], [0, 0, 0]])
        gamma = rng.uniform(size=(2, 3))
        omegas = rng.uniform(size=(5, 3))
        psis = rng.uniform(size=(5, 2))
        model = Model(A, B, gamma, omegas, psis)

        # Check that stored parameters match
        self.assertTrue((A == model.A).all())
        self.assertTrue((B == model.B).all())
        self.assertTrue((gamma == model.gamma).all())
        self.assertTrue((omegas == model.omegas).all())
        self.assertTrue((psis == model.psis).all())

        # Check that they were copied
        A[0, 0] = 1
        B[0, 0] = 1
        gamma[0, 0] = -1
        omegas[0, 0] = -1
        psis[0] = -1
        self.assertFalse((A == model.A).all())
        self.assertFalse((B == model.B).all())
        self.assertFalse((gamma == model.gamma).all())
        self.assertFalse((omegas == model.omegas).all())
        self.assertFalse((psis == model.psis).all())

    def test_copy(self):
        """Test Model.copy()"""
        rng = np.random.default_rng(42)
        A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0, 0, 0.5], [0, 0, 3], [0, 0, 0]])
        gamma = rng.uniform(size=(2, 3))
        omegas = rng.uniform(size=(5, 3))
        psis = rng.uniform(size=(5, 2))
        model = Model(A, B, gamma, omegas, psis)
        copy = model.copy()

        # Check that parameters match
        self.assertTrue((copy.A == model.A).all())
        self.assertTrue((copy.B == model.B).all())
        self.assertTrue((copy.gamma == model.gamma).all())
        self.assertTrue((copy.omegas == model.omegas).all())
        self.assertTrue((copy.psis == model.psis).all())

        # Check that changes do not propagate
        model.A[0, 0] = 1
        model.B[0, 0] = 1
        model.gamma[0, 0] = -1
        model.omegas[0, 0] = -1
        model.psis[0] = -1

        self.assertFalse((copy.A == model.A).all())
        self.assertFalse((copy.B == model.B).all())
        self.assertFalse((copy.gamma == model.gamma).all())
        self.assertFalse((copy.omegas == model.omegas).all())
        self.assertFalse((copy.psis == model.psis).all())

    def test_covariances_1(self):
        """Test model.covariances: here, that the dimensions are correct"""
        covariances = self.model.covariances()
        self.assertEqual((self.model.e, self.model.p,
                         self.model.p), covariances.shape)

    def test_covariances_2(self):
        """Test model.covariances: here, that for an latent matrix of zeros,
        the population covariances are the same as for a purely observational
        model."""
        rng = np.random.default_rng(42)
        A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0, 0, 0.5], [0, 0, 3], [0, 0, 0]])
        gamma = np.zeros((2, 3))
        omegas = rng.uniform(size=(5, 3))
        psis = rng.uniform(size=(5, 2))
        model = Model(A, B, gamma, omegas, psis)

        # Construct pop. covariances by hand
        pop_covariances = []
        I_B_inv = np.linalg.inv(np.eye(3) - B.T)
        for i, omegas_e in enumerate(omegas):
            cov = I_B_inv @ np.diag(omegas_e) @ I_B_inv.T
            pop_covariances.append(cov)
        pop_covariances = np.array(pop_covariances)
        # print(pop_covariances)
        # print(model.covariances())
        self.assertTrue(np.allclose(pop_covariances, model.covariances()))

    def test_covariances_3(self):
        """Test model.covariances: here, that for an identity latent effect
        matrix, the population covariances are the same as for a
        purely observational model where the observed variables
        receive the appropriate interventions.

        """
        rng = np.random.default_rng(42)
        A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0, 0, 0.5], [0, 0, 3], [0, 0, 0]])
        gamma = np.eye(3)
        omegas = rng.uniform(size=(5, 3))
        psis = rng.uniform(size=(5, 3))
        model = Model(A, B, gamma, omegas, psis)

        # Construct pop. covariances by hand
        pop_covariances = []
        p = 3
        I_B_inv = np.linalg.inv(np.eye(p) - B.T)
        for i, omegas_e in enumerate(omegas):
            cov = I_B_inv @ (np.diag(omegas_e) + np.diag(psis[i])) @ I_B_inv.T
            pop_covariances.append(cov)
        pop_covariances = np.array(pop_covariances)
        # print(pop_covariances)
        # print(model.covariances())
        self.assertTrue(np.allclose(pop_covariances, model.covariances()))

    def test_sample_1(self):
        """Test model.sample: here, basic tests about the dimensions of the output"""

        # Sample, without computing sample covariances (constant sample size)
        n = 123
        X = self.model.sample(n, False)
        self.assertEqual(self.model.e, len(X))
        for x in X:
            self.assertEqual((n, self.model.p), x.shape)

        # Sample, without computing sample covariances
        n_obs = [12, 34, 541, 123, 56]
        X = self.model.sample(n_obs, False)
        self.assertEqual(self.model.e, len(X))
        for i, x in enumerate(X):
            self.assertEqual((n_obs[i], self.model.p), x.shape)

        # Sample (constant sample size)
        n = 123
        X, sample_covariances, n_obs = self.model.sample(n, True)
        self.assertEqual((self.model.e, self.model.p,
                         self.model.p), sample_covariances.shape)
        self.assertEqual([n] * self.model.e, n_obs)
        self.assertEqual(self.model.e, len(X))
        for x in X:
            self.assertEqual((n, self.model.p), x.shape)

        # Sample
        n = [12, 34, 541, 123, 56]
        X, sample_covariances, n_obs = self.model.sample(n, True)
        self.assertEqual((self.model.e, self.model.p,
                         self.model.p), sample_covariances.shape)
        self.assertEqual(n, n_obs)
        self.assertEqual(self.model.e, len(X))
        for i, x in enumerate(X):
            self.assertEqual((n[i], self.model.p), x.shape)

    def test_sample_2(self):
        """Test model.sample: here, test that for a very large sample size the
        population and sample covariances are close"""
        n = int(1e7)
        sample_covariances = self.model.sample(n, True, random_state=1)[1]
        pop_covariances = self.model.covariances()
        # print(pop_covariances)
        # print(sample_covariances)
        print(abs(pop_covariances - sample_covariances).max())
        self.assertLess(abs(pop_covariances - sample_covariances).max(), 0.01)

# --------------------------------------------------------------------
#
