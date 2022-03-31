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
import utlvce
import utlvce.generators
import time

NUM_GRAPHS = 10


# Tested functions

def sample_model(p, I, num_latent, e, var_lo, var_hi, B_lo, B_hi, random_state=42, verbose=0):
    rng = np.random.default_rng(random_state)
    B = utlvce.generators._dag_avg_deg(
        p, 2.5, B_lo, B_hi, random_state=random_state)
    A = (B != 0).astype(int)
    gamma = rng.normal(0, 1 / np.sqrt(p), size=(num_latent, p))
    omegas = rng.uniform(var_hi * 1.5, var_hi * 3, size=(e, p))
    for j in range(p):
        if j not in I:
            omegas[:, j] = omegas[:, j].mean()
    psis = rng.uniform(var_lo, var_hi, size=(e, num_latent))
    model = utlvce.Model(A, B, gamma, omegas, psis)
    return model

# ---------------------------------------------------------------------
# Tests for the GES wrapper


class TestGES(unittest.TestCase):

    def test_ges(self):
        G = 5
        p = 10
        num_latent = 2
        e = 5
        var_lo, var_hi = 1, 2
        B_lo, B_hi = 2, 3
        size_I = (0, 3)
        n = 1000
        print("\n" + "-" * 70)
        print("Testing GES")
        for i in range(G):
            # Sample intervention targets
            true_I = utlvce.generators.intervention_targets(p, size_I)
            print("\nGraph %d - I = %s" % (i, true_I))
            # Build model
            model = sample_model(p, true_I, num_latent, e, var_lo,
                                 var_hi, B_lo, B_hi, random_state=i)
            X = model.sample(n)
            # Run GES
            print("Running GES...", end="")
            start = time.time()
            utlvce.main._fit_ges(X)
            print("done (%0.2f seconds)" % (time.time() - start))
