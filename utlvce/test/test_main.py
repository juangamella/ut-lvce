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
import ut_lvcm.main

# ---------------------------------------------------------------------
# Tests for the scree selection procedure


class ScreeSelectionTests(unittest.TestCase):

    def test_chain_graph(self):
        # See (scree_selection_debugging notebook)
        # Chain graph
        p = 50
        B_lo, B_hi = 0.6, 0.7
        h = 3
        e = 5
        var_lo, var_hi = 0.5, 0.6
        i_var_lo, i_var_hi = 6, 12
        psi_lo, psi_hi = 8, 8
        I = {0, 1}
        n = 1000
        model = experiments.chain_graph(p, I, h, e, var_lo, var_hi,
                                        i_var_lo, i_var_hi, psi_lo, psi_hi, B_lo, B_hi)
        XX, sc, _ = model.sample(n, random_state=42)
        covariance = np.cov(XX[0], rowvar=False)

        selected = ut_lvcm.main._scree_selection(covariance)

        self.assertEqual(3, selected)
