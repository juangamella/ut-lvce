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
Wrapper module for the calls to GES.
"""

import numpy as np
import ges
import ges.scores
import time
import ut_lvcm.utils as utils
# import pandas as pd
# import networkx as nx
# from cdt.causality.graph import GES

# ---------------------------------------------------------------------
# Module API


def fit(data, verbose=1, lambdas=None, phases=None):
    """Pool the data and run GES on it, returning the estimated CPDAG.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list containing the sample from each environment.
    verbose : int, default=0
        If debug and execution traces should be printed. `0`
        corresponds to no traces, higher values correspond to higher
        verbosity.

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
        print("  done (%0.2f seconds)" % (time.time() - start)) if verbose > 0 else None
    return np.array(graphs)

    # if lib == 'cdt':
    #     output = GES(verbose=verbose).predict(pd.DataFrame(pooled_data))
    #     result = nx.to_numpy_array(output)
    # elif lib == 'ges':
    #     result = ges.fit_bic(pooled_data)[0]
    # print("done (%0.2f seconds)" % (time.time() - start)) if verbose > 0 else None
    # return result
