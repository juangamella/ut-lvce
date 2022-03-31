Running the UT-LVCE algorithms
==============================

The `paper <https://arxiv.org/abs/2101.06950>`_ proposes two algorithms for estimating the interventional equivalence class, and a full causal discovery procedure:

- Algorithm 1: Estimating the equivalence class of best scoring DAGs from a set of candidate DAGs.
- Algorithm 2: Improving on the initial candidate set of DAGs and estimating the equivalence class of best scoring on.
- UT-LVCE as a Causal Discovery procedure, using the output of GES on the pooled data as an initial candidate set of DAGs.

These are provided through the functions :func:`utlvce.equivalence_class` and :func:`utlvce.equivalence_class_w_ges`, documented below.

----

We now provide details and examples on how to run the algorithms; the example code can also be found `here <https://github.com/juangamella/ut-lvce/blob/master/docs/algorithms_example.py>`_. To this end we will first generate some synthetic data using the :mod:`utlvce.generators` module:

.. literalinclude:: algorithms_example.py
   :lines: 1-12
                    


**Algorithm 1**

Algorithm 1 can be run by setting `prune_edges=False` when calling :func:`utlvce.equivalence_class` with the data and a candidate set of DAGs.

.. literalinclude:: algorithms_example.py
   :lines: 14-16


**Algorithm 2**

Conversely, algorithm 2 can be run by setting `prune_edges=True` when calling :func:`utlvce.equivalence_class`.

.. literalinclude:: algorithms_example.py
   :lines: 18-20


**UT-LVCE as a Causal Discovery Procedure**

For the causal discovery procedure, the :func:`utlvce.equivalence_class_w_ges` function is provided. This will estimate the equivalence class of the data-generating model, using the output of GES on the pooled data as an initial set of candidate DAGs.

Note that the call would be equivalent to passing the output DAGs of GES to :func:`utlvce.equivalence_class` with `prune_edges=True`. Any other causal discovery procedure can be used to generate a set of initial candidate DAGs and used together with UT-LVCE in this way.

.. literalinclude:: algorithms_example.py
   :lines: 22-24


Full function specification
---------------------------

Both functions can take additional parameters to control, among other things, the behaviour of the alternating optimization procedure used to compute the likelihood of the DAGs. The complete specification follows.

.. autofunction:: utlvce.equivalence_class


.. autofunction:: utlvce.equivalence_class_w_ges
