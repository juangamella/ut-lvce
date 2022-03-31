.. utlvce documentation master file, created by
   sphinx-quickstart on Wed Mar 30 19:29:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to utlvce's documentation!
==================================

The ``utlvce`` package is a Python implementation of the UT-LVCE algorithms from the 2022 `paper <https://arxiv.org/abs/2101.06950>`_ "Perturbations and Causality in Gaussian Latent Variable Models", by A. Taeb, JL. Gamella, C. Heinze-Deml and P. Bühlmann.

You can find the source code in the package's `GitHub repository <https://github.com/juangamella/ut-lvce>`_. The code to reproduce the experiments and figures in the paper can be found in a `separate repository <https://github.com/juangamella/ut-lvce-paper>`_.

**Installation**

You can install the latest version of the package using pip, i.e.

.. code-block:: bash
   
   pip install utlvce

How to run the UT-LVCE algorithms
---------------------------------

If you're interested in running the algorithms proposed in the paper for your own work, you can find details on how to do so and examples under the section :ref:`Running the UT-LVCE algorithms`.

We also document here the other functionalities of the package, such as generating synthetic data and random models for your own experiments (see :mod:`utlvce.generators` and :class:`utlvce.model`). If you're interested in the alternating optimization procedure from section 4.1, you can find more details under :mod:`utlvce.score`.

Versioning
----------

The pacakge is still at its infancy and its API may change in the future. Non backward-compatible changes to the API are reflected by a change to the minor or major version number, e.g.

    *code written using utlvce==0.1.2 will run with utlvce==0.1.3, but may not run with utlvce==0.2.0.*


License
-------
The code is open-source and shared under a BSD 3-Clause License. You can find the full license and the source code in the `GitHub repository <https://github.com/juangamella/ut-lvce>`_.

Feedback
--------

Feedback is most welcome! You can add an issue in the `repository <https://github.com/juangamella/ut-lvce>`_ or send an `email <mailto:juan.gamella@stat.math.ethz.ch>`_.



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   self
   algorithms
   model
   score
   generators
