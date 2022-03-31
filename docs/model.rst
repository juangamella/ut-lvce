utlvce.Model
============

The :class:`utlvce.Model` class is a representation of a linear Gaussian causal model with latent effects. It is used throughout the code, e.g. in the alternating optimization procedure implemented in :class:`utlvce.score` and to generate synthetic data.

The class also implements the `__str__` method; calling `print(model)` will return a human-readable representation of the model parameters and the assumption deviation metrics.

.. autoclass:: utlvce.Model
   :special-members: __init__
   :members:
