"""Semantic Path-HMLC: frozen label semantics and taxonomy graph fusion for
single-path hierarchical text classification.

This package is a reorganized, general-purpose extraction of the code used to
produce the results reported in the paper. The modeling logic, hyperparameters,
losses, and evaluation protocol are unchanged from the original research
notebooks; only path handling and execution entry points were generalized so
the pipeline can run outside Google Colab / Google Drive.

See the repository README for how the modules map onto the paper's sections
and algorithms.
"""

__version__ = "1.0.0"
