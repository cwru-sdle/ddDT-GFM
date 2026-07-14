Installation
============

Install the core package from PyPI:

.. code-block:: bash

   pip install ddDT-GFM

Full install (PyTorch Geometric stack)
--------------------------------------

The models and dataloaders depend on the PyTorch Geometric ecosystem
(``torch_geometric``, ``torch_sparse``, and ``torch-spatiotemporal``/``tsl``).
These packages frequently require wheels matched to your specific ``torch`` and
CUDA build, so they are provided as an optional ``full`` extra:

.. code-block:: bash

   pip install "ddDT-GFM[full]"

If that fails, install ``torch`` first, then follow the official PyTorch
Geometric installation guide for wheels matched to your platform:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Quick start
-----------

Download a photovoltaic site from the OSF database:

.. code-block:: python

   from dddt_gfm.util.io import OSF_download

   OSF_download(0, "downloaded_test.csv", "./")
