# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import PackageNotFoundError, version as _version

# -- Project information -----------------------------------------------------
project = "ddDT-GFM"
copyright = "2024, CWRU SDLE"
author = (
    "Benjamin G. Pierce, Raymond J. Wieser, Weiqi Yue, Yangxin Fan, "
    "Hein Htet Aung, Thomas G. Ciardi, Yinghui Wu, Pawan K. Tripathi, "
    "Roger H. French"
)

try:
    release = _version("ddDT-GFM")
except PackageNotFoundError:
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google / NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Heavy / hard-to-install dependencies are mocked so the docs build (e.g. on
# Read the Docs) without needing the full PyTorch Geometric stack installed.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torch_geometric",
    "torch_sparse",
    "tsl",
    "pytorch_lightning",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "seaborn",
    "matplotlib",
    "tqdm",
    "requests",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
