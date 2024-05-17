"""Sphinx configuration."""

project: str = "python-iitp"
author: str = "Anton"
copyright: str = f"2024, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
