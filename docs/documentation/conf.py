# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyscheduling_cc'
copyright = '2022, scheduling-cc'
author = 'scheduling-cc'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["nbsphinx",
    "autoapi.extension",
    "sphinx_panels",
    "sphinx_copybutton",
    "sphinx.ext.napoleon"]

autoapi_dirs = ["../../src"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/pyscheduling_navbar.png"

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    "css/pyscheduling.css",
]