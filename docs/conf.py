"""Sphinx configuration for the dnnlpy API reference."""

import tomllib
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

project = 'dnnlpy'
author = 'Yunjie Lin'

with open(PACKAGE_ROOT / 'pyproject.toml', 'rb') as project_file:
    release = tomllib.load(project_file)['project']['version']

extensions = [
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
exclude_patterns = [
    '_build',
    '.DS_Store',
    '.venv',
]

# Sphinx
add_module_names = False
python_use_unqualified_type_names = True

# AutoAPI
autoapi_dirs = ['../src']
autoapi_root = 'reference'
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_member_order = 'bysource'
autoapi_python_class_content = 'class'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]

# Docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# HTML
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']


def skip_attributes(
    app: Sphinx, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool:
    if what == 'attribute':
        return True
    return skip


def setup(app: Sphinx) -> None:
    app.connect('autoapi-skip-member', skip_attributes)
