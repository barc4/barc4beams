import os
import sys
from datetime import datetime


sys.path.insert(0, os.path.abspath("../src"))

project = "barc4beams"
author = "Rafael Celestre"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

from barc4beams import __version__ 

release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]