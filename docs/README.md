# `dnnlpy` Documentation and API Reference

The documentation is built with [Sphinx](https://www.sphinx-doc.org/), using the [Furo](https://pradyunsg.me/furo/) theme and [Sphinx AutoAPI](https://sphinx-autoapi.readthedocs.io/) to automatically generate API references from Python source code and docstrings.

## Building the Documentation

Install the package with its documentation dependencies:

```bash
uv sync --extra docs
```

Build the HTML documentation:

```bash
uv run sphinx-build -b html docs _site/
```

The generated documentation will be available in `_site/index.html`.

## API Reference Generation

Sphinx AutoAPI automatically discovers Python modules and generates API reference pages during the documentation build. No manual generation of reStructuredText (`.rst`) files is required.

When modules or docstrings are updated, simply rebuild the documentation to reflect the changes.
