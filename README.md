# dnnlpy

[![PyPI](https://img.shields.io/pypi/v/dnnlpy)](https://pypi.org/project/dnnlpy/)
[![Python](https://img.shields.io/pypi/pyversions/dnnlpy.svg)](https://pypi.org/project/dnnlpy/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![dnnlpy-ci](https://github.com/jshn9515/dnnlpy/actions/workflows/dnnlpy-ci.yml/badge.svg)](https://github.com/jshn9515/dnnlpy/actions/workflows/dnnlpy-ci.yml)
[![dnnlpy-docs](https://github.com/jshn9515/dnnlpy/actions/workflows/build-docs.yml/badge.svg)](https://github.com/jshn9515/dnnlpy/actions/workflows/build-docs.yml)
[![pre-commit.ci](https://results.pre-commit.ci/badge/github/jshn9515/dnnlpy/main.svg)](https://results.pre-commit.ci/latest/github/jshn9515/dnnlpy/main)
[![Codecov](https://codecov.io/gh/jshn9515/dnnlpy/graph/badge.svg?branch=main)](https://codecov.io/gh/jshn9515/dnnlpy)
[![Downloads](https://static.pepy.tech/badge/dnnlpy/month)](https://pepy.tech/project/dnnlpy)

**dnnlpy** is the companion Python package for **dnnl-notes**. It provides code examples, helper functions, and small utilities used throughout the tutorial, similar in spirit to the `d2l` package for _Dive into Deep Learning_.

The package follows a PyTorch-like structure, with lightweight, readable implementations designed for educational use:

- `dnnlpy.nn` provides reusable neural network modules, such as attention layers and positional encodings.
- `dnnlpy.nn.functional` provides stateless operations, such as functional attention implementations.
- `dnnlpy.optim` implements optimizers, such as SGD and Adam.

The APIs follow PyTorch conventions where practical, prioritizing simplicity and readability over framework-level completeness. The package serves as a code supplement to the notes, making examples easier to run, reuse, and extend.

The package also includes code for Stanford courses such as CS221 and CS234. Course modules use the `dnnlpy.csxxx` import path convention (for example, `dnnlpy.cs234`). See the `README.md` files in the course directories for more information.

## Requirements

- Python 3.12 or newer
- PyTorch 2.15 or newer

## Installation

Install the published package from PyPI with:

```bash
uv pip install dnnlpy
```

To install the latest version directly from this repository, use:

```bash
uv pip install https://github.com/jshn9515/dnnlpy.git
```

This project uses [uv](https://docs.astral.sh/uv/) for local package development. If you want to contribute to the package, you can install it in editable mode with:

```bash
git clone https://github.com/jshn9515/dnnlpy.git
cd dnnlpy
uv pip install -e .
```

This way, changes to the source code take effect immediately without reinstalling the package each time.

## Examples

After installation, you can import reusable neural network modules from `dnnlpy.nn`:

```python
import dnnlpy.nn as dnn
import torch

attn = dnn.MultiheadAttention(embed_dim=16, num_heads=4)

query = torch.randn(2, 8, 16)
key = torch.randn(2, 8, 16)
value = torch.randn(2, 8, 16)

output = attn(query, key, value)
```

You can also import stateless functions from `dnnlpy.nn.functional`:

```python
import dnnlpy.nn.functional as dF
import torch

query = torch.randn(2, 4, 8, 16)
key = torch.randn(2, 4, 8, 16)
value = torch.randn(2, 4, 8, 16)

output, weights = dF.scaled_dot_product_attention(
    query,
    key,
    value,
    need_weights=True,
)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

This project is licensed under the **MIT License**.
