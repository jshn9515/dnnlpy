# Contributing

Thank you for your interest in contributing to **dnnlpy**.

We welcome bug fixes, new features, tests, documentation improvements, and other contributions that help improve the project.

Please follow the guidelines below before submitting a Pull Request.

## 1. Set up the development environment

This project uses [uv](https://docs.astral.sh/uv/) to manage Python versions, dependencies, and virtual environments.

First, install uv by following the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Then, clone the repository:

```bash
git clone https://github.com/jshn9515/dnnlpy.git
cd dnnlpy
```

Install the project and its development dependencies:

```bash
uv sync --locked --all-extras
```

This creates a `.venv` environment and installs `dnnlpy` in editable mode, along with its dependencies.

The project supports Python 3.12-3.15, including free-threaded Python 3.14t and 3.15t. Python 3.14 is recommended for local development. Some dependencies may need to be built from source on free-threaded Python due to the missing prebuilt wheels. These builds may require additional system libraries, such as `pygame-ce`. If you do not need all optional features, you can install only the dependencies required for testing and linting:

```bash
uv sync --locked --extra test --extra dev
```

## 2. Install pre-commit hooks

The project uses [pre-commit](https://pre-commit.com/) to automate code quality checks.

Install pre-commit and configure the Git hooks:

```bash
uv tool install pre-commit
pre-commit install
```

You can also run all hooks manually:

```bash
pre-commit run --all-files
```

Please ensure that all checks pass before submitting a Pull Request.

## 3. Format and lint your code

The project uses [Ruff](https://docs.astral.sh/ruff/) for Python formatting and linting.

Format the code:

```bash
uv run ruff format .
```

Check for linting issues:

```bash
uv run ruff check .
```

Automatically fix supported linting issues:

```bash
uv run ruff check . --fix
```

Please follow the existing code style and avoid unrelated formatting changes.

## 4. Type-check your code

The project uses [ty](https://docs.astral.sh/ty/) for Python type checking. Run it from the repository root with:

```bash
uv run --extra dev ty check .
```

Resolve type errors in code you change before submitting a Pull Request.

## 5. Run tests

The project uses [pytest](https://docs.pytest.org/) for automated testing.

Run the test suite:

```bash
uv run --extra test pytest tests/
uv run pytest --doctest-modules src/  # for doctests
```

If your changes introduce new functionality or fix a bug, please add or update the corresponding tests.

The GitHub Actions CI runs tests across the following Python versions:

- Python 3.12
- Python 3.13
- Python 3.14 and 3.14t
- Python 3.15 and 3.15t

You do not need to test every Python version locally, but your changes should remain compatible with the supported versions.

## 6. Build the documentation (optional)

The documentation is built using [Sphinx](https://www.sphinx-doc.org/) with the Furo theme and Sphinx AutoAPI. If your changes affect documentation or public APIs, verify that the documentation builds successfully.

Install the documentation dependencies:

```bash
uv sync --python 3.14 --locked --extra docs
```

Build the HTML documentation:

```bash
uv run sphinx-build -b html docs _site/
```

After building, open `_site/index.html` to review the generated documentation.

The published documentation is available at [dnnlpy Documentation](https://jshn9515.github.io/dnnlpy/).

## 7. Sign your commits

This repository requires verified commits.

Before submitting a Pull Request, ensure that your Git commits are signed using a signing method supported by GitHub, such as GPG or SSH.

For setup instructions, refer to [GitHub&#39;s commit signature verification documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification).

Your commits should display the **Verified** badge on GitHub.

## 8. Submit a Pull Request

Once your changes are ready, open a Pull Request against the `main` branch.

Please ensure that:

- Your Pull Request has a clear and descriptive title.
- The description summarizes the changes and their motivation.
- Relevant tests have been added or updated.
- Formatting, linting, type checking, and tests pass.
- Documentation has been updated when necessary.
- Your commits are verified.

For substantial features or architectural changes, consider opening an Issue first to discuss your proposal.

Typical contributions include:

- Fixing bugs and improving existing implementations.
- Adding new algorithms, utilities, or learning examples.
- Improving test coverage and edge-case handling.
- Enhancing documentation and API references.
- Improving performance, compatibility, and CI workflows.

Thank you for contributing to **dnnlpy**.
