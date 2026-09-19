# Supported versions

| Component | Supported range | Reproducible development version |
| :--- | :--- | :--- |
| Python | 3.12, 3.13, 3.14 | 3.14.7 in `.python-version` |
| Polars | >=1.41.1,<2 | 1.44.2 in `uv.lock` |
| NumPy | 2.5.3 | 2.5.3 |
| Numba | 0.67.0 | 0.67.0 |
| SciPy | 1.18.1 | 1.18.1 |
| TA-Lib (optional) | 0.8.0 | 0.8.0 |

Upgrading the development environment does not require users to adopt its exact
Python or Polars version. `pyproject.toml` declares the supported range;
`uv.lock` records exact versions for reproducible development. Use
`uv sync --locked` to reproduce the lock. The default interpreter is 3.14.7;
select another supported minor with `uv sync --python 3.12 --locked` and use
the same `--python` selection on later uv commands.

## Why Python 3.12–3.14?

The current NumPy and SciPy releases require Python 3.12 or later, and the
selected numerical stack supports Python 3.14. Keeping Python 3.11 would require
older dependency versions and another validated configuration. Python 3.15 is
outside this tested range and is excluded in the package metadata.
See the [NumPy release metadata](https://pypi.org/pypi/numpy/2.5.3/json),
[SciPy release notes](https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html),
and [Numba support table](https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information).

## Why allow older Polars?

Supporting an older compatible release lets applications upgrade Polars on their
own schedule. CI tests the exact floor, **1.41.1**, and the locked release on
every supported Python, both with and without TA-Lib. These endpoint tests cover
12 environments; they do not individually test every intervening Polars release.
All optional integrations are additionally tested with locked Polars on all
three Python versions.

The former 1.41 floor is retained at its first non-yanked patch: **1.41.0 is
yanked on PyPI** ([release metadata](https://pypi.org/pypi/polars/1.41.0/json)).
Releases before 1.41.1 have no current compatibility evidence in this matrix.
Polars 2.x is excluded until its API and indicator behavior are validated.
Changing either boundary requires running the full parity and regression suite;
do not relax numerical tolerances merely to make a version pass.

See [development](development.md#ci-matrix) for exact reproduction commands and
[getting started](getting-started.md#install) for installation from source.
