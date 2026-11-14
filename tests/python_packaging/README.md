# Python packaging tests

The core package tests run without native extensions. The split Tensor tests
use the actual `_tensor` shared library from a Tensor-only CMake build. Point
`LABNEURA_TENSOR_EXTENSION` at that built artifact to run them:

```sh
LABNEURA_TENSOR_EXTENSION=/absolute/path/to/_tensor.so \
  .venv/bin/python -m pytest -p no:cacheprovider \
  tests/python_packaging/test_tensor_extension.py
```

The tests copy the supplied binary and the core initializer into an isolated
temporary `labneura` package, then run import checks in subprocesses with that
package on `PYTHONPATH`. Without the environment variable, these extension
tests report an explicit skip so ordinary test collection works before the
split extension is built. If the variable is set to an invalid path or an
extension with an incompatible suffix, the tests fail.

The independent EDA extension tests use the same isolated-package pattern.
Point `LABNEURA_EDA_EXTENSION` at a built `_eda` shared library:

```sh
LABNEURA_EDA_EXTENSION=/absolute/path/to/_eda.so \
  .venv/bin/python -m pytest -p no:cacheprovider \
  tests/python_packaging/test_eda_extension.py
```

The EDA checks verify native loading, package imports, representative EDA APIs,
Verilog parsing, the default FPGA library, and the absence of Tensor in an
EDA-only package. Without the environment variable, they report an explicit
skip; an invalid supplied path or extension suffix fails the tests.
