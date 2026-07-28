# Contributing to QUASAR

## Reproductions (Most Valuable)

Independent reproductions (or failures!) are the most valuable contribution.

1. Run `python run_all_tests.py` on your machine.
2. Open a [reproduction issue](https://github.com/holland202/quasar/issues/new) with:
   - Output of `run_all_tests.py`
   - `python --version`
   - Platform (e.g., `Linux-aarch64`, `macOS-arm64`, `Windows-x86_64`)
   - Runtime for each suite

## Code Style

- Pure NumPy only. No PyTorch, no JAX, no TensorFlow.
- Every mathematical claim must have a numerical test with a tolerance.
- Negative results must be kept and documented.
- Docstrings for all public functions.

## Pull Requests

1. Add tests in `tests/` for new functionality.
2. Ensure `python run_all_tests.py` passes.
3. Update `CHANGELOG.md`.
