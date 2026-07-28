# Changelog

## [0.2.0] — 2026-07-27
### Added
- Centralized `GradientEngine` in `quasar/gradients.py`
- `ProgressCurriculumSampler` wired into main `Quasar` class
- `ExperimentLogger` for reproducible experiment records
- Full test suite in `tests/` using unittest
- `pyproject.toml` for installable package
- Split web demo into `docs/index.html`, `docs/quasar.js`, `docs/style.css`

### Changed
- Extracted all mathematical primitives to `quasar/core.py`
- Deduplicated finite-difference code across all modules
- Fixed "6 tests" → "7 tests" print bug in self-test
- `note_curriculum_scale.py` renamed to `experiments/curriculum_scale.py` with CI assertions

### Removed
- Committed log artifacts (`test_output_*.log`)
- Backup files from version control
- Personal shell scripts (`bind_manifold_ui.sh`)
