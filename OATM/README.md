# OATM: Occlusion-Adaptive Temporal Memory

Active research workspace. Read [`METHODOLOGY.md`](METHODOLOGY.md) for the
scientific design, [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the
phased build plan, and
[`STUDENT_IMPLEMENTATION_ASSIGNMENT.md`](STUDENT_IMPLEMENTATION_ASSIGNMENT.md)
for how this project is meant to be built (one numbered task at a time, with
checkpoints).

This README currently documents **Phase 0 only** (the empty, testable
scaffold). Later phases will extend it as they land.

## Setup

From this `OATM/` directory:

```bash
python -m venv .venv
.venv/Scripts/pip install -e .[dev]   # Windows
# .venv/bin/pip install -e .[dev]     # macOS/Linux
```

This installs the pinned dependencies from `pyproject.toml` (currently just
`pydantic` and `pyyaml`, plus `pytest`/`ruff` for development).

## Running checks

```bash
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python -m ruff check src tests
```

Both must pass before any commit.

## Data

The local nuScenes mini dataset is expected at the repository root's `data/`
folder (`data/samples/`, `data/sweeps/`, `data/v1.0-mini/`) -- the same
location the completed assignments use. Nothing in this project hardcodes an
absolute path to it; `oatm.config.find_data_root()` discovers it relative to
the repository root, and raises a clear error if it can't be found.

## Configuration

Settings live in `configs/*.yaml` (currently just `configs/mini.yaml`) and are
loaded through `oatm.config.load_config()`, which validates every value and
resolves the dataset root, artifacts directory, and results directory. An
invalid or missing config raises `OATMConfigError` with a plain-language
message.

## Project layout (Phase 0)

```text
OATM/
  pyproject.toml       # pinned dependencies, pytest/ruff config
  configs/
    mini.yaml           # settings for the nuScenes mini split
  src/oatm/
    __init__.py
    config.py            # config loading/validation, data-root discovery
    records.py            # typed data contracts for every later phase
  scripts/                # empty until Phase 1 adds real entry points
  tests/
    unit/                 # config and record schema tests
    integration/           # end-to-end scaffold smoke test
  results/                # small, reviewable, committed reports
    project_map.md
  artifacts/              # local-only, git-ignored; large/regenerable outputs go here
```

`artifacts/` and `.venv/` are git-ignored -- see the repository root
`.gitignore`. Nothing under `OATM/` has touched the nuScenes dataset yet;
Phase 0's only job is a project that installs and tests cleanly.
