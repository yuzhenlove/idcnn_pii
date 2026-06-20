# NAS-IDCNN Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete three-experiment NAS-IDCNN search using NSGA-II, a diversity-preserving Archive, partial weight transfer, parallel candidate training, and final Pareto test evaluation.

**Architecture:** Keep search-specific behavior in focused modules while reusing the existing dataset, metrics, CascadePointer head, configuration, and utility functions. Candidate evaluation is a standalone process that writes an atomic result directory; the search coordinator owns NSGA-II state, caching, Archive updates, parallel launches, ETA, and resume behavior.

**Tech Stack:** Python 3.12, PyTorch, standard library multiprocessing/subprocess, unittest.

---

### Task 1: Individual encoding and experiment spaces

**Files:**
- Create: `src/nas_encoding.py`
- Create: `tests/test_nas_encoding.py`

- [ ] Write failing tests for 12D decoding, Identity normalization, experiment-specific canonicalization, random generation, crossover, mutation, and Hamming distance.
- [ ] Run `uv run python -m unittest tests.test_nas_encoding -v` and verify expected failures.
- [ ] Implement the minimum encoding functions and experiment definitions.
- [ ] Run the focused tests and commit.

### Task 2: Searchable shared-cell NAS encoder

**Files:**
- Create: `src/model_nas_idcnn.py`
- Create: `tests/test_model_nas_idcnn.py`

- [ ] Write failing tests for Conv, DWConv, SepConv, Identity, channel propagation, final channel restoration, shared Cell reuse, and last-Cell-only CascadePointer behavior.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement the searchable operations, shared Cell, encoder, and model wrapper.
- [ ] Run focused and existing model/head tests and commit.

### Task 3: NSGA-II and Archive

**Files:**
- Create: `src/nsga2.py`
- Create: `src/nas_archive.py`
- Create: `tests/test_nsga2.py`
- Create: `tests/test_nas_archive.py`

- [ ] Write failing tests for Pareto dominance, ranks, crowding distance, environmental selection, tournament selection, capacity-30 Archive update, source selection, and first-compatible-module transfer.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement NSGA-II and Archive behavior exactly as specified.
- [ ] Run focused tests and commit.

### Task 4: Candidate training and FLOPs

**Files:**
- Create: `src/nas_train.py`
- Create: `tests/test_nas_train.py`

- [ ] Write failing tests for model construction, fixed seed, Train/Dev-only dataset access, checkpoint initialization, length-128 FLOPs, and result serialization.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement candidate training using existing data, CascadePointer, metrics, and early stopping.
- [ ] Run focused tests and commit.

### Task 5: Parallel search coordinator and resume

**Files:**
- Create: `scripts/run_nas_search.py`
- Create: `tests/test_nas_search.py`

- [ ] Write failing tests for initial-population-plus-generation counting, duplicate cache reuse, parallel command construction, Archive update timing, state persistence/resume, ETA, and Pareto output.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement the coordinator with experiment isolation and `3,1,2` execution support.
- [ ] Run focused tests and commit.

### Task 6: Final Pareto test evaluation and documentation

**Files:**
- Create: `scripts/evaluate_nas_pareto.py`
- Modify: `README.md`
- Create: `tests/test_evaluate_nas_pareto.py`

- [ ] Write failing tests proving Test is only loaded by the explicit final evaluation command.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement final checkpoint Test evaluation and document smoke, 5-generation, 50-generation, resume, and final evaluation commands.
- [ ] Run focused tests and commit.

### Task 7: Verification

- [ ] Run `uv run python -m unittest discover -s tests`.
- [ ] Run CLI help and a no-training/dry-run smoke search.
- [ ] Inspect `git diff`, verify no generated data/checkpoints are staged, and verify `.gitignore` remains untouched.
- [ ] Commit remaining scoped changes locally; do not push.
