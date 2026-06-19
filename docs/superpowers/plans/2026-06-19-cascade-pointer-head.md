# Cascade Pointer Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth `cascade` NER head with typed start prediction and a 64-token conditional end pointer.

**Architecture:** Extend the existing data collator with compact Cascade labels, add one focused head in `src/heads.py`, and route Cascade through the existing multi-block training wrapper. Cascade uses banded `[batch, length, 64]` end scores and disables the token-logit dropout penalty.

**Tech Stack:** Python 3.12, PyTorch, unittest

---

### Task 1: Cascade labels

**Files:**
- Modify: `src/data.py`
- Create: `tests/test_data.py`

- [ ] Write tests asserting a length-64 entity produces a typed start label and end offset 63.
- [ ] Write a test asserting a length-65 entity produces no Cascade label.
- [ ] Run the focused tests and verify they fail because Cascade labels do not exist.
- [ ] Extend `collate_batch` and `make_collate_fn` with `cascade_max_span_len`.
- [ ] Run the focused tests and verify they pass.

### Task 2: Cascade head

**Files:**
- Modify: `src/heads.py`
- Modify: `tests/test_heads.py`

- [ ] Write tests for `[batch, length, 64]` banded end logits, invalid-offset masking, finite empty-entity loss, and entity decoding.
- [ ] Run the focused tests and verify they fail because `CascadePointerHead` does not exist.
- [ ] Implement typed start classification, projected start/end pointer scores, two cross-entropy losses, and greedy decoding.
- [ ] Run the focused tests and verify they pass.

### Task 3: Training integration

**Files:**
- Modify: `src/train.py`
- Modify: `scripts/run_experiments.py`
- Modify: `tests/test_train.py`
- Modify: `configs.yaml`

- [ ] Write tests asserting Cascade is built with configured width 64 and `drop_penalty=0`.
- [ ] Run the focused tests and verify they fail because `build_model` does not support Cascade.
- [ ] Add Cascade model construction, label movement, decode routing, output sizing, and CLI choices.
- [ ] Add `cascade_max_span_len: 64` and `cascade_pointer_size: 64`.
- [ ] Run the focused tests and verify they pass.

### Task 4: Documentation and CPU verification

**Files:**
- Modify: `README.md`

- [ ] Document the fourth head and example command.
- [ ] Run `uv run python -m unittest discover -s tests -v`.
- [ ] Run a CPU-only synthetic forward/backward/decode smoke test.
- [ ] Confirm no CUDA process was started by the verification.
- [ ] Commit only Cascade-related files; leave `PROJECT_HANDOFF.md` untouched.

