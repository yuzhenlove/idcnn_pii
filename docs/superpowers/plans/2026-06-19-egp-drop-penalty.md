# EGP Dropout Penalty Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the token-logit dropout penalty from collapsing the quadratic EGP span output while preserving Softmax and CRF behavior.

**Architecture:** Keep the shared training configuration unchanged. Select the effective penalty in `build_model`: use the configured value for token heads and zero for EGP.

**Tech Stack:** Python 3.12, PyTorch, unittest

---

### Task 1: Add the regression test

**Files:**
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds all three heads and asserts:

```python
self.assertEqual(softmax_model.drop_penalty, 0.0001)
self.assertEqual(crf_model.drop_penalty, 0.0001)
self.assertEqual(egp_model.drop_penalty, 0.0)
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run python -m unittest tests.test_train.TrainConfigurationTest.test_egp_disables_token_logit_dropout_penalty -v
```

Expected: failure because EGP currently receives `0.0001`.

### Task 2: Implement the minimal fix

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Select the effective penalty by head**

Use:

```python
drop_penalty = 0.0 if head == "egp" else cfg["train"]["drop_penalty"]
return IDCNNForTokenClassification(encoder, token_head, drop_penalty=drop_penalty)
```

- [ ] **Step 2: Verify the focused test passes**

Run:

```bash
uv run python -m unittest tests.test_train.TrainConfigurationTest.test_egp_disables_token_logit_dropout_penalty -v
```

Expected: one passing test.

### Task 3: Verify and commit

**Files:**
- Verify: `src/train.py`
- Verify: `tests/test_train.py`

- [ ] **Step 1: Run the full test suite**

```bash
uv run python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run a short EGP diagnostic**

Train a few batches and compare supervised loss with the effective dropout penalty. Expected:
`model.drop_penalty == 0.0` and no quadratic penalty is added.

- [ ] **Step 3: Commit the fix**

```bash
git add src/train.py tests/test_train.py docs/superpowers
git commit -m "fix egp dropout penalty collapse"
```

