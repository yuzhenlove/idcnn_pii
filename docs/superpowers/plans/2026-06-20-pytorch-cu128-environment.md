# PyTorch CUDA 12.8 Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incompatible PyTorch CUDA 13.0 build with an official PyTorch 2.11.0 CUDA 12.8 build that runs on the server's NVIDIA 570.124.04 driver and five RTX 5090 GPUs.

**Architecture:** Configure uv to resolve `torch==2.11.0` from PyTorch's explicit CUDA 12.8 wheel index, regenerate the lockfile, and synchronize the existing project environment. Verify compatibility with actual CUDA allocation and BF16 computation on every GPU before running the repository test suite.

**Tech Stack:** uv 0.8.13, Python 3.12, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA RTX 5090

---

### Task 1: Pin the compatible PyTorch distribution

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add the official CUDA 12.8 PyTorch index and pin torch**

Configure `pyproject.toml` with:

```toml
dependencies = [
    "torch==2.11.0",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
]
```

Then run:

```bash
uv lock
```

Expected: `uv.lock` resolves CUDA 12 dependencies instead of CUDA 13 dependencies.

- [ ] **Step 2: Synchronize from the updated lockfile**

Run:

```bash
uv sync --locked
```

Expected: the project virtual environment installs `torch 2.11.0+cu128`.

### Task 2: Verify CUDA runtime compatibility

**Files:**
- No source changes

- [ ] **Step 1: Verify package and driver visibility**

Run:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected: PyTorch reports `2.11.0+cu128`, CUDA `12.8`, availability `True`, and device count `5`.

- [ ] **Step 2: Run CUDA and BF16 computation on every GPU**

Run a Python probe that creates BF16 matrices on devices 0 through 4, performs matrix multiplication, synchronizes each device, and reports the result.

Expected: all five devices complete without CUDA initialization, architecture, or BF16 errors.

### Task 3: Verify the repository baseline

**Files:**
- No source changes

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
uv run python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 2: Inspect the final dependency diff**

Run:

```bash
git diff -- pyproject.toml uv.lock
git status --short
```

Expected: only the PyTorch dependency configuration, lockfile, and this implementation plan are changed. Do not commit until the user explicitly approves a commit.
