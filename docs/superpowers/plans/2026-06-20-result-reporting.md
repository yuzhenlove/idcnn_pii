# Experiment Result Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate isolated three-head and four-head experiment reports from the same completed metrics.

**Architecture:** Add head filtering and tagged report directories to the summarizer. Make plotting metadata and layouts depend on the selected heads instead of fixed three-head globals.

**Tech Stack:** Python 3.12, pandas, matplotlib, unittest

---

### Task 1: Add reporting regression tests

**Files:**
- Create: `tests/test_reporting.py`

- [ ] Test that three selected heads produce 36 summary rows and 12 grouped rows.
- [ ] Test that four selected heads produce 48 summary rows and 16 grouped rows.
- [ ] Test that plotting a four-head report creates all PNG, PDF, and SVG files.
- [ ] Run the tests and confirm they fail because the required APIs do not exist.

### Task 2: Add tagged summary generation

**Files:**
- Modify: `scripts/summarize_results.py`

- [ ] Add reusable summary generation accepting `outputs_dir`, `heads`, and `tag`.
- [ ] Write report CSV files under `outputs/reports/{tag}`.
- [ ] Write `logs/experiments_{tag}.csv`.
- [ ] Add CLI options `--heads` and `--tag`.
- [ ] Run the summary tests and confirm they pass.

### Task 3: Make plotting dynamic

**Files:**
- Modify: `scripts/plot_results.py`

- [ ] Add Cascade label and color metadata.
- [ ] Load tagged report CSV files.
- [ ] Pass selected heads and report figure directory through plotting functions.
- [ ] Make grouped bars, heatmaps, seed subplots, legends, and Y limits dynamic.
- [ ] Run the plotting test and confirm it passes.

### Task 4: Verify reports and publish

**Files:**
- Verify: `scripts/summarize_results.py`
- Verify: `scripts/plot_results.py`
- Verify: `tests/test_reporting.py`

- [ ] Run the complete unit test suite.
- [ ] Generate `baseline` and `all_heads` reports from the 48 completed runs.
- [ ] Verify expected CSV row counts and figure counts.
- [ ] Commit only reporting files and documentation.
- [ ] Push branch `codex/result-reporting`.
