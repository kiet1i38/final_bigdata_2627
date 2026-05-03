# MMDS Final Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a submission-ready `final__2_523K0011` folder with three executed notebooks and one IEEE-style report for the MMDS final project.

**Architecture:** Each task is implemented in one self-contained notebook with small, focused classes and saved outputs. PySpark handles the distributed parts of the pipeline, while local Python handles notebook-friendly orchestration, plotting, and report-ready summaries.

**Tech Stack:** Python, PySpark, pandas, numpy, matplotlib, scikit-learn, Jupyter notebook, LaTeX

---

### Task 1: Workspace And Submission Layout

**Files:**
- Create: `final__2_523K0011/Source/Task01/`
- Create: `final__2_523K0011/Source/Task02/`
- Create: `final__2_523K0011/Source/Task03/`
- Create: `final__2_523K0011/Report/`

- [ ] Create the required submission directories.
- [ ] Save the design document and implementation plan under `docs/superpowers/`.
- [ ] Verify the expected folder tree exists.

### Task 2: Task01 Notebook

**Files:**
- Create: `final__2_523K0011/Source/Task01/Task01.ipynb`
- Create: `final__2_523K0011/Source/Task01/task01_strings.csv`
- Create: `final__2_523K0011/Source/Task01/task01_metrics.csv`
- Create: `final__2_523K0011/Source/Task01/task01_global_avg_dist.png`
- Create: `final__2_523K0011/Source/Task01/task01_tsne_3d.png`

- [ ] Build a self-contained notebook with rubric mapping, Spark configuration, and deterministic seeds.
- [ ] Implement the distributed string dataset generator and 4-shingle pipeline.
- [ ] Save the generated dataset with columns `index`, `string`, `shingles`.
- [ ] Implement the OOP agglomerative clustering workflow.
- [ ] Execute the notebook and confirm all output artifacts are generated.

### Task 3: Task02 Notebook

**Files:**
- Create: `final__2_523K0011/Source/Task02/Task02.ipynb`
- Create: `final__2_523K0011/Source/Task02/gold_prices.csv`
- Create: `final__2_523K0011/Source/Task02/task02_results.csv`
- Create: `final__2_523K0011/Source/Task02/task02_loss_curves.png`
- Create: `final__2_523K0011/Source/Task02/task02_twin_bars.png`

- [ ] Copy the source CSV into the task folder for self-contained execution.
- [ ] Implement the configurable PySpark data pipeline for lag-feature generation.
- [ ] Implement the CUR-based reducer class and experiment runner.
- [ ] Execute the notebook and save metrics and figures.

### Task 4: Task03 Notebook

**Files:**
- Create: `final__2_523K0011/Source/Task03/Task03.ipynb`
- Create: `final__2_523K0011/Source/Task03/ratings2k.csv`
- Create: `final__2_523K0011/Source/Task03/task03_rmse_by_n.png`
- Create: `final__2_523K0011/Source/Task03/task03_runtime_compare.png`
- Create: `final__2_523K0011/Source/Task03/task03_results.csv`

- [ ] Copy the source CSV into the task folder for self-contained execution.
- [ ] Implement the sparse user-profile builder and user-user similarity logic.
- [ ] Implement the fast similar-user lookup index and the baseline lookup.
- [ ] Execute the notebook and save RMSE and runtime comparisons.

### Task 5: Report

**Files:**
- Create: `final__2_523K0011/Report/report.tex`
- Create: `final__2_523K0011/Report/references.bib`

- [ ] Adapt the IEEE example into the project report structure.
- [ ] Fill in authors, abstract, three task sections, contributions, self-evaluation, conclusion, and references.
- [ ] Reuse the generated figures from the three notebooks.

### Task 6: Final Verification

**Files:**
- Verify: `final__2_523K0011/`

- [ ] Re-execute notebooks as needed so outputs are preserved in the `.ipynb` files.
- [ ] Verify required artifacts exist in each task folder.
- [ ] Verify report source exists and follows the required section structure.
- [ ] Summarize any remaining assumptions or limitations for the user.
