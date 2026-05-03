# MMDS Final Project Design

## Goal

Produce a submission-ready project folder named `final__2_523K0011` that satisfies all requirements in `2526-HK2-MMDS-CK.pdf`, maximizes rubric coverage, and keeps the implementation easy to explain in an interview with the lecturer.

## Design Principles

1. Rubric-first execution: every bullet in the assignment must map to a concrete notebook section, output file, figure, or table.
2. Distributed-first pipeline: all three technical tasks must use PySpark for the heavy-lifting portions of data processing and experimentation.
3. OOP-light organization: notebooks stay self-contained, but code is organized into a small set of focused classes with explicit responsibilities.
4. Reproducibility: deterministic seeds, configuration cells, saved outputs, and execution-friendly notebooks.
5. Interviewability: short functions, explicit assumptions, and markdown explanations after hard decisions.

## Submission Structure

The final submission lives under:

- `final__2_523K0011/Source/Task01/Task01.ipynb`
- `final__2_523K0011/Source/Task02/Task02.ipynb`
- `final__2_523K0011/Source/Task03/Task03.ipynb`
- `final__2_523K0011/Report/report.tex`

Each task directory also stores the generated output artifacts required for evidence and reporting, such as CSV summaries and figures.

## Task 1 Design

Task 1 uses a synthetic but clusterable string dataset. Instead of generating 10,000 unrelated random strings, the data generator creates several prototype strings and mutates them to preserve meaningful 4-shingle overlap. This keeps the dataset valid while making hierarchical clustering non-trivial and visually demonstrable.

The task uses PySpark to generate strings, compute shingles, and derive candidate similarities. The in-memory agglomerative loop then runs on compact cluster metadata collected from Spark, which matches the assignment requirement of an in-memory agglomerative algorithm while still preserving distributed computation in the expensive preprocessing stages.

The design uses a representative-based non-Euclidean strategy:

- cluster representation: medoid-like representative chosen by minimum average Jaccard distance within the cluster
- cluster distance: Jaccard distance between current representatives
- stopping rule: terminate when merged-cluster diameter shows an abnormal jump relative to recent history

## Task 2 Design

Task 2 uses PySpark end-to-end: CSV loading, lag feature construction, train/test split, CUR-based feature reduction, linear regression training, and evaluation.

The reducer is implemented as a class with a CUR-inspired workflow:

- compute leverage scores with `RowMatrix.computeSVD`
- select informative columns for each target dimension from 15 down to 5
- select representative rows for the CUR core
- derive a compact row embedding for train and test feature vectors

Experiments are configuration-driven so that each feature dimension produces the same comparable outputs: objective history, train RMSE, test RMSE, and saved plots.

## Task 3 Design

Task 3 represents each user as a sparse rating profile where zero means missing, not negative. Similarity is computed only on co-rated items, using mean-centered cosine behavior so missing values do not penalize users.

To satisfy the fast similar-user lookup requirement, the notebook builds an LSH-style bucket index over lightweight user signatures. Candidate retrieval uses constant-time hash table lookups on average, and exact similarity is computed only on the returned candidates. The notebook then compares the optimized pipeline against a full-scan baseline for runtime and RMSE.

## Report Design

The report follows the IEEE conference format and stays within five pages. Each task section contains:

- a short problem statement
- the chosen approach
- one key figure or table
- a concise result summary

The report also includes `Contributions`, `Self-evaluation`, and `Conclusion` sections exactly as requested by the assignment.
