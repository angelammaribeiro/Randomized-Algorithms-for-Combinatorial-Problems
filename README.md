# Randomized Algorithms for Maximum Weighted Matching

This repository contains the code and report for the Advanced Algorithms project on randomized heuristics for the Maximum Weighted Matching (MWM) problem. Three complementary strategies live inside a shared controller:

- **Pure random sampling** for fast, diverse exploration.
- **Randomized greedy construction** that perturbs edge weights before a biased greedy sweep.
- **Simulated-annealing local search** that iteratively edits the incumbent matching under a cooling schedule.

The framework includes dataset loaders, instrumentation for counting basic operations, and experiment scripts that reproduce the formal/empirical analysis in the accompanying IEEE-style report (`conference_101719.tex`).

## Repository layout

| Path | Description |
|------|-------------|
| `accuracy.py`, `performance.py`, `datasets.py`, `randomized_mwm.py`, `run_all_experiments.py` | Core modules (accuracy references, instrumentation, dataset ingestion, randomized generators, and the CLI entry point). |
| `experiments.py`, `scripts/` | Helpers for orchestrating batch experiments, scaling probes, and plotting. |
| `graphs/`, `SW_ALGUNS_GRAFOS/` | Sample project/teacher graphs used throughout the benchmarks. |
| `results/` | JSON artifacts and figures referenced in Sections V and VI of the report. |
| `conference_101719.tex` | Final report (IEEEtran). |

Generated caches (virtual environments, `__pycache__`, downloaded data, etc.) were intentionally removed to keep the repository lightweight. Recreate them locally as needed.

## Prerequisites

- Python 3.10+.
- No third-party libraries are required for the core algorithms; plotting scripts rely on `matplotlib` if you want to regenerate figures.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -U pip matplotlib
```

Run the full experiment suite (graphs from `graphs/`, `SW_ALGUNS_GRAFOS/`, and optional public datasets):

```bash
python run_all_experiments.py --output-dir results/section_v_rerun
```

Reproduce the scalability probe used in Section VI:

```bash
python scripts/find_largest_graph.py --output-dir results/section_vi
python scripts/compute_scaling_model.py --input results/section_vi --output results/section_vi/empirical_scaling_from_largest.json
```

The commands above regenerate the `(n, T)` traces for random, greedy, and local search, plus the fitted exponents used in Sections VI-B and VI-C.

## Reporting & documentation

- Edit `conference_101719.tex` for the written report. Compile with `pdflatex` (TeX Live) using the IEEEtran class.
- `results/section_v_rerun/` and `results/section_vi/` contain the JSON summaries and PDF figures cited throughout the document.

## Notes

- Large public datasets are fetched on demand; populate `data_cache/` locally if you wish to avoid repeated downloads.
- Feel free to add new graph sources under `graphs/` or point the CLI at external directories via command-line flags.
- If you recreate heavy artifacts (virtual environments, caches, raw downloads), keep them out of version control to preserve the lean structure restored here.
