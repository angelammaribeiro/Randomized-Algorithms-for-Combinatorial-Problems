#!/usr/bin/env python3
"""CLI entry point to run all randomized MWM experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import datasets
from accuracy import brute_force_optimal
from experiments import find_largest_graph, run_benchmark_suite, summarize_results
from randomized_mwm import Graph


def _load_requested_graphs(dataset_args: List[str]) -> Dict[str, Graph]:
    graphs: Dict[str, Graph] = {}
    if not dataset_args or dataset_args == ["all"]:
        graphs = datasets.load_all_graphs(include_external=True)
        return graphs
    for arg in dataset_args:
        path = Path(arg)
        if path.exists():
            graphs.update(datasets.load_graph_directory(path))
        elif arg == "project":
            graphs.update(datasets.load_project_graphs())
        elif arg == "teacher":
            graphs.update(datasets.load_teacher_graphs())
        elif arg == "mendeley":
            graphs.update(datasets.load_mendeley_graphs())
        elif arg == "snap":
            snap_graphs = datasets.load_default_snap_graphs()
            if snap_graphs:
                graphs.update(snap_graphs)
            else:
                print("[cli] SNAP graphs require explicit filenames; none loaded by default.")
        elif arg == "brunel":
            graphs.update(datasets.load_brunel_graphs())
        else:
            print(f"[cli] Unknown dataset '{arg}'")
    return graphs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run randomized MWM experiments")
    parser.add_argument("--datasets", nargs="*", default=["project", "teacher"], help="Datasets to include or directories")
    parser.add_argument("--algorithms", nargs="*", default=["random", "greedy", "local"], help="Algorithms to run")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--compare-optimal", action="store_true")
    parser.add_argument("--output-folder", type=str, default="results")
    parser.add_argument("--find-largest-graph", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--max-vertices", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    graphs = _load_requested_graphs(args.datasets)
    if not graphs:
        raise SystemExit("No graphs available for experiments")
    output = Path(args.output_folder)
    results = run_benchmark_suite(
        graphs,
        args.algorithms,
        repetitions=args.repetitions,
        output_folder=output,
        max_candidates=args.max_candidates,
        time_limit=args.time_limit,
        compare_accuracy=args.compare_optimal,
        plot=args.plot,
    )
    summary = summarize_results(results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, stats in summary.items():
        print(f"{key}: mean_weight={stats['mean_weight']:.2f} runtime={stats['mean_runtime']:.3f}s candidates={stats['mean_candidates']:.1f}")
    if args.find_largest_graph:
        largest = find_largest_graph(args.algorithms[0], time_limit_per_graph=args.time_limit or 5.0, max_vertices=args.max_vertices)
        (output / "largest_graph.json").write_text(json.dumps(largest, indent=2), encoding="utf-8")
        print("[cli] Largest graph summary saved to largest_graph.json")


if __name__ == "__main__":
    main()
