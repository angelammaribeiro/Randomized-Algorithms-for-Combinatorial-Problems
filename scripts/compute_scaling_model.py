#!/usr/bin/env python3
"""Estimate empirical runtime scaling exponents for each heuristic."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from experiments import run_experiment
from performance import empirical_scaling_curve, fit_complexity_curve
from randomized_mwm import Graph

ALG_ORDER = ("random", "greedy", "local")
DEFAULT_SIZES = (6, 12, 18, 24, 30)
EDGE_PROB = 0.5
DEFAULT_REPETITIONS = 2
MAX_CANDIDATES = 150
DEFAULT_TIME_LIMIT = 0.75
OUTPUT = Path("results/section_vi/empirical_scaling.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate empirical runtime scaling exponents.")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="Vertex counts to benchmark.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="Per-run time cap in seconds (set to 0 for no cap).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=MAX_CANDIDATES,
        help="Maximum candidates per run (ignored when --candidates-per-vertex > 0).",
    )
    parser.add_argument(
        "--candidates-per-vertex",
        type=int,
        default=0,
        help="If >0, budget equals value * n candidates; useful for scaling studies.",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="Repetitions per size.")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Destination JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes: Sequence[int] = tuple(sorted(set(args.sizes)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, object]] = {}
    for alg in ALG_ORDER:
        pairs: List[Tuple[int, float]] = []
        for n in sizes:
            runtimes: List[float] = []
            for rep in range(max(1, args.repetitions)):
                seed = hash((alg, n, rep)) & 0xFFFFFFFF
                rng = random.Random(seed)
                graph = Graph.random_graph(n, edge_probability=EDGE_PROB, rng=rng)
                if args.candidates_per_vertex > 0:
                    max_candidates = max(args.candidates_per_vertex * n, 1)
                else:
                    max_candidates = args.max_candidates
                time_limit = None if args.time_limit <= 0 else args.time_limit
                result = run_experiment(
                    graph,
                    alg,
                    max_candidates=max_candidates,
                    time_limit=time_limit,
                    adapt_size=True,
                )
                runtimes.append(result.runtime)
            median_time = statistics.median(runtimes)
            pairs.append((n, median_time))
        curve = fit_complexity_curve(pairs)
        incremental = empirical_scaling_curve(pairs)
        summary[alg] = {
            "pairs": pairs,
            "complexity": curve,
            "incremental_slope": incremental,
            "mean_runtime": statistics.mean(t for _, t in pairs),
        }
        print(
            f"[scaling] {alg}: alpha={curve['slope']:.2f}, inc={incremental:.2f}, last_runtime={pairs[-1][1]:.4f}s"
        )
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
