"""Experiment runners for randomized MWM algorithms."""
from __future__ import annotations

import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from dataclasses import asdict

from accuracy import compare_algorithms
from performance import (
    empirical_scaling_curve,
    fit_complexity_curve,
    generate_complexity_plots,
    get_operation_counts,
    reset_operation_counts,
)
from randomized_mwm import (
    Graph,
    generate_random_greedy_matching,
    generate_random_matching,
    local_search_matching,
    run_random_search,
)

ALGORITHMS: Dict[str, Callable[..., Tuple[int, ...]]] = {
    "random": generate_random_matching,
    "greedy": generate_random_greedy_matching,
    "local": local_search_matching,
}


@dataclass
class ExperimentResult:
    graph_name: str
    algorithm: str
    best_weight: float
    best_matching: Tuple[int, ...]
    runtime: float
    candidates: int
    improvements: int
    duplicates: int
    infeasible: int
    size_history: Dict[int, int]
    weight_history: List[float]
    stopping_reason: str
    operations: Dict[str, int]


def _resolve_algorithm(algorithm: str | Callable[..., Tuple[int, ...]]) -> Callable[..., Tuple[int, ...]]:
    if callable(algorithm):
        return algorithm
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'")
    return ALGORITHMS[algorithm]


def run_experiment(
    graph: Graph,
    algorithm: str | Callable[..., Tuple[int, ...]],
    params: Dict[str, Any] | None = None,
    max_candidates: int = 1000,
    time_limit: float | None = None,
    max_no_improve: int | None = None,
    adapt_size: bool = True,
) -> ExperimentResult:
    params = params or {}
    rng_seed = params.pop("seed", None)
    rng = random.Random(rng_seed)
    generator_fn = _resolve_algorithm(algorithm)
    reset_operation_counts()
    start = time.perf_counter()
    result = run_random_search(
        graph=graph,
        generator_fn=generator_fn,
        rng=rng,
        max_candidates=max_candidates,
        time_limit=time_limit,
        max_no_improve=max_no_improve,
        adapt_size=adapt_size,
        **params,
    )
    runtime = time.perf_counter() - start
    return ExperimentResult(
        graph_name=getattr(graph, "name", "graph"),
        algorithm=algorithm.__name__ if callable(algorithm) else str(algorithm),
        best_weight=result["best_weight"],
        best_matching=result["best_matching"],
        runtime=runtime,
        candidates=result["num_candidates"],
        improvements=result["num_improvements"],
        duplicates=result["duplicate_rejections"],
        infeasible=result["infeasible_rejections"],
        size_history=result["size_histogram"],
        weight_history=result["weight_history"],
        stopping_reason=result["stopping_reason"],
        operations=get_operation_counts(),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_benchmark_suite(
    graphs: Dict[str, Graph],
    algorithms: Sequence[str],
    repetitions: int,
    output_folder: Path,
    max_candidates: int = 1000,
    time_limit: float | None = None,
    compare_accuracy: bool = False,
    plot: bool = False,
) -> List[ExperimentResult]:
    output_folder.mkdir(parents=True, exist_ok=True)
    all_results: List[ExperimentResult] = []
    accuracy_collector: Dict[str, Dict[str, List[float]]] = {}
    for graph_name, graph in graphs.items():
        graph.name = graph_name  # type: ignore[attr-defined]
        accuracy_collector[graph_name] = {alg: [] for alg in algorithms}
        for alg in algorithms:
            for rep in range(repetitions):
                params = {"seed": hash((graph_name, alg, rep)) & 0xFFFFFFFF}
                result = run_experiment(
                    graph,
                    alg,
                    params=params,
                    max_candidates=max_candidates,
                    time_limit=time_limit,
                )
                all_results.append(result)
                accuracy_collector[graph_name][alg].append(result.best_weight)
                record = {
                    "graph": graph_name,
                    "algorithm": alg,
                    "repetition": rep,
                    "best_weight": result.best_weight,
                    "runtime": result.runtime,
                    "candidates": result.candidates,
                    "improvements": result.improvements,
                    "duplicates": result.duplicates,
                    "infeasible": result.infeasible,
                    "stopping_reason": result.stopping_reason,
                }
                _append_csv(output_folder / "raw_results.csv", record)
                _write_json(
                    output_folder / f"{graph_name}_{alg}_{rep}.json",
                    {
                        "graph": graph_name,
                        "algorithm": alg,
                        "result": result.__dict__,
                    },
                )
            if plot and result.weight_history:
                try:
                    import matplotlib.pyplot as plt  # type: ignore

                    plt.figure()
                    plt.plot(result.weight_history)
                    plt.xlabel("Candidates")
                    plt.ylabel("Weight")
                    plt.title(f"Convergence {graph_name} - {alg}")
                    plot_path = output_folder / f"{graph_name}_{alg}_convergence.png"
                    plt.savefig(plot_path, bbox_inches="tight")
                    plt.close()
                except ImportError:
                    print("[experiments] matplotlib not available; skipping plots")
        if compare_accuracy:
            summary = compare_algorithms(graph, accuracy_collector[graph_name])
            serializable = {k: asdict(v) for k, v in summary.items()}
            _write_json(output_folder / f"{graph_name}_accuracy.json", serializable)
    return all_results


def summarize_results(results: List[ExperimentResult]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for result in results:
        key = (result.graph_name, result.algorithm)
        group = summary.setdefault(key, {
            "weights": [],
            "runtimes": [],
            "candidates": [],
        })
        group["weights"].append(result.best_weight)
        group["runtimes"].append(result.runtime)
        group["candidates"].append(result.candidates)
    final: Dict[str, Dict[str, Any]] = {}
    for (graph_name, algorithm), group in summary.items():
        final[f"{graph_name}:{algorithm}"] = {
            "mean_weight": statistics.mean(group["weights"]),
            "std_weight": statistics.pstdev(group["weights"]) if len(group["weights"]) > 1 else 0.0,
            "mean_runtime": statistics.mean(group["runtimes"]),
            "mean_candidates": statistics.mean(group["candidates"]),
        }
    return final


def find_largest_graph(
    algorithm: str,
    time_limit_per_graph: float,
    seed: int = 0,
    max_vertices: int = 200,
    min_vertices: int = 4,
    step: int = 1,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    sizes = []
    runtimes = []
    operations: List[Dict[str, int]] = []
    last_success = None
    tolerance = max(0.05, 0.01 * time_limit_per_graph)
    for vertices in range(min_vertices, max_vertices + 1, max(1, step)):
        try:
            graph = Graph.random_graph(vertices, edge_probability=min(0.5, 4 / vertices), rng=rng)
        except ValueError:
            continue
        start = time.perf_counter()
        result = run_experiment(graph, algorithm, max_candidates=2000, time_limit=time_limit_per_graph)
        duration = time.perf_counter() - start
        if duration > time_limit_per_graph + tolerance:
            break
        last_success = {
            "vertices": vertices,
            "edges": len(graph.edges),
            "runtime": duration,
            "best_weight": result.best_weight,
        }
        sizes.append(vertices)
        runtimes.append(duration)
        operations.append(dict(result.operations))
    slope = empirical_scaling_curve(list(zip(sizes, runtimes))) if sizes else 0.0
    complexity = fit_complexity_curve(list(zip(sizes, runtimes))) if sizes else {}
    return {
        "largest_graph": last_success,
        "sizes": sizes,
        "runtimes": runtimes,
        "operations": operations,
        "scaling_slope": slope,
        "complexity": complexity,
        "extrapolated_runtime": None if not sizes else (runtimes[-1] * ((max_vertices + 20) / sizes[-1]) ** slope if slope else None),
    }


__all__ = [
    "run_experiment",
    "run_benchmark_suite",
    "summarize_results",
    "find_largest_graph",
]
