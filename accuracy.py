"""Accuracy evaluation utilities for randomized MWM experiments."""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from randomized_mwm import Graph, canonical_matching, is_feasible_matching, matching_weight


@dataclass
class AccuracyMetrics:
    best_weight: float
    reference_weight: float
    absolute_gap: float
    relative_gap: float
    approximation_ratio: float


@dataclass
class AccuracySummary:
    mean_weight: float
    std_weight: float
    min_weight: float
    max_weight: float
    absolute_gap: float
    relative_gap: float
    approximation_ratio: float


def brute_force_optimal(graph: Graph, limit_edges: int = 20) -> Tuple[Tuple[int, ...], float]:
    if len(graph.edges) > limit_edges:
        raise ValueError("Graph too large for brute force optimal computation")
    best_matching: Tuple[int, ...] = tuple()
    best_weight = -1.0
    for size in range(len(graph.edges) + 1):
        for combo in itertools.combinations(range(len(graph.edges)), size):
            matching = canonical_matching(combo)
            if not is_feasible_matching(graph, matching):
                continue
            weight = matching_weight(graph, matching)
            if weight > best_weight:
                best_matching = matching
                best_weight = weight
    return best_matching, best_weight


def compute_accuracy_metrics(best_weight: float, reference_weight: float) -> AccuracyMetrics:
    absolute_gap = reference_weight - best_weight
    relative_gap = absolute_gap / reference_weight if reference_weight else 0.0
    approximation_ratio = best_weight / reference_weight if reference_weight else 0.0
    return AccuracyMetrics(
        best_weight=best_weight,
        reference_weight=reference_weight,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        approximation_ratio=approximation_ratio,
    )


def summarize_accuracy(samples: Sequence[float], reference: float) -> AccuracySummary:
    mean_weight = statistics.mean(samples) if samples else 0.0
    std_weight = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    min_weight = min(samples) if samples else 0.0
    max_weight = max(samples) if samples else 0.0
    metrics = compute_accuracy_metrics(mean_weight, reference)
    return AccuracySummary(
        mean_weight=mean_weight,
        std_weight=std_weight,
        min_weight=min_weight,
        max_weight=max_weight,
        absolute_gap=metrics.absolute_gap,
        relative_gap=metrics.relative_gap,
        approximation_ratio=metrics.approximation_ratio,
    )


def compare_algorithms(
    graph: Graph,
    algorithm_results: Dict[str, List[float]],
    reference_weight: Optional[float] = None,
) -> Dict[str, AccuracySummary]:
    summaries: Dict[str, AccuracySummary] = {}
    ref = reference_weight
    if ref is None:
        try:
            _, ref = brute_force_optimal(graph)
        except ValueError:
            ref = max((max(weights) for weights in algorithm_results.values() if weights), default=0.0)
    for name, weights in algorithm_results.items():
        summaries[name] = summarize_accuracy(weights, ref)
    return summaries


def evaluate_against_reference(
    graph: Graph,
    randomized_weight: float,
    exact_solver_weight: Optional[float],
    heuristic_weight: Optional[float],
) -> Dict[str, AccuracyMetrics]:
    reference = exact_solver_weight or heuristic_weight or randomized_weight
    metrics = {
        "randomized": compute_accuracy_metrics(randomized_weight, reference)
    }
    if exact_solver_weight is not None:
        metrics["optimal"] = compute_accuracy_metrics(exact_solver_weight, exact_solver_weight)
    if heuristic_weight is not None:
        metrics["heuristic"] = compute_accuracy_metrics(heuristic_weight, reference)
    return metrics


__all__ = [
    "AccuracyMetrics",
    "AccuracySummary",
    "brute_force_optimal",
    "compute_accuracy_metrics",
    "summarize_accuracy",
    "compare_algorithms",
    "evaluate_against_reference",
]
