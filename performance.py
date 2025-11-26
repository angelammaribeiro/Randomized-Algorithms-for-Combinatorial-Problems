"""Performance measurement utilities for randomized MWM experiments."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import linear_regression, mean
from typing import Any, Callable, Dict, Iterable, List, Tuple

_OPERATIONS: Dict[str, int] = defaultdict(int)
_OPERATION_STACK: List[str] = []


def reset_operation_counts(label: str | None = None) -> None:
    if label:
        _OPERATIONS[label] = 0
    else:
        _OPERATIONS.clear()


def get_operation_counts() -> Dict[str, int]:
    return dict(_OPERATIONS)


def count_operations(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _OPERATION_STACK.append(label)
            try:
                result = func(*args, **kwargs)
            finally:
                _OPERATION_STACK.pop()
            return result

        return wrapper

    return decorator


def add_operations(amount: int = 1, label: str | None = None) -> None:
    target = label or (_OPERATION_STACK[-1] if _OPERATION_STACK else "global")
    _OPERATIONS[target] += amount


def operations_per_second(operations: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return operations / seconds


@dataclass
class TimingResult:
    seconds: float
    operations: int | None = None


def time_function(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> TimingResult:
    start_ops = sum(_OPERATIONS.values())
    start = time.perf_counter()
    try:
        return_value = fn(*args, **kwargs)
    finally:
        end = time.perf_counter()
    end_ops = sum(_OPERATIONS.values())
    return TimingResult(seconds=end - start, operations=end_ops - start_ops)


@contextmanager
def measure_time(label: str | None = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if label:
            print(f"[performance] {label}: {duration:.6f}s")


def estimate_time_complexity(results: Iterable[Tuple[int, float]]) -> Dict[str, float]:
    x = []
    y = []
    for size, seconds in results:
        if size > 0 and seconds > 0:
            x.append(math.log(size))
            y.append(math.log(seconds))
    if len(x) < 2:
        return {"slope": 0.0, "intercept": 0.0}
    slope, intercept = linear_regression(x, y)  # type: ignore[attr-defined]
    return {"slope": slope, "intercept": intercept}


def estimate_operation_growth(results: Iterable[Tuple[int, int]]) -> Dict[str, float]:
    sizes = []
    ops = []
    for size, count in results:
        if size > 0 and count > 0:
            sizes.append(math.log(size))
            ops.append(math.log(count))
    if len(sizes) < 2:
        return {"slope": 0.0, "intercept": 0.0}
    slope, intercept = linear_regression(sizes, ops)  # type: ignore[attr-defined]
    return {"slope": slope, "intercept": intercept}


def fit_complexity_curve(results: Iterable[Tuple[int, float]]) -> Dict[str, Any]:
    regression = estimate_time_complexity(results)
    return {
        "model": f"O(n^{regression['slope']:.2f})",
        "slope": regression["slope"],
        "intercept": regression["intercept"],
    }


def generate_complexity_plots(data: List[Tuple[int, float]], path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[performance] matplotlib not available; skipping plot.")
        return
    sizes = [d[0] for d in data]
    times = [d[1] for d in data]
    plt.figure()
    plt.loglog(sizes, times, marker="o")
    plt.xlabel("Graph size (|V|)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime scaling")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def empirical_scaling_curve(data: List[Tuple[int, float]]) -> float:
    if len(data) < 2:
        return 0.0
    slopes = []
    for (n1, t1), (n2, t2) in zip(data[:-1], data[1:]):
        if n1 > 0 and n2 > 0 and t1 > 0 and t2 > 0:
            slopes.append(math.log(t2 / t1) / math.log(n2 / n1))
    return mean(slopes) if slopes else 0.0

__all__ = [
    "count_operations",
    "add_operations",
    "reset_operation_counts",
    "get_operation_counts",
    "operations_per_second",
    "time_function",
    "measure_time",
    "estimate_time_complexity",
    "estimate_operation_growth",
    "fit_complexity_curve",
    "generate_complexity_plots",
    "empirical_scaling_curve",
]
