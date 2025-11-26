#!/usr/bin/env python3
"""Generate find_largest_graph summaries for all algorithms under the 2 s budget."""
from __future__ import annotations

import json
from pathlib import Path

from experiments import find_largest_graph


def main() -> None:
    output_dir = Path("results/section_vi")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "random": {"max_vertices": 400, "step": 2},
        "greedy": {"max_vertices": 300, "step": 4},
        "local": {"max_vertices": 400, "step": 2},
    }

    summary = {}
    for algorithm, params in configs.items():
        result = find_largest_graph(
            algorithm,
            time_limit_per_graph=2.0,
            seed=42,
            **params,
        )
        summary[algorithm] = result
        (output_dir / f"largest_graph_{algorithm}.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(
            f"[largest-graph] {algorithm}: n={result['largest_graph']['vertices'] if result['largest_graph'] else None}"
        )

    (output_dir / "largest_graph_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
