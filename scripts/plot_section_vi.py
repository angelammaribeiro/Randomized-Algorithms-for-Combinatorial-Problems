#!/usr/bin/env python3
"""Plot runtime and candidate-operation scaling for Section VI."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ALG_LABELS = {
    "random": "Random sampling",
    "greedy": "Randomized greedy",
    "local": "Local search",
}
COLORS = {
    "random": "#2c7fb8",
    "greedy": "#f03b20",
    "local": "#4daf4a",
}


def load_data() -> dict[str, dict]:
    data = {}
    for alg in ALG_LABELS:
        path = Path(f"results/section_vi/largest_graph_{alg}.json")
        data[alg] = json.loads(path.read_text())
    return data


def main() -> None:
    output_dir = Path("results/section_vi/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    fig, (ax_time, ax_ops) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 6.0))

    for alg in ALG_LABELS:
        n_vals = data[alg]["sizes"]
        runtimes = data[alg]["runtimes"]
        cand_ops = [entry["candidate_generation"] for entry in data[alg]["operations"]]
        label = ALG_LABELS[alg]
        color = COLORS[alg]

        ax_time.plot(n_vals, runtimes, label=label, color=color)
        ax_time.scatter(
            data[alg]["largest_graph"]["vertices"],
            data[alg]["largest_graph"]["runtime"],
            color=color,
            marker="o",
        )

        ax_ops.plot(n_vals, cand_ops, label=label, color=color)
        ax_ops.scatter(
            data[alg]["largest_graph"]["vertices"],
            data[alg]["operations"][-1]["candidate_generation"],
            color=color,
            marker="o",
        )

    ax_time.axhline(2.0, color="#555555", linestyle="--", linewidth=0.9, label="2 s budget")
    ax_time.set_ylabel("Runtime (s)")
    ax_time.set_title("Largest graph search under 2 s cap")
    ax_time.legend(loc="upper right")

    ax_ops.set_ylabel("Candidate operations")
    ax_ops.set_xlabel("|V|")
    ax_ops.set_yscale("log")
    ax_ops.set_title("Candidate-generation work")

    fig.tight_layout()
    fig.savefig(output_dir / "largest_graph_scaling.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
