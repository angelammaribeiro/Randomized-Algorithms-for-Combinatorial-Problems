"""Dataset loading utilities for randomized MWM experiments.

This module centralizes all logic required to obtain graph instances from the
project's local folders as well as third-party repositories. All formats are
normalized into the ``Graph`` class defined in ``randomized_mwm``.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import urlretrieve

from randomized_mwm import Graph

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_GRAPH_DIR = ROOT_DIR / "graphs"
TEACHER_GRAPH_DIR = ROOT_DIR / "SW_ALGUNS_GRAFOS"
DATA_CACHE_DIR = ROOT_DIR / "data_cache"
DATA_CACHE_DIR.mkdir(exist_ok=True)

MENDELEY_URL = "https://data.mendeley.com/datasets/rr5bkj6dw5/9/files"
SNAP_BASE_URL = "https://snap.stanford.edu/data"
BRUNEL_BASE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/info.html"
SNAP_DEFAULT_GRAPHS = {
    "ca-GrQc": "ca-GrQc.txt.gz",
    "email-Enron": "email-Enron.txt.gz",
}


@dataclass
class GraphRecord:
    name: str
    graph: Graph
    source: str


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json"}:
        return "json"
    if suffix in {".txt", ".dat"}:
        text = _safe_read_text(path)
        lower_name = path.name.lower()
        if "adj_matrix" in lower_name:
            return "adj_matrix"
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("adjacency"):
                return "adj_matrix"
            break
        if "EDGE_WEIGHT_SECTION" in text:
            return "tsplib"
        if text.strip().startswith("DIMENSION"):
            return "tsplib"
        if any(line.startswith("#") for line in text.splitlines()):
            return "snap"
        if any("," in line for line in text.splitlines()[:5]):
            return "csv"
        return "edge_list"
    if suffix in {".mtx"}:
        return "matrix_market"
    if suffix in {".gz"}:
        return "gzip"
    raise ValueError(f"Unknown graph format for file {path}")


def _edge_list_from_lines(lines: Iterable[str], weighted: bool = True) -> Tuple[int, List[Tuple[int, int, float]]]:
    edges: List[Tuple[int, int, float]] = []
    max_vertex = -1
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        u, v = int(parts[0]), int(parts[1])
        w = float(parts[2]) if weighted and len(parts) > 2 else 1.0
        if u == v:
            continue
        edges.append((u, v, w))
        max_vertex = max(max_vertex, u, v)
    if max_vertex < 1:
        raise ValueError("Edge list did not contain any valid edges")
    return max_vertex + 1, edges


def _parse_json_graph(path: Path) -> Graph:
    data = json.loads(_safe_read_text(path))
    if isinstance(data, dict) and "edges" in data and "num_vertices" in data:
        num_vertices = int(data["num_vertices"])
        edges = [(int(u), int(v), float(w)) for u, v, w in data["edges"]]
        return Graph(num_vertices, edges)
    if isinstance(data, list):
        edges = [(int(e[0]), int(e[1]), float(e[2])) for e in data]
        num_vertices = max(max(u, v) for u, v, _ in edges) + 1
        return Graph(num_vertices, edges)
    raise ValueError(f"Unsupported JSON structure in {path}")


def _parse_tsplib(path: Path) -> Graph:
    content = _safe_read_text(path)
    num_vertices = 0
    edges: List[Tuple[int, int, float]] = []
    reading_matrix = False
    matrix_values: List[List[float]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("DIMENSION"):
            num_vertices = int(stripped.split(":")[-1])
        elif stripped.upper() == "EDGE_WEIGHT_SECTION":
            reading_matrix = True
            continue
        elif stripped.upper() == "EOF":
            break
        elif reading_matrix:
            row_values = [float(val) for val in stripped.split()]
            matrix_values.append(row_values)
    if num_vertices == 0:
        num_vertices = len(matrix_values)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            w = matrix_values[i][j]
            if w > 0:
                edges.append((i, j, float(w)))
    return Graph(num_vertices, edges)


def _parse_snap(path: Path) -> Graph:
    lines = [line for line in _safe_read_text(path).splitlines()]
    num_vertices, edges = _edge_list_from_lines(lines, weighted=False)
    weighted_edges = [(u, v, 1.0) for u, v, _ in edges]
    return Graph(num_vertices, weighted_edges)


def _parse_edge_list(path: Path) -> Graph:
    with path.open("r", encoding="utf-8") as handle:
        num_vertices, edges = _edge_list_from_lines(handle)
    return Graph(num_vertices, edges)


def _parse_matrix_market(path: Path) -> Graph:
    with path.open("r", encoding="utf-8") as handle:
        edges: List[Tuple[int, int, float]] = []
        num_vertices = 0
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) == 3 and num_vertices == 0:
                num_vertices = max(int(parts[0]), int(parts[1]))
                continue
            if len(parts) >= 3:
                u, v, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                edges.append((u, v, w))
                num_vertices = max(num_vertices, u + 1, v + 1)
    return Graph(num_vertices, edges)


def _parse_gzip(path: Path) -> Graph:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.readlines()
    num_vertices, edges = _edge_list_from_lines(lines)
    return Graph(num_vertices, edges)


def _parse_adjacency_matrix(path: Path) -> Graph:
    content = _safe_read_text(path).splitlines()
    numeric_rows: List[List[float]] = []
    for raw_line in content:
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if all(part.replace(".", "", 1).lstrip("-+").isdigit() for part in parts):
            numeric_rows.append([float(part) for part in parts])
        elif numeric_rows:
            break
        else:
            continue
    if not numeric_rows:
        raise ValueError(f"Adjacency matrix in {path} did not contain numeric rows")
    header = numeric_rows[0]
    if all(abs(value - idx) < 1e-8 for idx, value in enumerate(header)) and len(numeric_rows) > 1:
        numeric_rows = numeric_rows[1:]
    size = len(numeric_rows)
    processed_rows: List[List[float]] = []
    for idx, row in enumerate(numeric_rows):
        if len(row) == size + 1 and abs(row[0] - idx) < 1e-8:
            row = row[1:]
        if len(row) != size:
            raise ValueError("Adjacency matrix must be square")
        processed_rows.append(row)
    edges: List[Tuple[int, int, float]] = []
    for i in range(size):
        for j in range(i + 1, size):
            weight = processed_rows[i][j]
            if weight > 0:
                edges.append((i, j, float(weight)))
    return Graph(size, edges)


FORMAT_PARSERS = {
    "json": _parse_json_graph,
    "tsplib": _parse_tsplib,
    "snap": _parse_snap,
    "edge_list": _parse_edge_list,
    "csv": _parse_edge_list,
    "matrix_market": _parse_matrix_market,
    "gzip": _parse_gzip,
    "adj_matrix": _parse_adjacency_matrix,
}


def load_graph_file(path: Path) -> Graph:
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = _detect_format(path)
    parser = FORMAT_PARSERS.get(fmt)
    if not parser:
        raise ValueError(f"No parser registered for format {fmt}")
    return parser(path)


def load_graph_directory(directory: Path) -> Dict[str, Graph]:
    graphs: Dict[str, Graph] = {}
    if not directory.exists():
        return graphs
    for file in directory.iterdir():
        if not file.is_file():
            continue
        if file.stem.lower() in {"readme", "license"}:
            continue
        if file.suffix.lower() not in {".json", ".txt", ".dat", ".gz", ".mtx"}:
            continue
        try:
            graphs[file.stem] = load_graph_file(file)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[datasets] Skipping {file}: {exc}")
    return graphs


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"[datasets] Downloading {url} -> {dest}")
    urlretrieve(url, dest)  # nosec - trusted educational datasets
    return dest


def download_and_extract(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / Path(url).name
    download_file(url, archive_path)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(dest_dir)
    elif archive_path.suffix in {".tar", ".gz", ".tgz"}:
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(dest_dir)
    return dest_dir


def load_mendeley_graphs() -> Dict[str, Graph]:
    result: Dict[str, Graph] = {}
    # The dataset hosts multiple text files; users may need credentials.
    mendeley_dir = DATA_CACHE_DIR / "mendeley"
    if not mendeley_dir.exists():
        print("[datasets] Please download the Mendeley dataset manually due to access restrictions.")
        return result
    for file in mendeley_dir.glob("**/*"):
        if file.is_file() and file.suffix.lower() in {".txt", ".json", ".dat"}:
            try:
                result[file.stem] = load_graph_file(file)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[datasets] Skipping {file}: {exc}")
    return result


def load_snap_graph(name: str, filename: str) -> Graph:
    dest_dir = DATA_CACHE_DIR / "snap"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / filename
    download_file(f"{SNAP_BASE_URL}/{filename}", dest)
    if dest.suffix == ".gz":
        return _parse_gzip(dest)
    return load_graph_file(dest)


def load_brunel_graphs() -> Dict[str, Graph]:
    brunel_dir = DATA_CACHE_DIR / "brunel"
    brunel_dir.mkdir(exist_ok=True)
    graphs: Dict[str, Graph] = {}
    for file in brunel_dir.glob("**/*.txt"):
        try:
            graphs[file.stem] = load_graph_file(file)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[datasets] Skipping {file}: {exc}")
    return graphs


def load_default_snap_graphs() -> Dict[str, Graph]:
    graphs: Dict[str, Graph] = {}
    for name, filename in SNAP_DEFAULT_GRAPHS.items():
        try:
            graphs[name] = load_snap_graph(name, filename)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[datasets] Could not load SNAP graph {name}: {exc}")
    if not graphs:
        print("[datasets] No SNAP graphs loaded; please provide explicit filenames.")
    return graphs


def load_teacher_graphs() -> Dict[str, Graph]:
    return load_graph_directory(TEACHER_GRAPH_DIR)


def load_project_graphs() -> Dict[str, Graph]:
    return load_graph_directory(LOCAL_GRAPH_DIR)


def load_all_graphs(include_external: bool = True) -> Dict[str, Graph]:
    graphs = {}
    graphs.update(load_project_graphs())
    graphs.update(load_teacher_graphs())
    if include_external:
        graphs.update(load_mendeley_graphs())
        graphs.update(load_brunel_graphs())
    return graphs


def guess_format_and_load(path: Path) -> Graph:
    return load_graph_file(path)


def load_graphs_from_sources(sources: Iterable[str]) -> Dict[str, Graph]:
    graphs: Dict[str, Graph] = {}
    for source in sources:
        if source == "project":
            graphs.update(load_project_graphs())
        elif source == "teacher":
            graphs.update(load_teacher_graphs())
        elif source == "mendeley":
            graphs.update(load_mendeley_graphs())
        elif source == "snap":
            graphs.update(load_default_snap_graphs())
        elif source == "brunel":
            graphs.update(load_brunel_graphs())
    return graphs

__all__ = [
    "GraphRecord",
    "load_graph_file",
    "load_graph_directory",
    "load_project_graphs",
    "load_teacher_graphs",
    "load_mendeley_graphs",
    "load_brunel_graphs",
    "load_snap_graph",
    "load_default_snap_graphs",
    "load_all_graphs",
    "guess_format_and_load",
    "load_graphs_from_sources",
]
