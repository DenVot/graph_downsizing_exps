#!/usr/bin/env python3
"""Exact comparison on four connected dense Erdos-Renyi graphs.

For every (graph, r) pair the script enumerates every r-vertex subset, runs
the complete paper algorithm five times, and compares both answers. Only the
complete Downsizing algorithm is measured; there are no standalone Problem 2
trials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx

from authors_algorithm import best_of_five_full_runs


CASES = (
    {"id": "G30_p55_s6", "vertices": 30, "probability": 0.55, "seed": 6,
     "expected_edges": 231, "sizes": tuple(range(10, 30, 2))},
    {"id": "G30_p55_s12", "vertices": 30, "probability": 0.55, "seed": 12,
     "expected_edges": 252, "sizes": tuple(range(10, 30, 2))},
    {"id": "G32_p55_s1", "vertices": 32, "probability": 0.55, "seed": 1,
     "expected_edges": 275, "sizes": tuple(range(10, 32, 2))},
    {"id": "G32_p55_s7", "vertices": 32, "probability": 0.55, "seed": 7,
     "expected_edges": 291, "sizes": tuple(range(10, 32, 2))},
)


def build_graph(spec: dict) -> nx.Graph:
    """Recreate one deterministic G(n, 0.55) test graph."""
    graph = nx.gnp_random_graph(
        spec["vertices"], spec["probability"], seed=spec["seed"]
    )
    if not nx.is_connected(graph):
        raise AssertionError(f"{spec['id']} is unexpectedly disconnected")
    if graph.number_of_edges() != spec["expected_edges"]:
        raise AssertionError(
            f"{spec['id']} has {graph.number_of_edges()} edges; "
            f"expected {spec['expected_edges']}"
        )
    return graph


def task_path(output: Path, kind: str, case_id: str, size: int) -> Path:
    """Return the checkpoint path for one independent task."""
    return output / "tasks" / f"{kind}_{case_id}_r{size}.json"


def write_task(path: Path, payload: dict) -> None:
    """Atomically write a checkpoint so interrupted writes are not reused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def graph_hash(graph: nx.Graph) -> str:
    """Hash the sorted edge list to identify the exact generated graph."""
    edges = "\n".join(
        f"{u} {v}" for u, v in sorted(tuple(sorted(edge)) for edge in graph.edges)
    )
    return hashlib.sha256(edges.encode()).hexdigest()


def exact_task(spec: dict, size: int, output_name: str) -> str:
    """Find the exact best r-subgraph by enumerating every vertex subset.

    Minimum degree is a safe upper bound on vertex connectivity, so NetworkX
    is called only when that bound can improve the current exact optimum.
    """
    output = Path(output_name)
    graph = build_graph(spec)
    task_id = f"oracle_{spec['id']}_r{size}"
    total = math.comb(len(graph), size)
    adjacency_masks = {
        vertex: sum(1 << neighbor for neighbor in graph.neighbors(vertex))
        for vertex in graph.nodes
    }
    optimum = -1
    witness: tuple[int, ...] = ()
    connectivity_checks = 0
    started = last_progress = time.monotonic()

    for index, vertices in enumerate(itertools.combinations(graph.nodes, size), 1):
        subset_mask = sum(1 << vertex for vertex in vertices)
        degree_bound = min(
            (adjacency_masks[vertex] & subset_mask).bit_count()
            for vertex in vertices
        )
        if degree_bound > optimum:
            value = nx.node_connectivity(graph.subgraph(vertices))
            connectivity_checks += 1
            if value > optimum:
                optimum = value
                witness = vertices

        now = time.monotonic()
        if now - last_progress >= 30:
            print(
                f"PROGRESS {task_id}: {index:,}/{total:,} "
                f"({index / total:.1%}), best={optimum}, "
                f"connectivity_checks={connectivity_checks:,}",
                flush=True,
            )
            last_progress = now

    payload = {
        "task": task_id, "kind": "oracle", "case": spec["id"],
        "vertices": len(graph), "edges": graph.number_of_edges(), "size": size,
        "optimum": optimum, "witness": list(witness),
        "subsets_visited": total, "connectivity_checks": connectivity_checks,
        "seconds": round(time.monotonic() - started, 3),
        "graph_hash": graph_hash(graph),
    }
    write_task(task_path(output, "oracle", spec["id"], size), payload)
    print(
        f"DONE {task_id}: optimum={optimum}, seconds={payload['seconds']}",
        flush=True,
    )
    return task_id


def algorithm_task(spec: dict, size: int, output_name: str) -> str:
    """Run five complete Algorithm 1 executions and keep the best result."""
    output = Path(output_name)
    graph = build_graph(spec)
    case_index = next(
        index for index, item in enumerate(CASES) if item["id"] == spec["id"]
    )
    random_seed = 2_000_000 + case_index * 1000 + size * 10
    stats: dict[str, int] = {}
    started = time.monotonic()
    result = best_of_five_full_runs(graph, size, random_seed, stats=stats)
    seconds = time.monotonic() - started
    vertices = sorted(result)
    if len(vertices) != size:
        raise AssertionError(
            f"algorithm_{spec['id']}_r{size} returned {len(vertices)} vertices"
        )
    actual_k = nx.node_connectivity(graph.subgraph(vertices))
    task_id = f"algorithm_{spec['id']}_r{size}"
    payload = {
        "task": task_id, "kind": "algorithm", "case": spec["id"],
        "vertices_count": len(graph), "edges": graph.number_of_edges(), "size": size,
        "actual_k": actual_k, "reported_k": stats["reported_k"],
        "result_vertices": vertices, "full_runs": stats["full_runs"],
        "pdownsize_calls": stats["pdownsize_calls"], "random_seed": random_seed,
        "seconds": round(seconds, 3), "graph_hash": graph_hash(graph),
    }
    write_task(task_path(output, "algorithm", spec["id"], size), payload)
    print(
        f"DONE {task_id}: actual={actual_k}, reported={stats['reported_k']}, "
        f"calls={stats['pdownsize_calls']}, seconds={seconds:.3f}",
        flush=True,
    )
    return task_id


def selected_instances(
    only_cases: list[str] | None, only_sizes: list[int] | None
):
    """Yield the requested subset of the 42 predefined instances."""
    known_cases = {spec["id"] for spec in CASES}
    unknown_cases = set(only_cases or ()) - known_cases
    if unknown_cases:
        raise ValueError(f"unknown cases: {sorted(unknown_cases)}")
    for spec in CASES:
        if only_cases and spec["id"] not in only_cases:
            continue
        for size in spec["sizes"]:
            if only_sizes and size not in only_sizes:
                continue
            yield spec, size


def run(
    output: Path,
    workers: int,
    only_cases: list[str] | None,
    only_sizes: list[int] | None,
) -> None:
    """Submit all missing oracle and algorithm tasks to a process pool."""
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for spec, size in selected_instances(only_cases, only_sizes):
            oracle_path = task_path(output, "oracle", spec["id"], size)
            if not oracle_path.exists():
                pending.append(executor.submit(exact_task, spec, size, str(output)))
            algorithm_path = task_path(output, "algorithm", spec["id"], size)
            if not algorithm_path.exists():
                pending.append(
                    executor.submit(algorithm_task, spec, size, str(output))
                )

        print(f"SUBMITTED {len(pending)} tasks to {workers} processes", flush=True)
        for number, future in enumerate(as_completed(pending), 1):
            task_id = future.result()
            print(f"CHECKPOINT {number}/{len(pending)} {task_id}", flush=True)


def load_completed_rows(
    output: Path,
    only_cases: list[str] | None,
    only_sizes: list[int] | None,
) -> list[dict]:
    """Load and independently validate every completed task pair."""
    rows = []
    for spec, size in selected_instances(only_cases, only_sizes):
        oracle_path = task_path(output, "oracle", spec["id"], size)
        algorithm_path = task_path(output, "algorithm", spec["id"], size)
        if not oracle_path.exists() or not algorithm_path.exists():
            continue

        graph = build_graph(spec)
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        algorithm = json.loads(algorithm_path.read_text(encoding="utf-8"))
        digest = graph_hash(graph)
        if oracle["graph_hash"] != digest or algorithm["graph_hash"] != digest:
            raise AssertionError(f"stale checkpoint for {spec['id']} r={size}")
        if nx.node_connectivity(graph.subgraph(oracle["witness"])) != oracle["optimum"]:
            raise AssertionError(f"invalid oracle witness for {spec['id']} r={size}")
        if len(algorithm["result_vertices"]) != size:
            raise AssertionError(f"invalid algorithm size for {spec['id']} r={size}")
        actual_k = nx.node_connectivity(graph.subgraph(algorithm["result_vertices"]))
        if actual_k != algorithm["actual_k"]:
            raise AssertionError(f"invalid algorithm connectivity for {spec['id']} r={size}")
        if actual_k > oracle["optimum"]:
            raise AssertionError(f"algorithm exceeds optimum for {spec['id']} r={size}")

        rows.append({
            "case": spec["id"], "vertices": len(graph),
            "edges": graph.number_of_edges(), "size": size,
            "optimum": oracle["optimum"],
            "optimum_witness": " ".join(map(str, oracle["witness"])),
            "actual_k": actual_k, "reported_k": algorithm["reported_k"],
            "actual_vertices": " ".join(map(str, algorithm["result_vertices"])),
            "full_runs": algorithm["full_runs"],
            "pdownsize_calls": algorithm["pdownsize_calls"],
            "random_seed": algorithm["random_seed"],
            "oracle_seconds": oracle["seconds"],
            "algorithm_seconds": algorithm["seconds"],
            "subsets_visited": oracle["subsets_visited"],
            "connectivity_checks": oracle["connectivity_checks"],
            "graph_hash": digest,
        })
    return rows


def write_results(rows: list[dict], output: Path) -> None:
    """Write the combined CSV for all completed pairs."""
    if not rows:
        raise RuntimeError("no completed oracle/algorithm pairs to merge")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], path: Path) -> None:
    """Draw exact and algorithm connectivity curves as SVG."""
    case_ids = list(dict.fromkeys(row["case"] for row in rows))
    panel_width, panel_height = 510, 330
    width = 1060
    height = 90 + math.ceil(len(case_ids) / 2) * panel_height + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#20242a}.title{font-size:21px;font-weight:bold}.label{font-size:15px}.tick{font-size:12px;fill:#5b6470}.grid{stroke:#dce2e8;stroke-width:1}.axis{stroke:#6e7885;stroke-width:1.2}</style>',
        '<text x="28" y="31" class="title">Dense random graphs: exact optimum vs. Algorithm 1</text>',
        '<line x1="28" y1="56" x2="62" y2="56" stroke="#1675b8" stroke-width="3"/><text x="70" y="61" class="label">Exact optimum</text>',
        '<line x1="240" y1="56" x2="274" y2="56" stroke="#d44b34" stroke-width="3"/><text x="282" y="61" class="label">Best of five full runs</text>',
    ]
    for index, case_id in enumerate(case_ids):
        group = sorted(
            (row for row in rows if row["case"] == case_id),
            key=lambda row: row["size"],
        )
        left = 25 + index % 2 * panel_width
        top = 90 + index // 2 * panel_height
        x0, x1 = left + 54, left + 470
        y0, y1 = top + 242, top + 35
        minimum_size, maximum_size = group[0]["size"], group[-1]["size"]
        maximum_k = max(4, *(row["optimum"] for row in group))

        def coordinates(row: dict, field: str) -> tuple[float, float]:
            x = x0 + (row["size"] - minimum_size) * (x1 - x0) / max(
                1, maximum_size - minimum_size
            )
            y = y0 - row[field] * (y0 - y1) / maximum_k
            return x, y

        parts.append(
            f'<text x="{left + 5}" y="{top + 14}" class="label">{case_id}: '
            f'n={group[0]["vertices"]}, m={group[0]["edges"]}</text>'
        )
        for value in range(maximum_k + 1):
            y = y0 - value * (y0 - y1) / maximum_k
            parts.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>'
            )
            if value % max(1, maximum_k // 7) == 0 or value == maximum_k:
                parts.append(
                    f'<text x="{x0 - 22}" y="{y + 4:.1f}" class="tick">{value}</text>'
                )
        parts.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>'
        )
        for row in group:
            x, _ = coordinates(row, "optimum")
            parts.append(
                f'<text x="{x - 7:.1f}" y="{y0 + 18}" class="tick">{row["size"]}</text>'
            )
        for field, color in (("optimum", "#1675b8"), ("actual_k", "#d44b34")):
            points = [coordinates(row, field) for row in group]
            encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(
                f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2.8"/>'
            )
            parts.extend(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
                for x, y in points
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def merge(
    output: Path,
    only_cases: list[str] | None,
    only_sizes: list[int] | None,
) -> None:
    """Validate checkpoints and rebuild CSV, SVG and text summary."""
    rows = load_completed_rows(output, only_cases, only_sizes)
    write_results(rows, output)
    write_plot(rows, output / "comparison.svg")
    misses = sum(row["actual_k"] < row["optimum"] for row in rows)
    expected = sum(1 for _ in selected_instances(only_cases, only_sizes))
    summary = (
        f"Completed instances: {len(rows)}/{expected}\n"
        f"Below exact optimum: {misses}\n"
        f"Invalid output sizes: 0\n"
        f"Standalone Problem 2 trials: 0\n"
    )
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("downsizing_dense_results"),
    )
    parser.add_argument(
        "--workers", type=int,
        default=min(11, max(1, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--only-case", action="append", help="Run one graph; may repeat")
    parser.add_argument("--only-size", action="append", type=int, help="Run one r; may repeat")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.merge_only:
        run(args.output, args.workers, args.only_case, args.only_size)
    merge(args.output, args.only_case, args.only_size)


if __name__ == "__main__":
    main()
