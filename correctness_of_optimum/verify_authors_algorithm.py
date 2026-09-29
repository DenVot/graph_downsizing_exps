#!/usr/bin/env python3
"""Independent small-graph checks for the local paper algorithm."""

from __future__ import annotations

from itertools import combinations
import random

import networkx as nx

from authors_algorithm import (
    Inundation_downsizing,
    best_of_five_full_runs,
    enumerate_maximal_k_vc,
    expand_single_node,
)


def brute_maximal_k_vc(graph: nx.Graph, k: int) -> set[frozenset[int]]:
    vertices = list(graph)
    valid = [
        frozenset(subset)
        for size in range(k + 1, len(vertices) + 1)
        for subset in combinations(vertices, size)
        if nx.node_connectivity(graph.subgraph(subset)) >= k
    ]
    return {subset for subset in valid if not any(subset < other for other in valid)}


def main() -> None:
    expansion_graph = nx.Graph([(0, 1), (1, 2), (0, 2),
                                (3, 0), (3, 1), (4, 2), (4, 3)])
    expanded = expand_single_node(expansion_graph, expansion_graph.subgraph((0, 1, 2)), 5, 2)
    assert expanded is not None and len(expanded) == 5
    assert nx.node_connectivity(expanded) >= 2

    rng = random.Random(792)
    enumeration_checks = 0
    for order in range(3, 8):
        for _ in range(60):
            graph = nx.gnp_random_graph(order, rng.random(), seed=rng.randrange(10**9))
            for k in range(1, order):
                result = {frozenset(core) for core in enumerate_maximal_k_vc(graph, k)}
                expected = brute_maximal_k_vc(graph, k)
                assert result == expected, (order, k, list(graph.edges), result, expected)
                enumeration_checks += 1

    full_runs = 0
    for order in range(4, 10):
        for _ in range(20):
            graph = nx.gnp_random_graph(order, 0.45, seed=rng.randrange(10**9))
            if not nx.is_connected(graph):
                continue
            size = rng.randrange(2, order)
            for retries in (1, 5):
                stats: dict[str, int] = {}
                result = Inundation_downsizing(graph, size, rng, retries, stats)
                assert len(result) == size and nx.is_connected(result)
                assert nx.node_connectivity(result) >= stats["reported_k"]
                full_runs += 1
            stats: dict[str, int] = {}
            result = best_of_five_full_runs(graph, size, rng.randrange(10**9), stats)
            assert len(result) == size and stats["full_runs"] == 5
            assert nx.node_connectivity(result) >= stats["reported_k"]

    print(f"Exact enumerator vs. all subsets: {enumeration_checks} checks passed")
    print(f"Complete algorithm: {full_runs} checks passed")


if __name__ == "__main__":
    main()
