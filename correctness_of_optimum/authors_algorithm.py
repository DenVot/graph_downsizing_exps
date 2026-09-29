"""Self-contained implementation of Algorithms 1 and 2 from Liu et al. (2026).

The paper treats enumeration of maximal k-VC subgraphs as an external
subroutine. Here it is implemented exactly by recursive vertex separators.
This is slower than the specialized enumerators cited in the paper.
"""

from __future__ import annotations

from collections import deque
import random

import networkx as nx


def enumerate_maximal_k_vc(graph: nx.Graph, k: int) -> list[nx.Graph]:
    """List all maximal induced subgraphs with vertex connectivity >= k.

    Algorithm 1 explicitly calls an *external* enumerator. This exact recursive
    separator implementation fills that role; it is not the Wen et al. (2019)
    optimized enumerator and can be slow on large graphs.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if len(graph) <= k:
        return []
    if k == 1:
        return [graph.subgraph(vertices).copy()
                for vertices in nx.connected_components(graph) if len(vertices) >= 2]

    order = {vertex: index for index, vertex in enumerate(graph)}
    pending = [frozenset(graph)]
    visited: set[frozenset] = set()
    candidates: set[frozenset] = set()

    while pending:
        vertices = pending.pop()
        if len(vertices) <= k or vertices in visited:
            continue
        visited.add(vertices)
        subgraph = graph.subgraph(vertices)
        if nx.is_connected(subgraph):
            cut = set(nx.minimum_node_cut(subgraph))
            if len(cut) >= k:
                candidates.add(vertices)
                continue
        else:
            cut = set()
        parts = list(nx.connected_components(subgraph.subgraph(vertices - cut)))
        if len(parts) < 2:
            raise RuntimeError("separator did not split a non-k-connected graph")
        for part in parts:
            child = frozenset(part | cut)
            if len(child) < len(vertices):
                pending.append(child)

    maximal = [vertices for vertices in candidates
               if not any(vertices < other for other in candidates)]
    maximal.sort(key=lambda vertices: (-len(vertices),
                                      tuple(sorted(order[vertex] for vertex in vertices))))
    return [graph.subgraph(vertices).copy() for vertices in maximal]


def expand_single_node(graph: nx.Graph, core: nx.Graph, r: int, k: int) -> nx.Graph | None:
    """Algorithm 2, lines 7-12: add any vertex with >= k neighbors in core."""
    chosen = set(core)
    neighbor_counts = {vertex: 0 for vertex in graph if vertex not in chosen}
    for vertex in chosen:
        for neighbor in graph.neighbors(vertex):
            if neighbor in neighbor_counts:
                neighbor_counts[neighbor] += 1
    employable = deque(vertex for vertex in graph
                       if vertex in neighbor_counts and neighbor_counts[vertex] >= k)
    while len(chosen) < r:
        if not employable:
            return None
        vertex = employable.popleft()
        chosen.add(vertex)
        del neighbor_counts[vertex]
        for neighbor in graph.neighbors(vertex):
            if neighbor in neighbor_counts:
                neighbor_counts[neighbor] += 1
                if neighbor_counts[neighbor] == k:
                    employable.append(neighbor)
    return graph.subgraph(chosen).copy()


def PDownsize(graph: nx.Graph, r: int, k: int,
              rng: random.Random | None = None) -> nx.Graph | None:
    """Algorithm 2: sample, enumerate maximal k-VC cores, expand the largest."""
    if not 1 <= r <= len(graph):
        raise ValueError("r must be between 1 and the graph order")
    if k < 1:
        raise ValueError("k must be positive")
    rng = rng if rng is not None else random.Random()
    sampled_vertices = rng.sample(list(graph), r)
    sampled = graph.subgraph(sampled_vertices).copy()
    cores = enumerate_maximal_k_vc(sampled, k)
    if any(len(core) == r for core in cores):
        return sampled
    if not cores:
        return None
    largest = max(cores, key=len)
    return expand_single_node(graph, largest, r, k)


def connected_subgraph_bfs(graph: nx.Graph, r: int) -> nx.Graph:
    """Algorithm 1, line 22: connected induced r-vertex fallback."""
    first = next(iter(graph))
    queue = deque([first])
    selected = {first}
    while queue and len(selected) < r:
        vertex = queue.popleft()
        for neighbor in graph.neighbors(vertex):
            if neighbor not in selected:
                selected.add(neighbor)
                queue.append(neighbor)
                if len(selected) == r:
                    break
    if len(selected) != r:
        raise RuntimeError("graph must be connected and contain at least r vertices")
    return graph.subgraph(selected).copy()


def Inundation_downsizing(graph: nx.Graph, r: int,
                          rng: random.Random | None = None,
                          retries_per_node: int = 1,
                          stats: dict[str, int] | None = None) -> nx.Graph:
    """Algorithm 1; retries_per_node=1 follows its pseudocode literally.

    The published main.py instead uses retries_per_node=5 and returns on the
    first success at a given tree node. This differs from the paper's separate
    experimental protocol of five full runs and choosing the best result.
    """
    if not 1 <= r < len(graph):
        raise ValueError("the paper assumes 0 < tau < |V|, so 1 <= r < |V|")
    if not nx.is_connected(graph):
        raise ValueError("the paper assumes a connected input graph")
    if retries_per_node < 1:
        raise ValueError("retries_per_node must be positive")
    rng = rng if rng is not None else random.Random()
    if stats is not None:
        stats["pdownsize_calls"] = 0

    root = graph.copy()
    levels: dict[int, list[nx.Graph]] = {1: [root]}
    frontier = [root]
    level = 1
    while frontier:
        next_level: list[nx.Graph] = []
        seen: set[frozenset] = set()
        for parent in frontier:
            for child in enumerate_maximal_k_vc(parent, level + 1):
                vertices = frozenset(child)
                if len(vertices) >= r and vertices not in seen:
                    next_level.append(child)
                    seen.add(vertices)
        if not next_level:
            break
        level += 1
        levels[level] = next_level
        frontier = next_level

    for current_level in range(level, 1, -1):
        for node in levels[current_level]:
            for _ in range(retries_per_node):
                if stats is not None:
                    stats["pdownsize_calls"] += 1
                result = PDownsize(node, r, current_level, rng)
                if result is not None and len(result) == r:
                    if stats is not None:
                        stats["reported_k"] = current_level
                    return result
    if stats is not None:
        stats["reported_k"] = 1
    return connected_subgraph_bfs(graph, r)


def best_of_five_full_runs(graph: nx.Graph, r: int,
                           seed: int | None = None,
                           stats: dict[str, int] | None = None) -> nx.Graph:
    """Section 5: five complete Algorithm 1 runs, keep the best connectivity."""
    rng = random.Random(seed)
    runs: list[tuple[nx.Graph, dict[str, int]]] = []
    for _ in range(5):
        run_stats: dict[str, int] = {}
        result = Inundation_downsizing(graph, r, rng=rng, stats=run_stats)
        runs.append((result, run_stats))
    best, best_stats = max(runs, key=lambda run: nx.node_connectivity(run[0]))
    if stats is not None:
        stats["pdownsize_calls"] = sum(item["pdownsize_calls"] for _, item in runs)
        stats["reported_k"] = best_stats["reported_k"]
        stats["full_runs"] = 5
    return best
