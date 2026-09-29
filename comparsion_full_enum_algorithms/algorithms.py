"""Exact fixed-order vertex-connectivity maximization, Python standard library.

Graph = tuple of frozensets, vertices numbered 0,...,n-1. Simple undirected graphs.
The flow primitive is Edmonds--Karp on a vertex-split network.
"""
from collections import deque
from dataclasses import dataclass, asdict
from itertools import combinations
from time import perf_counter


@dataclass
class Stats:
    subsets: int = 0
    flows: int = 0
    augmentations: int = 0
    cutoff_hits: int = 0
    tree_nodes: int = 0
    enumeration_calls: int = 0
    pdownsize_calls: int = 0


def graph_from_edges(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError('Expected a simple graph on 0,...,n-1')
        adj[u].add(v)
        adj[v].add(u)
    return tuple(map(frozenset, adj))


def induced(graph, vertices):
    index = {v: i for i, v in enumerate(vertices)}
    return tuple(frozenset(index[w] for w in graph[v] if w in index)
                 for v in vertices)


class FlowNetwork:
    """Build topology once per induced graph; reset residuals for every pair.

    Vertex arcs have capacity one; original-edge arcs capacity n. For adjacent
    terminals only their direct source->sink arc is reduced to one, so the
    length-one path is counted once (important for complete graphs).
    """
    def __init__(self, graph):
        self.n = n = len(graph)
        self.adj = [[] for _ in range(2*n)]
        self.to, self.capacity = [], []
        self.edge_ids = {}
        def add(u, v, capacity):
            idx = len(self.to)
            self.adj[u].append(idx)
            self.to.append(v)
            self.capacity.append(capacity)
            self.adj[v].append(idx+1)
            self.to.append(u)
            self.capacity.append(0)
            return idx
        for v in range(n):
            add(2*v, 2*v+1, 1)
        for u in range(n):
            for v in sorted(graph[u]):
                self.edge_ids[u, v] = add(2*u+1, 2*v, n)

    def flow(self, source, target, stats, cutoff=None):
        stats.flows += 1
        capacity = self.capacity.copy()
        direct = self.edge_ids.get((source, target))
        if direct is not None:
            capacity[direct] = 1
        start, finish = 2*source+1, 2*target
        total = 0
        while cutoff is None or total < cutoff:
            parent = [-1] * (2*self.n)
            parent[start] = -2
            queue = deque([start])
            while queue and parent[finish] == -1:
                u = queue.popleft()
                for edge in self.adj[u]:
                    v = self.to[edge]
                    if capacity[edge] and parent[v] == -1:
                        parent[v] = edge
                        queue.append(v)
                        if v == finish:
                            break
            if parent[finish] == -1:
                # For nonadjacent terminals a minimum cut consists of these
                # split-vertex arcs; original-edge capacity n cannot saturate.
                cut = frozenset(v for v in range(self.n)
                                if parent[2*v] != -1 and parent[2*v+1] == -1)
                return total, cut
            v = finish
            while v != start:
                edge = parent[v]
                capacity[edge] -= 1
                capacity[edge ^ 1] += 1
                v = self.to[edge ^ 1]
            total += 1
            stats.augmentations += 1
        stats.cutoff_hits += 1
        return total, None


def connectivity(graph, stats=None):
    """Literal min of full maximum flows over ALL unordered vertex pairs."""
    stats = stats if stats is not None else Stats()
    if len(graph) < 2:
        return 0
    network = FlowNetwork(graph)
    return min(network.flow(u, v, stats)[0]
               for u, v in combinations(range(len(graph)), 2))


def at_least_k(graph, k, stats):
    if k <= 0:
        return True
    if len(graph) <= k:
        return False
    network = FlowNetwork(graph)
    for u, v in combinations(range(len(graph)), 2):
        if network.flow(u, v, stats, cutoff=k)[0] < k:
            return False
    return True


def components(graph, vertices):
    remaining = set(vertices)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component, stack = {seed}, [seed]
        while stack:
            u = stack.pop()
            new = graph[u] & remaining
            remaining.difference_update(new)
            component.update(new)
            stack.extend(sorted(new, reverse=True))
        result.append(frozenset(component))
    return result


def maximal_k_connected(graph, vertices, k, stats):
    """Exact enumeration by k-core pruning and recursive separator splitting.

    If |S|<k separates H, any k-connected subgraph lies in C union S for one
    component C of H-S. Enumerate EVERY component; then remove duplicates and
    proper subsets. This is a simple exact enumerator, not Wen et al.'s engine.
    """
    found, visited = set(), set()
    def visit(vertices):
        stats.enumeration_calls += 1
        vertices = frozenset(vertices)
        while True:
            reduced = frozenset(v for v in vertices if len(graph[v] & vertices) >= k)
            if reduced == vertices:
                break
            vertices = reduced
        if len(vertices) <= k or vertices in visited:
            return
        visited.add(vertices)
        order = sorted(vertices)
        local = induced(graph, order)
        network = FlowNetwork(local)
        for u, v in combinations(range(len(order)), 2):
            if v in local[u]:
                continue  # A global minimum vertex separator has nonadjacent ends.
            value, cut = network.flow(u, v, stats, cutoff=k)
            if value < k:
                separator = frozenset(order[i] for i in cut)
                assert len(separator) == value
                pieces = components(graph, vertices - separator)
                assert len(pieces) >= 2
                for component in pieces:
                    visit(component | separator)
                return
        found.add(vertices)
    visit(vertices)
    return sorted((s for s in found if not any(s < t for t in found)),
                  key=lambda s: tuple(sorted(s)))


def pdownsize(graph, vertices, r, k, stats):
    stats.pdownsize_calls += 1
    for subset in combinations(sorted(vertices), r):
        stats.subsets += 1
        if at_least_k(induced(graph, subset), k, stats):
            return subset
    return None


def exhaustive(graph, r):
    _validate(graph, r)
    stats = Stats()
    start = perf_counter()
    best, witness = -1, None
    for subset in combinations(range(len(graph)), r):
        stats.subsets += 1
        k = connectivity(induced(graph, subset), stats)
        if k > best:
            best, witness = k, subset
    return dict(k=best, witness=witness, seconds=perf_counter()-start,
                stats=asdict(stats))


def _validate(graph, r):
    if not 1 <= r <= len(graph):
        raise ValueError('Require 1 <= r <= n')


def inundation(graph, r):
    """Algorithm 1 with exact threshold PDownsize and exact enumeration.

    Store hierarchy levels explicitly. A node at level k may have connectivity
    >k; PDownsize tests >=k. Descending search makes the first success optimal.
    Levels above r-1 cannot contain a feasible r-vertex solution and are skipped.
    Disconnected inputs and r=1 are handled beyond the paper's assumptions.
    """
    _validate(graph, r)
    stats = Stats()
    start = perf_counter()
    roots = [c for c in components(graph, range(len(graph))) if len(c) >= r]
    nodes = [(1, c) for c in roots] if r > 1 else []
    frontier = nodes.copy()
    while frontier:
        level, vertices = frontier.pop()
        if level >= r-1:
            continue
        for child in maximal_k_connected(graph, vertices, level+1, stats):
            if len(child) >= r:
                nodes.append((level+1, child))
                frontier.append((level+1, child))
    stats.tree_nodes = len(nodes)
    build_seconds = perf_counter()-start
    witness, best = tuple(range(r)), 0
    for k, vertices in sorted(nodes, key=lambda x: (-x[0], tuple(sorted(x[1])))):
        if k == 1:
            # Connected prefix of a deterministic traversal (Algorithm 1 line 22).
            reached, queue = {min(vertices)}, deque([min(vertices)])
            ordered = []
            while queue and len(ordered) < r:
                u = queue.popleft()
                ordered.append(u)
                for v in sorted((graph[u] & vertices) - reached):
                    reached.add(v)
                    queue.append(v)
            candidate = tuple(ordered)
        else:
            candidate = pdownsize(graph, vertices, r, k, stats)
        if candidate is not None:
            witness, best = candidate, k
            break
    seconds = perf_counter()-start
    return dict(k=best, witness=witness, seconds=seconds,
                build_seconds=build_seconds, search_seconds=seconds-build_seconds,
                stats=asdict(stats))
