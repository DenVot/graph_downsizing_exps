#!/usr/bin/env python3
"""Candidate graphs with moderate density variation; construction and preview only."""

from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx


SPECS = (
    {"id": "candidate100", "background": 36, "background_seed": 101,
     "blocks": ((32, 0.55, 7), (32, 0.52, 9))},
    {"id": "candidate140", "background": 44, "background_seed": 102,
     "blocks": ((32, 0.55, 7), (32, 0.50, 21), (32, 0.58, 5))},
    {"id": "candidate180", "background": 52, "background_seed": 103,
     "blocks": ((32, 0.55, 7), (32, 0.50, 21), (32, 0.58, 5), (32, 0.53, 31))},
)
COLORS = ("#2479b5", "#d2763e", "#52a471", "#9462b6")


def build_graph(spec: dict):
    graph = nx.Graph()
    blocks = []
    offset = 0
    for index, (size, probability, seed) in enumerate(spec["blocks"]):
        block = nx.gnp_random_graph(size, probability, seed=seed)
        if not nx.is_connected(block):
            raise AssertionError("A planted region is disconnected")
        graph.add_nodes_from(range(offset, offset + size))
        graph.add_edges_from((offset + u, offset + v) for u, v in block.edges)
        blocks.append({"index": index, "size": size, "probability": probability,
                       "seed": seed, "offset": offset, "edges": block.number_of_edges()})
        offset += size
    background = list(range(offset, offset + spec["background"]))
    sparse_graph = nx.random_regular_graph(8, len(background), seed=spec["background_seed"])
    graph.add_edges_from((offset + u, offset + v) for u, v in sparse_graph.edges)
    # Every exterior vertex has one independent attachment to a planted region.
    # This raises exterior degree to 9 while making the interface distributed.
    for index, vertex in enumerate(background):
        block = blocks[index % len(blocks)]
        local = (index * 17 + spec["background_seed"]) % block["size"]
        graph.add_edge(vertex, block["offset"] + local)
    if not nx.is_connected(graph):
        raise AssertionError("The candidate graph is disconnected")
    assert all(graph.degree(vertex) == 9 for vertex in background)
    digest = hashlib.sha256("\n".join(f"{u} {v}" for u, v in sorted(tuple(sorted(edge)) for edge in graph.edges)).encode()).hexdigest()
    return graph, blocks, background, digest


def render() -> str:
    width, panel_height = 1500, 690
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel_height * len(SPECS)}" viewBox="0 0 {width} {panel_height * len(SPECS)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.title{font-size:28px;font-weight:bold}.note{font-size:17px;fill:#475569}.label{font-size:17px;font-weight:bold}</style>',
    ]
    for panel, spec in enumerate(SPECS):
        graph, blocks, background, _ = build_graph(spec)
        top = panel * panel_height
        parts.append(f'<rect x="10" y="{top + 10}" width="1480" height="670" rx="12" fill="#fafcff" stroke="#dce3eb"/>')
        parts.append(f'<text x="35" y="{top + 49}" class="title">{spec["id"]}: {len(graph)} вершин, {graph.number_of_edges()} рёбер</text>')
        parts.append(f'<text x="35" y="{top + 78}" class="note">Цветные области: 32 вершины, плотность 0,50–0,58; серый фон: случайный 8-регулярный граф + по одной связи с областью.</text>')
        positions = nx.spring_layout(graph, seed=spec["background_seed"], iterations=500, k=0.21)
        xs = [point[0] for point in positions.values()]
        ys = [point[1] for point in positions.values()]
        xlow, xhigh, ylow, yhigh = min(xs), max(xs), min(ys), max(ys)
        coords = {vertex: (95 + 1310 * (x - xlow) / (xhigh - xlow), top + 115 + 470 * (y - ylow) / (yhigh - ylow))
                  for vertex, (x, y) in positions.items()}
        region = {vertex: block["index"] for block in blocks
                  for vertex in range(block["offset"], block["offset"] + block["size"])}
        for u, v in graph.edges:
            ax, ay = coords[u]
            bx, by = coords[v]
            if u in region and region.get(u) == region.get(v):
                color, opacity, weight = COLORS[region[u]], "0.17", "1"
            elif u not in region and v not in region:
                color, opacity, weight = "#858f9c", "0.31", "1.2"
            else:
                color, opacity, weight = "#5c6470", "0.40", "1.1"
            parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="{weight}"/>')
        for vertex in graph.nodes:
            x, y = coords[vertex]
            color = COLORS[region[vertex]] if vertex in region else "#7b8490"
            radius = 4.4 if vertex in region else 3.9
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
        caption = "; ".join(f"область {block['index'] + 1}: {block['edges']} рёбер" for block in blocks)
        parts.append(f'<text x="750" y="{top + 632}" text-anchor="middle" class="label">{caption}; фон: {len(background) * 4} рёбер</text>')
        parts.append(f'<text x="750" y="{top + 657}" text-anchor="middle" class="note">Каждая серая вершина имеет степень 9 во всём графе</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    path = Path(__file__).with_name("downsizing_blended_preview.svg")
    path.write_text(render(), encoding="utf-8")
    print(path)
