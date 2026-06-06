"""Generate rustworkx 0.17.1 topological-sort fixtures for the managed
adjacency builder's parity test (`tests/adjacency_build.rs::fuzz_*`).

Run under DDR's venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_toposort_fixtures.py \
        > ~/projects/ddrs/tests/fixtures/toposort_fuzz.jsonl

Each line is a dendritic MERIT-like network plus the engine's topological order:

    {"records": [[comid, up1, up2, up3, up4], ...],   # raw FlowpathRecord fields
     "order":   [comid, ...]}                          # rx.topological_sort result

`records` feeds `build_conus_adjacency` on the Rust side verbatim (up==0 means
"no upstream"); `order` is produced by replaying the engine's exact
`build_upstream_dict -> build_graph -> topological_sort` pipeline so the Rust
build's `order` must match element-for-element.

Dendritic invariant: every node has at most one successor (one downstream),
matching real MERIT — this is the only shape the builder accepts.
"""

import json
import random
import sys

import rustworkx as rx


def engine_order(records):
    """Replay build_upstream_dict -> build_graph -> topological_sort."""
    upstream = {}
    for comid, *ups in records:
        for up in ups:
            if up > 0:
                upstream.setdefault(comid, []).append(up)
    for k in upstream:
        upstream[k] = sorted(set(upstream[k]))

    g = rx.PyDiGraph(check_cycle=False)
    node_idx = {}
    for to_comid in sorted(upstream.keys()):
        if to_comid not in node_idx:
            node_idx[to_comid] = g.add_node(to_comid)
        for from_comid in upstream[to_comid]:
            if from_comid not in node_idx:
                node_idx[from_comid] = g.add_node(from_comid)
    for to_comid, froms in upstream.items():
        for from_comid in froms:
            g.add_edge(node_idx[from_comid], node_idx[to_comid], None)

    ts = rx.topological_sort(g)
    connected = [g.get_node_data(i) for i in ts]

    all_comids = {r[0] for r in records}
    isolated = sorted(all_comids - set(connected))
    return connected + isolated


def random_dendritic(rng, n):
    """A random dendritic forest: each node may flow to one later-ranked node.

    Returns a list of [comid, up1, up2, up3, up4] rows (zero-padded). Confluences
    arise naturally (a node can have several upstreams) but every node has <= 1
    downstream, so the graph is dendritic.
    """
    comids = rng.sample(range(1, 50_000_000), n)
    rank = list(range(n))
    rng.shuffle(rank)  # rank[i] = topo rank of node i; downstream = higher rank
    pos = {i: rank[i] for i in range(n)}

    downstream_of = {}  # node i -> node j it flows into
    for i in range(n):
        cands = [j for j in range(n) if pos[j] > pos[i]]
        if cands and rng.random() < 0.85:
            downstream_of[i] = rng.choice(cands)

    # Invert to upstream lists, capped at 4 (MERIT up1..up4).
    upstreams = {j: [] for j in range(n)}
    for i, j in downstream_of.items():
        upstreams[j].append(i)

    records = []
    for i in range(n):
        ups = upstreams[i][:4]  # cap at 4; drop overflow edges
        ups = [comids[u] for u in ups]
        ups += [0] * (4 - len(ups))
        records.append([comids[i]] + ups)
    return records


def main():
    n_graphs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260606
    rng = random.Random(seed)
    out = sys.stdout
    for _ in range(n_graphs):
        n = rng.randint(1, 80)
        records = random_dendritic(rng, n)
        # Skip the rare case where capping at 4 upstreams orphaned an edge such
        # that a node still has >1 downstream (cannot happen here — downstream is
        # unique by construction; capping only drops upstream edges).
        order = engine_order(records)
        out.write(json.dumps({"records": records, "order": order}) + "\n")


if __name__ == "__main__":
    main()
