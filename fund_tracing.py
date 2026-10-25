"""
Phase 4, Step 4: Fund tracing. Find the rings and the mule WITHOUT being told
where they are.

This file imports build_graph() from graph_tracing.py, so keep both files in
the same folder. (I said Step 4 would go inside graph_tracing.py, but a separate
file is cleaner: graph_tracing.py builds the graph, this file investigates it.)

Run:  python fund_tracing.py

Remember: the network is SYNTHETIC. We planted the rings, so finding them proves
the technique works, not that this dataset contains real fraud rings.
"""

import networkx as nx

from graph_tracing import build_graph, load_scored_transactions


# ------------------------------------------------------------------
# TOOL 1: shrink the graph to the part that matters
# ------------------------------------------------------------------
def transfer_graph(G):
    """
    Merchants are dead ends: they only receive money, never pass it on.
    Throw them away and keep only user/offshore accounts. What is left is the
    network of account-to-account TRANSFERS, which is where tracing happens.
    """
    keep = [n for n, d in G.nodes(data=True) if d["kind"] != "merchant"]
    return G.subgraph(keep).copy()


# ------------------------------------------------------------------
# TOOL 2: follow the money forward from one account
# ------------------------------------------------------------------
def trace_forward(T, start, max_hops=4):
    """Return every path money could take from `start`, up to max_hops steps."""
    paths = nx.single_source_shortest_path(T, start, cutoff=max_hops)
    # drop the trivial path [start] that goes nowhere
    return [p for p in paths.values() if len(p) > 1]


# ------------------------------------------------------------------
# TOOL 3: find connected clusters (weakly connected components)
# A component = a group of accounts linked to each other, ignoring direction.
# Most accounts have no transfers at all, so we only keep groups of 3 or more.
# ------------------------------------------------------------------
def find_clusters(T, min_size=3):
    comps = [c for c in nx.weakly_connected_components(T) if len(c) >= min_size]
    return sorted(comps, key=len, reverse=True)


def describe_cluster(T, nodes):
    sub = T.subgraph(nodes)
    sources = [n for n in sub if sub.in_degree(n) == 0]    # money starts here
    sinks = [n for n in sub if sub.out_degree(n) == 0]     # money ends here
    hub = max(sub.nodes, key=sub.in_degree)
    # chain  = nobody receives from 2+ accounts
    # fan-in = at least one account collects from several senders
    shape = "fan-in hub" if sub.in_degree(hub) >= 3 else "chain"
    risk = sum(d["mean_prob"] for _, _, d in sub.edges(data=True)) / sub.number_of_edges()
    return {
        "nodes": len(sub), "edges": sub.number_of_edges(), "shape": shape,
        "sources": sources, "sinks": sinks, "hub": hub,
        "hub_in": sub.in_degree(hub), "risk": round(risk, 3),
    }


# ------------------------------------------------------------------
# TOOL 4: mule detection
# A mule collects from several senders and passes the money on.
# We use degree INSIDE the transfer graph, so the mule's normal shopping
# at merchants no longer confuses us (that was the issue in Step 3).
# ------------------------------------------------------------------
def find_mules(T, min_in=3):
    return [
        n for n in T.nodes
        if T.nodes[n]["kind"] == "user" and T.in_degree(n) >= min_in and T.out_degree(n) >= 1
    ]


# ------------------------------------------------------------------
# TOOL 5: draw a cluster (darker red = riskier according to the model)
# ------------------------------------------------------------------
def draw_cluster(T, nodes, title):
    import matplotlib.pyplot as plt   # imported here so the web server does not need matplotlib
    sub = T.subgraph(nodes)
    pos = nx.spring_layout(sub, seed=1)
    risks = [sub.nodes[n]["risk"] for n in sub]
    labels = {n: n.replace("USER_", "U").replace("OFFSHORE_", "OFF_") for n in sub}
    plt.figure(figsize=(7, 5))
    nx.draw(sub, pos, labels=labels, node_color=risks, cmap=plt.cm.Reds,
            vmin=0, vmax=1, node_size=1100, font_size=8, arrowsize=18)
    plt.title(title)


# ------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt   # only needed when running this file directly

    df, _ = load_scored_transactions()
    G = build_graph(df)
    T = transfer_graph(G)

    print(f"Full graph:     {G.number_of_nodes()} accounts, {G.number_of_edges()} edges")
    print(f"Transfer graph: {T.number_of_nodes()} accounts, {T.number_of_edges()} edges"
          "  (merchants removed)")

    # ---- Find all suspicious clusters ----
    clusters = find_clusters(T)
    print(f"\nClusters of 3+ connected accounts found: {len(clusters)}")

    # Ground truth, used ONLY to grade ourselves afterwards
    planted = df[df["Planted_Pattern"] != "none"]
    label_of_edge = dict(zip(zip(planted["Sender_ID"], planted["Receiver_ID"]),
                             planted["Planted_Pattern"]))

    rows = []
    for nodes in clusters:
        info = describe_cluster(T, nodes)
        truth = {label_of_edge.get(e, "?") for e in T.subgraph(nodes).edges}
        rows.append((info, truth, nodes))

    rows.sort(key=lambda r: r[0]["risk"], reverse=True)  # riskiest first
    print("\nCluster | accounts | shape      | avg edge risk | planted label (answer key)")
    for i, (info, truth, _) in enumerate(rows, 1):
        print(f"   {i}    |    {info['nodes']:2d}    | {info['shape']:10s} |     "
              f"{info['risk']:.3f}     | {', '.join(sorted(truth))}")

    # ---- Mule detection ----
    mules = find_mules(T)
    print("\nAccounts that look like mules (3+ senders in, passes money on):", mules)

    # ---- Trace forward from one account of the riskiest chain ----
    chain_rows = [r for r in rows if r[0]["shape"] == "chain"]
    top = chain_rows[0]
    start = top[0]["sources"][0]
    print(f"\nFollowing the money forward from {start} (riskiest chain):")
    for path in trace_forward(T, start, max_hops=6):
        print("  " + " -> ".join(path))

    # ---- Path between two specific accounts ----
    end = top[0]["sinks"][0]
    print(f"\nShortest path {start} -> {end}:", nx.shortest_path(T, start, end))
    print(f"Can money from {end} get back to {start}?", nx.has_path(T, end, start))

    # ---- Draw the riskiest chain and the hub ----
    draw_cluster(T, top[2], "Riskiest chain (synthetic)")
    hub_rows = [r for r in rows if r[0]["shape"] == "fan-in hub"]
    if hub_rows:
        draw_cluster(T, hub_rows[0][2], "Mule hub (synthetic)")
    plt.show()