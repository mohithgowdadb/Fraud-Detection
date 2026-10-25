import os
import threading

os.environ.setdefault("MPLBACKEND", "Agg")   # fund_tracing imports matplotlib; no window on a server

import pandas as pd
from fund_tracing import describe_cluster, find_clusters, find_mules, trace_forward, transfer_graph
from graph_tracing import build_graph, load_scored_transactions

_lock = threading.Lock()
_base = None      # scored transactions from the CSV
_live = []        # transfers added from the Check page
_cache = None     # (transfer graph, clusters), rebuilt after every change


def _state():
    global _base, _cache
    with _lock:
        if _base is None:
            _base, _ = load_scored_transactions()
        if _cache is None:
            df = pd.concat([_base, pd.DataFrame(_live)], ignore_index=True) if _live else _base
            T = transfer_graph(build_graph(df))
            infos = [(c, describe_cluster(T, c)) for c in find_clusters(T, min_size=2)]
            infos.sort(key=lambda x: (-x[1]["risk"], -len(x[0]), min(x[0])))
            _cache = (T, [{"id": i, "nodes": n, "info": inf} for i, (n, inf) in enumerate(infos, 1)])
        return _cache


def warm():
    _state()


def add_transfer(sender, receiver, amount, prob, flagged):
    global _cache
    with _lock:
        _live.append({"Sender_ID": sender, "Receiver_ID": receiver, "Transaction_ID": f"LIVE_{len(_live) + 1}",
                      "Transaction_Amount": float(amount), "Fraud_Prob": float(prob), "Flagged": int(bool(flagged)),
                      "Planted_Pattern": "live"})
        _cache = None
    T, clusters = _state()
    cid = next((c["id"] for c in clusters if sender in c["nodes"] and receiver in c["nodes"]), None)
    return {"added": True, "cluster_id": cid, "live_transfers": len(_live)}


def reset():
    global _cache
    with _lock:
        _live.clear()
        _cache = None


def stats():
    T, clusters = _state()
    return {"clusters": len(clusters), "accounts": T.number_of_nodes(), "edges": T.number_of_edges(), "mules": len(find_mules(T))}


def list_clusters():
    _, clusters = _state()
    pairs = {(r["Sender_ID"], r["Receiver_ID"]) for r in _live}
    return [{"id": c["id"], "shape": c["info"]["shape"], "n_accounts": c["info"]["nodes"], "n_edges": c["info"]["edges"],
             "avg_risk": float(c["info"]["risk"]), "has_live": any(e in pairs for e in _edges(c))} for c in clusters]


def _edges(c):
    T, _ = _state()
    return T.subgraph(c["nodes"]).edges


def get_cluster(cid):
    T, clusters = _state()
    c = next((c for c in clusters if c["id"] == cid), None)
    if c is None:
        return None
    sub = T.subgraph(c["nodes"])
    mules = set(find_mules(T))
    return {"id": cid, "shape": c["info"]["shape"], "avg_risk": float(c["info"]["risk"]),
            "nodes": [{"id": n, "kind": sub.nodes[n]["kind"], "risk": float(sub.nodes[n]["risk"])} for n in sub],
            "edges": [{"source": a, "target": b, "mean_prob": float(d["mean_prob"]), "n_txn": int(d["n_txn"]),
                       "total_amount": float(d["total_amount"])} for a, b, d in sub.edges(data=True)],
            "mules": [n for n in sub if n in mules]}


def get_account(aid):
    T, clusters = _state()
    if aid not in T:
        return None
    cid = next((c["id"] for c in clusters if aid in c["nodes"]), None)
    return {"account": aid, "kind": T.nodes[aid]["kind"], "cluster_id": cid, "in_degree": T.in_degree(aid), "out_degree": T.out_degree(aid)}


def trace(aid, hops):
    T, _ = _state()
    if aid not in T:
        return None
    return {"start": aid, "paths": sorted(trace_forward(T, aid, hops), key=len)}