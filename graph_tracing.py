"""
Phase 4, Step 3: Build the transaction graph and attach model risk to it.

What this file does:
  1. Loads transactions_network.csv (made by build_network.py: SYNTHETIC receivers)
  2. Scores every transaction with your saved XGBoost model -> Fraud_Prob
  3. Builds a NetworkX directed graph:
        node = account            edge = "sender paid receiver"
  4. Stores risk numbers on the edges and the nodes

Files needed in the same folder:
  transactions_network.csv, fraud_model.joblib,
  fraud_preprocessor.joblib, fraud_threshold.txt

Run:  python graph_tracing.py
Step 4 will add the tracing functions to this same file.
"""

from pathlib import Path

import joblib
import networkx as nx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent   # folder this file lives in


# ------------------------------------------------------------------
# PART A: score every transaction with the saved model
# ------------------------------------------------------------------
def load_scored_transactions(csv_path=BASE_DIR / "transactions_network.csv"):
    df = pd.read_csv(csv_path)

    # Recreate the SAME engineered features as in main.py
    # (the preprocessor expects these exact columns)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Transaction_Hour"] = df["Timestamp"].dt.hour
    df["Day_Of_Week"] = df["Timestamp"].dt.dayofweek
    df["Is_Weekend_Corrected"] = (df["Day_Of_Week"] >= 5).astype(int)

    model = joblib.load(BASE_DIR / "fraud_model.joblib")
    preprocessor = joblib.load(BASE_DIR / "fraud_preprocessor.joblib")
    with open(BASE_DIR / "fraud_threshold.txt") as f:
        threshold = float(f.read())

    # The preprocessor picks the columns it needs by NAME, so passing the whole
    # DataFrame (with the extra Sender_ID / Receiver_ID columns) is fine.
    X_processed = preprocessor.transform(df)
    df["Fraud_Prob"] = model.predict_proba(X_processed)[:, 1]
    df["Flagged"] = (df["Fraud_Prob"] >= threshold).astype(int)
    return df, threshold


# ------------------------------------------------------------------
# PART B: turn the table into a graph
# ------------------------------------------------------------------
def account_kind(name):
    """Label each account so we can tell users, merchants and offshore apart."""
    if name.startswith("MERCH_"):
        return "merchant"
    if name.startswith("OFFSHORE_"):
        return "offshore"
    return "user"


def build_graph(df):
    # Many transactions can go from the same sender to the same receiver.
    # A DiGraph holds ONE edge per pair, so we first combine them.
    edges = (
        df.groupby(["Sender_ID", "Receiver_ID"])
        .agg(
            n_txn=("Transaction_ID", "count"),
            total_amount=("Transaction_Amount", "sum"),
            max_prob=("Fraud_Prob", "max"),
            mean_prob=("Fraud_Prob", "mean"),
            n_flagged=("Flagged", "sum"),
        )
        .reset_index()
    )

    G = nx.DiGraph()
    for row in edges.itertuples(index=False):
        G.add_edge(
            row.Sender_ID,
            row.Receiver_ID,
            n_txn=row.n_txn,
            total_amount=round(row.total_amount, 2),
            max_prob=round(row.max_prob, 4),
            mean_prob=round(row.mean_prob, 4),
            n_flagged=row.n_flagged,
        )

    # Node-level information
    for node in G.nodes:
        G.nodes[node]["kind"] = account_kind(node)

    # A node's risk = the average risk of the money coming IN or going OUT of it
    for node in G.nodes:
        probs = [d["mean_prob"] for _, _, d in G.in_edges(node, data=True)]
        probs += [d["mean_prob"] for _, _, d in G.out_edges(node, data=True)]
        G.nodes[node]["risk"] = round(sum(probs) / len(probs), 4) if probs else 0.0

    return G


# ------------------------------------------------------------------
# Run this file directly to check that everything is wired up correctly
# ------------------------------------------------------------------
if __name__ == "__main__":
    df, threshold = load_scored_transactions()
    print(f"Scored {len(df)} transactions. Flag threshold = {threshold:.4f}")
    print(f"Transactions flagged as fraud by the model: {df['Flagged'].sum()}")

    G = build_graph(df)
    print(f"\nGraph: {G.number_of_nodes()} accounts, {G.number_of_edges()} edges")

    kinds = pd.Series([d["kind"] for _, d in G.nodes(data=True)]).value_counts()
    print("\nAccounts by kind:")
    print(kinds.to_string())

    # The interesting edges: user -> user (everything else goes to merchants)
    user_edges = [
        (a, b) for a, b in G.edges
        if G.nodes[a]["kind"] == "user" and G.nodes[b]["kind"] == "user"
    ]
    print(f"\nUser -> user edges: {len(user_edges)}  (we planted 24 ring links + 8 feeders = 32)")

    # How does the model score the planted transactions vs normal ones?
    print("\nAverage model fraud probability by planted pattern:")
    print(df.groupby("Planted_Pattern")["Fraud_Prob"].mean().round(3).to_string())

    # Peek at the mule account from build_network.py
    mule = df[df["Planted_Pattern"] == "mule_hub"]["Receiver_ID"].value_counts().index[0]
    print(f"\nMule account: {mule}")
    print("  in_degree :", G.in_degree(mule), "(who sends to it)")
    print("  out_degree:", G.out_degree(mule), "(who it sends to)")
    print("  risk      :", G.nodes[mule]["risk"])
    print("  sends to  :", list(G.successors(mule)))