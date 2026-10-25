"""
Phase 4, Step 1: Graph basics with a toy example.

A graph is just two things:
  - NODES  = things (here: bank accounts)
  - EDGES  = connections between things (here: money transfers)
A DIRECTED edge A -> B means "money went FROM A TO B". Direction matters!

Run:  pip install networkx matplotlib   (if you don't have them)
      python toy_graph.py
"""

import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Build a small graph by hand
# ------------------------------------------------------------------
G = nx.DiGraph()  # DiGraph = Directed Graph

# Each tuple is (sender, receiver, amount)
transfers = [
    ("A", "B", 5000),   # A sends 5000 to B
    ("B", "C", 4900),   # B passes almost all of it to C
    ("C", "D", 4800),   # C passes it on to D  -> a CHAIN: A -> B -> C -> D
    ("E", "B", 300),    # E also sends a little to B
    ("F", "B", 250),    # F also sends a little to B
    ("E", "F", 100),    # E and F know each other
]

for sender, receiver, amount in transfers:
    G.add_edge(sender, receiver, amount=amount)  # nodes are created automatically

# ------------------------------------------------------------------
# 2. Look at the basic properties
# ------------------------------------------------------------------
print("Nodes:", list(G.nodes))
print("Edges:", list(G.edges))
print("Number of accounts:", G.number_of_nodes())
print("Number of transfers:", G.number_of_edges())

# Edge data is stored like a dictionary
print("\nAmount on A -> B:", G["A"]["B"]["amount"])

# ------------------------------------------------------------------
# 3. Neighbours: who is directly connected to whom?
# ------------------------------------------------------------------
print("\nWho did B send money TO?    ", list(G.successors("B")))
print("Who sent money TO B?        ", list(G.predecessors("B")))

# ------------------------------------------------------------------
# 4. Degree: how many edges touch a node?
#    in_degree  = money coming IN  (how many senders)
#    out_degree = money going OUT  (how many receivers)
# ------------------------------------------------------------------
print("\nAccount | in | out")
for node in G.nodes:
    print(f"   {node}    |  {G.in_degree(node)} |  {G.out_degree(node)}")

# Think about it: B receives from 3 accounts (A, E, F) and sends to 1 (C).
# Many in, few out, money passing straight through... that is the shape
# of a "mule" account. We will use this idea in Step 4.

# ------------------------------------------------------------------
# 5. Paths: can money get from one account to another?
# ------------------------------------------------------------------
print("\nIs there a path A -> D?", nx.has_path(G, "A", "D"))
print("The path:", nx.shortest_path(G, "A", "D"))
print("Is there a path D -> A?", nx.has_path(G, "D", "A"))  # direction matters!

# All accounts that money from A could have reached
print("Reachable from A:", nx.descendants(G, "A"))
# All accounts that could have sent money to D
print("Could have fed D:", nx.ancestors(G, "D"))

# ------------------------------------------------------------------
# 6. Draw it
# ------------------------------------------------------------------
pos = nx.spring_layout(G, seed=42)  # seed makes the layout the same every run
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=1200, font_size=14, arrowsize=20)
edge_labels = nx.get_edge_attributes(G, "amount")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Toy transfer network")
plt.savefig("toy_graph.png", dpi=120, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# YOUR TURN (try these, then run again):
#   1. Add a transfer ("D", "G", 4700). What is the path from A to G now?
#   2. Add ("D", "A", 100). Is there a cycle? Try nx.find_cycle(G).
#   3. Which account has the highest in_degree? Which has the highest out_degree?
#   4. Use G.successors() to list everyone two hops away from A.
# ------------------------------------------------------------------