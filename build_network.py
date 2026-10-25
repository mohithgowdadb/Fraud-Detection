"""
Phase 4, Step 2: Add a SYNTHETIC receiver to every transaction.

!! IMPORTANT !!
The original dataset has a sender (User_ID) but NO receiver, so there is no
real money trail in it. Everything this script adds is SYNTHETIC (made up by
us) and must be labelled that way in the README / report.

What we create:
  1. Normal transactions  -> paid to a merchant account (MERCH_<category>_<n>).
                             Merchants only receive money, they never send it
                             on, so they are dead ends for tracing.
  2. Fraud RINGS          -> chains of 5 user accounts, each passing money to
                             the next:  U1 -> U2 -> U3 -> U4 -> U5
  3. One MULE HUB         -> many fraud senders -> one mule account -> one
                             offshore exit account

Output: transactions_network.csv  (original columns + Receiver_ID + Planted_Pattern)

Run:  python build_network.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed = same network every run

N_RINGS = 6          # how many chain rings to plant
RING_SIZE = 5        # accounts per ring
N_HUB_FEEDERS = 8    # how many fraud senders feed the mule account

df = pd.read_csv("synthetic_fraud_dataset.csv")
df["Sender_ID"] = df["User_ID"]

# ------------------------------------------------------------------
# 1. Everyone pays a merchant (default for ALL transactions)
#    The merchant is chosen from the transaction's own Merchant_Category,
#    so the synthetic data stays consistent with the real column.
# ------------------------------------------------------------------
MERCHANTS_PER_CATEGORY = 40
merchant_num = rng.integers(1, MERCHANTS_PER_CATEGORY + 1, size=len(df))
df["Receiver_ID"] = (
    "MERCH_" + df["Merchant_Category"].str.upper().str.replace(" ", "_")
    + "_" + pd.Series(merchant_num).astype(str).str.zfill(2)
)
df["Planted_Pattern"] = "none"

# ------------------------------------------------------------------
# 2. Pick users who have at least one fraud transaction.
#    We take ONE fraud row per user and reuse it as a transfer.
# ------------------------------------------------------------------
fraud_rows = df[df["Fraud_Label"] == 1]
one_fraud_row_per_user = fraud_rows.groupby("User_ID").sample(n=1, random_state=42)

# Shuffle, then hand out users without reusing any
pool = list(rng.permutation(one_fraud_row_per_user.index.to_numpy()))


def take(n):
    """Remove and return n row indexes from the pool."""
    chosen, rest = pool[:n], pool[n:]
    pool[:] = rest
    return chosen


# ------------------------------------------------------------------
# 3. Plant the chain rings:  U1 -> U2 -> U3 -> U4 -> U5
#    Row of U1 gets receiver U2, row of U2 gets receiver U3, and so on.
#    The last member sends to an offshore account.
# ------------------------------------------------------------------
for ring in range(1, N_RINGS + 1):
    rows = take(RING_SIZE)
    members = df.loc[rows, "User_ID"].tolist()
    for i, row in enumerate(rows):
        if i < RING_SIZE - 1:
            df.loc[row, "Receiver_ID"] = members[i + 1]
        else:
            df.loc[row, "Receiver_ID"] = f"OFFSHORE_RING{ring}"
        df.loc[row, "Planted_Pattern"] = f"ring_{ring}"

# ------------------------------------------------------------------
# 4. Plant the mule hub:  8 senders -> MULE -> OFFSHORE_HUB
# ------------------------------------------------------------------
mule_row = take(1)[0]
mule_user = df.loc[mule_row, "User_ID"]
df.loc[mule_row, "Receiver_ID"] = "OFFSHORE_HUB"
df.loc[mule_row, "Planted_Pattern"] = "mule_hub"

for row in take(N_HUB_FEEDERS):
    df.loc[row, "Receiver_ID"] = mule_user
    df.loc[row, "Planted_Pattern"] = "mule_hub"

# ------------------------------------------------------------------
# 5. Save + summary
# ------------------------------------------------------------------
df.to_csv("transactions_network.csv", index=False)

print("Saved transactions_network.csv")
print("Total transactions:", len(df))
print("\nPlanted patterns:")
print(df["Planted_Pattern"].value_counts())
print("\nMule account (ground truth, remember this!):", mule_user)
print("\nExample ring 1 transfers (sender -> receiver):")
print(df[df["Planted_Pattern"] == "ring_1"][["Sender_ID", "Receiver_ID", "Transaction_Amount"]])