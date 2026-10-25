# Tracer: Fraud Detection and Fund Tracing

**Score the transaction. Follow the money.**

Tracer is an end-to-end machine learning project. An XGBoost model scores individual card and bank transactions for fraud risk, and a graph layer then follows the money between accounts to expose patterns that a single score can miss: chains of accounts passing funds along, and "mule" accounts that collect from many senders. Everything is served through a FastAPI backend and a single-page dashboard.

It is a learning project built on **synthetic data**. It shows the full workflow, from data and model to API and interface, and it is honest about what it can and cannot claim (see [Limitations](#limitations)).

## What it does

- **Scores a transaction.** Send 17 fields to the API and get back a fraud probability, a flag, and the decision threshold.
- **Builds a transaction network.** Accounts become nodes and payments become directed edges carrying the model's risk.
- **Finds suspicious structure.** It groups connected accounts into clusters, labels them as chains or fan-in hubs, detects mule-like accounts, and traces money forward from any account.
- **Presents it in a dashboard.** A transaction checker, a session history of flagged cases, an interactive network view, and a page explaining the model.

## How it works

### 1. The model
- **Data:** `synthetic_fraud_dataset2.csv`, 70,000 transactions with 21 columns and roughly 32% labelled fraud.
- **Features:** six categorical fields (transaction type, device, card type, authentication method, merchant category, location) and numeric fields such as amount, balance, daily transaction count, 7-day averages, failed attempts, card age and distance. Three features are engineered from the timestamp: hour, day of week and a corrected weekend flag (the raw `Is_Weekend` column was inconsistent with the timestamps). The dataset's `Risk_Score` column is deliberately not used as a feature.
- **Preprocessing:** one-hot encoding for categories, standard scaling for numbers, in a single scikit-learn `ColumnTransformer`.
- **Model:** XGBoost classifier (100 trees, learning rate 0.01, depth 3), trained on a 60/20/20 train, validation and test split with a fixed random seed.
- **Threshold:** chosen on the **validation** set by maximising F1 over the precision-recall curve, then evaluated once on the untouched test set. The saved threshold is about **0.751**, and a transaction is flagged when its probability is at or above it.

### 2. The transaction network
The dataset has a sender (`User_ID`) but no receiver, so it contains no real money trail. `build_network.py` therefore adds a **synthetic, clearly labelled** receiver to every transaction:

- Ordinary transactions go to **merchant accounts** (`MERCH_<CATEGORY>_<n>`), which only receive money.
- **6 fraud rings** are planted: chains of 5 user accounts passing money along, ending in an offshore account.
- **1 mule hub** is planted: 8 senders paying one account, which forwards the money offshore.

The planted rows reuse real fraud-labelled transactions, so the model scores them with their real features.

### 3. Fund tracing (NetworkX)
- Each edge stores the number of transactions, the total amount, and the mean and maximum fraud probability. A node's risk is the average risk of the edges touching it.
- Merchants are dead ends, so they are removed to leave the **transfer graph** of account-to-account payments.
- **Clusters** are the weakly connected components of that graph. Each is classified as a *chain* or a *fan-in hub* and given an average edge risk.
- **Mules** are user accounts with three or more incoming senders that pass money on.
- **Tracing** returns the paths money can take forward from an account, up to a chosen number of hops.

On the included data this finds 7 clusters (six chains of 6 accounts and one hub of 10), 8,971 accounts and 39 transfer edges in the transfer graph, and one mule account, without being told where they are.

### 4. API and dashboard
A FastAPI app exposes the model and the network. The dashboard, **Tracer**, is a single `index.html` with plain HTML, CSS and JavaScript (no framework, no build step), served by the same app.

Dashboard features:
- **Check:** a validated form with example inputs, an animated risk meter with the threshold marked, a session history with a flagged-only filter, CSV export, and optional sender and receiver IDs to add the transfer to the live network.
- **Investigate:** cluster list, an interactive graph with risk-coloured nodes, mule highlighting, and a "trace money forward" animation.
- **About:** model card, metrics and limitations.
- Light and dark themes, responsive layout, and a clearly labelled demo mode if the API cannot be reached.

## Results

Evaluated on the held-out test set at the saved threshold:

| Metric | Value |
|---|---|
| Accuracy | TBD |
| Fraud precision | TBD |
| Fraud recall | TBD |
| Fraud F1 | TBD |
| ROC-AUC | TBD |
| Average precision | TBD |

Run `python main.py` to reproduce these numbers and print the full classification report and confusion matrix.

## Tech stack

Python 3.13, pandas, NumPy, scikit-learn, XGBoost, NetworkX, FastAPI, Pydantic, Uvicorn, and vanilla HTML, CSS and JavaScript.

## Project structure

```
.
├── app.py                         # FastAPI app: routes and input validation
├── model.py                       # loads the saved model, predict_transaction()
├── network.py                     # in-memory network state used by the API
├── graph_tracing.py               # scores transactions and builds the graph
├── fund_tracing.py                # clusters, mule detection, forward tracing
├── build_network.py               # adds the synthetic receivers (run once)
├── main.py                        # trains the model, picks the threshold, saves files
├── index.html                     # the Tracer dashboard
├── fraud_model.joblib             # trained XGBoost model
├── fraud_preprocessor.joblib      # fitted preprocessing pipeline
├── fraud_threshold.txt            # decision threshold
├── synthetic_fraud_dataset2.csv   # training data
├── transactions_network.csv       # transactions with synthetic receivers
├── toy_graph.py / toy_graph.png   # small graph-basics example
└── requirements.txt               # pinned dependencies
```

## Getting started

```bash
git clone https://github.com/mohithgowdadb/Fraud-Detection.git
cd Fraud-Detection

python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> for the dashboard, or <http://127.0.0.1:8000/docs> for the interactive API documentation.

The dependency versions are pinned on purpose. A saved scikit-learn or XGBoost model only loads safely with the versions that created it (scikit-learn 1.7.2 and XGBoost 3.4.1 here). `xgboost-cpu` is the CPU-only build of the same library, which keeps the install small.

To retrain the model, run `python main.py`. It reads `synthetic_fraud_dataset2.csv` and overwrites the three saved model files.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | The dashboard |
| `POST` | `/predict` | Score a transaction. Optional `Sender_ID` and `Receiver_ID` add the transfer to the live network |
| `GET` | `/clusters` | All clusters, ranked by average risk |
| `GET` | `/clusters/{id}` | Nodes, edges and mule accounts of one cluster |
| `GET` | `/accounts/{id}` | Summary of one account |
| `GET` | `/trace/{account_id}?hops=6` | Paths money can take forward (1 to 10 hops) |
| `GET` | `/network/stats` | Cluster, account, edge and mule counts |
| `POST` | `/network/reset` | Remove live transfers added during a session |

Example request to `/predict`:

```json
{
  "Timestamp": "2023-08-14T19:30:00",
  "Transaction_Type": "Online",
  "Device_Type": "Mobile",
  "Card_Type": "Visa",
  "Authentication_Method": "OTP",
  "Merchant_Category": "Electronics",
  "Location": "Mumbai",
  "Transaction_Amount": 450,
  "Account_Balance": 12000,
  "IP_Address_Flag": 1,
  "Previous_Fraudulent_Activity": 0,
  "Daily_Transaction_Count": 6,
  "Avg_Transaction_Amount_7d": 120,
  "Failed_Transaction_Count_7d": 3,
  "Card_Age": 20,
  "Transaction_Distance": 2500,
  "Is_Weekend": 0
}
```

Response shape (values are illustrative):

```json
{ "fraud_probability": 0.1595, "is_flagged": false, "threshold": 0.751 }
```

Invalid input, such as an unknown category or a negative amount, is rejected with a `422` error that names the offending field.

## Limitations

- **The data is synthetic.** Results say nothing about real-world fraud rates or behaviour.
- **The network is synthetic.** Receivers were generated, and the rings and the mule were planted by me. Finding them shows the technique works, not that such rings exist in real data. The network is also sparse, which makes detection easier than it would be in practice.
- **The model and the network use different data files.** The model is trained on the 70,000-row dataset, while `transactions_network.csv` was generated from an earlier 50,000-row version of the same data. That original file is not included, so `build_network.py` expects it under its old name, `synthetic_fraud_dataset.csv`.
- **The threshold is a trade-off.** A higher threshold flags fewer transactions and misses more fraud.
- **No per-prediction explanations yet.** The model returns a score, not the reasons for it.
- **Tracing ignores timing.** It follows who paid whom, not the order in time of the transfers.
- **Live transfers are not persisted.** Transfers added through the dashboard live in memory and disappear when the server restarts.
- **This is a learning project** and must not be used for real financial decisions.

## Roadmap

- Regenerate the network from the full 70,000-row dataset so everything uses the same data.
- Add per-transaction explanations (for example SHAP values).
- Make tracing time-aware.
- Containerise with Docker and deploy.

## Author

Built by **Mohith Gowda DB** as a hands-on machine learning project covering modelling, graph analysis and deployment-ready web development. GitHub: [@mohithgowdadb](https://github.com/mohithgowdadb)
