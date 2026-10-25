from datetime import datetime
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import network
from model import predict_transaction

app = FastAPI()

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

class Transaction(BaseModel):
    Timestamp: datetime
    Transaction_Type: Literal["ATM Withdrawal", "Bank Transfer", "Online", "POS"]
    Device_Type: Literal["Laptop", "Mobile", "Tablet"]
    Card_Type: Literal["Amex", "Discover", "Mastercard", "Visa"]
    Authentication_Method: Literal["Biometric", "OTP", "PIN", "Password"]
    Merchant_Category: Literal["Clothing", "Electronics", "Groceries", "Restaurants", "Travel"]
    Location: Literal["London", "Mumbai", "New York", "Sydney", "Tokyo"]
    Transaction_Amount: float = Field(ge=0)
    Account_Balance: float = Field(ge=0)
    IP_Address_Flag: Literal[0, 1]
    Previous_Fraudulent_Activity: Literal[0, 1]
    Daily_Transaction_Count: int = Field(ge=0)
    Avg_Transaction_Amount_7d: float = Field(ge=0)
    Failed_Transaction_Count_7d: int = Field(ge=0)
    Card_Age: int = Field(ge=0)
    Transaction_Distance: float = Field(ge=0)
    Is_Weekend: Literal[0, 1]
    # Optional: if both are given, the transfer is added to the network graph
    Sender_ID: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,30}$")
    Receiver_ID: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,30}$")


@app.on_event("startup")
def warm_network():
    network.warm()   # score the CSV and build the graph once, so the first page visit is fast


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.model_dump()
    sender, receiver = data.pop("Sender_ID"), data.pop("Receiver_ID")   # model.py never sees these
    if sender and receiver and sender.upper() == receiver.upper():
        raise HTTPException(422, detail=[{"loc": ["body", "Receiver_ID"], "msg": "Receiver must differ from sender.", "type": "value_error"}])
    try:
        result = predict_transaction(data)
        if sender and receiver:
            result["network"] = network.add_transfer(sender.upper(), receiver.upper(), data["Transaction_Amount"],
                                                     result["fraud_probability"], result["is_flagged"])
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")


@app.get("/clusters")
def clusters():
    return network.list_clusters()


@app.get("/clusters/{cluster_id}")
def cluster(cluster_id: int):
    c = network.get_cluster(cluster_id)
    if c is None:
        raise HTTPException(404, "Cluster not found")
    return c


@app.get("/accounts/{account_id}")
def account(account_id: str):
    a = network.get_account(account_id.upper())
    if a is None:
        raise HTTPException(404, "Account not found")
    return a


@app.get("/trace/{account_id}")
def trace(account_id: str, hops: int = Query(6, ge=1, le=10)):
    t = network.trace(account_id.upper(), hops)
    if t is None:
        raise HTTPException(404, "Account not found")
    return t


@app.get("/network/stats")
def network_stats():
    return network.stats()


@app.post("/network/reset")
def network_reset():
    network.reset()
    return {"ok": True}