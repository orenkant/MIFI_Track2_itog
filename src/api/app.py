"""REST API для PD-модели."""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.models.predict import load_model, predict

state = {}


@asynccontextmanager
async def lifespan(app):
    # модель загружается один раз при старте, пути можно переопределить (для тестов)
    model, meta = load_model(
        os.getenv("MODEL_PATH", "models/model.pkl"),
        os.getenv("META_PATH", "models/model_meta.json"),
    )
    state["model"], state["meta"] = model, meta
    yield
    state.clear()


app = FastAPI(title="Credit Default Prediction API", version="1.0.0", lifespan=lifespan)


class ClientData(BaseModel):
    """Данные клиента, ограничения те же, что в правилах Great Expectations."""

    LIMIT_BAL: float = Field(gt=0, le=2_000_000)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_0: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float = Field(ge=-1_000_000, le=3_000_000)
    BILL_AMT2: float = Field(ge=-1_000_000, le=3_000_000)
    BILL_AMT3: float = Field(ge=-1_000_000, le=3_000_000)
    BILL_AMT4: float = Field(ge=-1_000_000, le=3_000_000)
    BILL_AMT5: float = Field(ge=-1_000_000, le=3_000_000)
    BILL_AMT6: float = Field(ge=-1_000_000, le=3_000_000)
    PAY_AMT1: float = Field(ge=0, le=3_000_000)
    PAY_AMT2: float = Field(ge=0, le=3_000_000)
    PAY_AMT3: float = Field(ge=0, le=3_000_000)
    PAY_AMT4: float = Field(ge=0, le=3_000_000)
    PAY_AMT5: float = Field(ge=0, le=3_000_000)
    PAY_AMT6: float = Field(ge=0, le=3_000_000)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 20000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": -1,
                "PAY_4": -1,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 3913,
                "BILL_AMT2": 3102,
                "BILL_AMT3": 689,
                "BILL_AMT4": 0,
                "BILL_AMT5": 0,
                "BILL_AMT6": 0,
                "PAY_AMT1": 0,
                "PAY_AMT2": 689,
                "PAY_AMT3": 0,
                "PAY_AMT4": 0,
                "PAY_AMT5": 0,
                "PAY_AMT6": 0,
            }
        },
    )


class Prediction(BaseModel):
    default_prediction: int
    default_probability: float


def run_model(clients):
    df = pd.DataFrame([c.model_dump() for c in clients])
    classes, proba = predict(state["model"], df, state["meta"]["threshold"])
    return [
        Prediction(default_prediction=int(c), default_probability=float(p))
        for c, p in zip(classes, proba)
    ]


@app.get("/")
def root():
    return {"message": "Credit Default Prediction API is alive!"}


@app.get("/health")
def health():
    meta = state["meta"]
    return {
        "status": "ok",
        "model_name": meta["model_name"],
        "threshold": meta["threshold"],
    }


@app.post("/predict", response_model=Prediction)
def predict_one(client: ClientData):
    return run_model([client])[0]


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(clients: list[ClientData]):
    if not clients:
        raise HTTPException(status_code=422, detail="Пустой список клиентов")
    return run_model(clients)
