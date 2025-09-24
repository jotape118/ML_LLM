"""
Inferencia a partir del pipeline guardado.
- predict_single: dict -> {pred, proba}
- predict_batch: DataFrame -> array de proba
"""
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from joblib import load

PIPE_PATH = Path("artifacts/ml/pipeline.joblib")

def _load_pipeline():
    if not PIPE_PATH.exists():
        raise FileNotFoundError("No existe artifacts/ml/pipeline.joblib. Entrena con `python -m src.train`.")
    return load(PIPE_PATH)

def predict_single(features: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _load_pipeline()
    df = pd.DataFrame([features])
    proba = pipe.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}

def predict_batch(df: pd.DataFrame):
    pipe = _load_pipeline()
    return pipe.predict_proba(df)[:, 1]
