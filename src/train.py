"""
Entrenamiento reproducible:
- Carga data
- Split train/valid/test (stratify)
- Baseline (LogisticRegression)
- Modelo mejorado (XGBoost) [fallback a RandomForest si falla]
- Selecciona por F1 en valid
- Guarda pipeline + metadata en artifacts/
"""
import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

from .features import build_preprocessor, NUM_FEATURES, CAT_FEATURES

load_dotenv()
SEED = int(os.getenv("SEED", "42"))

ARTIF_DIR = Path("artifacts/ml")
ARTIF_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = Path("data/titanic.csv")

@dataclass
class Metadata:
    model_name: str
    seed: int
    train_shape: tuple
    valid_shape: tuple
    test_shape: tuple
    valid_metrics: dict
    created_at: str
    features: dict

def _load_data() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError("No se encontró data/titanic.csv. Corre `python data/get_data.py` primero.")
    df = pd.read_csv(DATA_CSV)
    # Drop rows sin features clave (imputador manejará nulos)
    return df

def _mk_splits(df: pd.DataFrame):
    y = df["survived"].astype(int)
    X = df.drop(columns=["survived"])
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    # Persistimos test para eval posterior
    Path("eval").mkdir(exist_ok=True)
    test_path = Path("eval/test.csv")
    pd.concat([y_test.rename("survived"), X_test], axis=1).to_csv(test_path, index=False)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def _build_models():
    baseline = Pipeline(steps=[
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=200, random_state=SEED))
    ])
    try:
        from xgboost import XGBClassifier
        improved = Pipeline(steps=[
            ("pre", build_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=SEED,
                eval_metric="logloss",
                n_jobs=-1
            ))
        ])
    except Exception:
        # Fallback si no hay xgboost
        from sklearn.ensemble import RandomForestClassifier
        improved = Pipeline(steps=[
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=SEED,
                n_jobs=-1
            ))
        ])
    return baseline, improved

def main():
    df = _load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = _mk_splits(df)
    baseline, improved = _build_models()

    # Entrenar ambos
    baseline.fit(X_train, y_train)
    improved.fit(X_train, y_train)

    # Validación
    def eval_model(model, splitX, splity):
        proba = model.predict_proba(splitX)[:, 1]
        pred = (proba >= 0.5).astype(int)
        f1 = f1_score(splity, pred)
        try:
            auc = roc_auc_score(splity, proba)
        except Exception:
            auc = None
        return {"f1": float(f1), "auc": float(auc) if auc is not None else None}

    m_base = eval_model(baseline, X_valid, y_valid)
    m_impr = eval_model(improved, X_valid, y_valid)

    # Selección por F1
    best_model, best_name, best_metrics = (
        (improved, "improved", m_impr) if m_impr["f1"] >= m_base["f1"] else (baseline, "baseline", m_base)
    )

    # Guardar pipeline
    pipe_path = ARTIF_DIR / "pipeline.joblib"
    dump(best_model, pipe_path)

    # Metadata
    meta = Metadata(
        model_name=best_name,
        seed=SEED,
        train_shape=X_train.shape,
        valid_shape=X_valid.shape,
        test_shape=X_test.shape,
        valid_metrics=best_metrics,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        features={"num": NUM_FEATURES, "cat": CAT_FEATURES},
    )
    with open(ARTIF_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    print(f"[OK] Guardado {pipe_path} con modelo='{best_name}', valid={best_metrics}")

if __name__ == "__main__":
    main()
