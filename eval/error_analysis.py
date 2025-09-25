"""
Error analysis del modelo en test set.
Genera CSV con ejemplos de FP y FN para revisar patrones.
"""
from pathlib import Path
import pandas as pd
from joblib import load

TEST_CSV = Path("eval/test.csv")
PIPE_PATH = Path("artifacts/ml/pipeline.joblib")
OUT_CSV = Path("eval/error_analysis.csv")

def main():
    df = pd.read_csv(TEST_CSV)
    y = df["survived"].astype(int)
    X = df.drop(columns=["survived"])

    pipe = load(PIPE_PATH)
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    df["prediction"] = pred
    df["proba"] = proba

    # Filtrar FP y FN
    fp = df[(df["survived"] == 0) & (df["prediction"] == 1)]
    fn = df[(df["survived"] == 1) & (df["prediction"] == 0)]

    out = pd.concat([fp.assign(error_type="FP"), fn.assign(error_type="FN")])
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Guardado {OUT_CSV} con {len(out)} ejemplos (FP={len(fp)}, FN={len(fn)})")

if __name__ == "__main__":
    main()
