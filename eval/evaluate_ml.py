"""
Evalúa el modelo guardado sobre el test set persistido en train.
- Imprime F1/AUC, genera (opcional) matriz de confusión/ROC a PNG.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from joblib import load

TEST_CSV = Path("eval/test.csv")
PIPE_PATH = Path("artifacts/ml/pipeline.joblib")
REPORTS = Path("eval/reports")
REPORTS.mkdir(parents=True, exist_ok=True)

def main():
    if not TEST_CSV.exists():
        raise FileNotFoundError("Falta eval/test.csv. Corre train primero.")
    if not PIPE_PATH.exists():
        raise FileNotFoundError("Falta artifacts/ml/pipeline.joblib. Corre train primero.")
    df = pd.read_csv(TEST_CSV)
    y = df["survived"].astype(int)
    X = df.drop(columns=["survived"])

    pipe = load(PIPE_PATH)
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    f1 = f1_score(y, pred)
    auc = roc_auc_score(y, proba)

    print(f"F1={f1:.4f}  AUC={auc:.4f}")

    # Guardar matriz de confusión rápida
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots()
    ax.matshow(cm)
    ax.set_title("Confusion Matrix")
    for (i, j), v in zip([(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])], cm.flatten()):
        ax.text(j, i, str(v), ha='center', va='center')
    plt.savefig(REPORTS / "confusion_matrix.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
