"""
Descarga/prepara el dataset. Por defecto usa Titanic de seaborn (liviano y sin claves).
Si quieres Kaggle: añade lógica con kaggle API y guarda en data/raw/.
"""
import os
from pathlib import Path
import pandas as pd
import seaborn as sns

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Titanic vía seaborn (columns: survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, etc.)
    df = sns.load_dataset("titanic")
    # Normaliza nombres y selecciona un subconjunto compatible con un modelo clásico
    cols = ["survived","pclass","sex","age","sibsp","parch","fare","embarked"]
    df = df[cols].dropna(subset=["survived"])  # target no nulo
    # Guardar CSV procesado simple
    out_csv = PROCESSED_DIR / "titanic.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Dataset guardado en {out_csv} (shape={df.shape})")

if __name__ == "__main__":
    main()
