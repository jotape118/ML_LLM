"""
Preprocesamiento y features para Titanic.
- Imputación numérica
- One-hot en categóricas
- Escalado en numéricas
"""
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUM_FEATURES: List[str] = ["age", "sibsp", "parch", "fare"]
CAT_FEATURES: List[str] = ["pclass", "sex", "embarked"] 

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_FEATURES),
            ("cat", cat_pipe, CAT_FEATURES),
        ]
    )
    return pre

from sklearn.pipeline import Pipeline 
