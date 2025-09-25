# src/llm.py
import os, time, json, re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from joblib import load as joblib_load

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEED = os.getenv("SEED", "42")

META_PATH = Path("artifacts/ml/metadata.json")
PIPE_PATH = Path("artifacts/ml/pipeline.joblib")
TEST_CSV = Path("eval/test.csv")

OOD_TRIGGERS = ["contraseña", "tarjeta de crédito", "pii", "hackear", "exploit"]

def _is_ood(q: str) -> bool:
    return any(t in q.lower() for t in OOD_TRIGGERS)

def _low_confidence(q: str) -> bool:
    return len(q.strip()) < 6

def _load_metadata() -> Dict[str, Any]:
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    except Exception:
        return {}

def _load_test_metrics() -> Dict[str, Any]:
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        if not (TEST_CSV.exists() and PIPE_PATH.exists()):
            return {}
        df = pd.read_csv(TEST_CSV)
        y = df["survived"].astype(int)
        X = df.drop(columns=["survived"])
        pipe = joblib_load(PIPE_PATH)
        proba = pipe.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {"test_f1": float(f1_score(y, pred)), "test_auc": float(roc_auc_score(y, proba))}
    except Exception:
        return {}

def _kb() -> Dict[str, Any]:
    meta = _load_metadata()
    testm = _load_test_metrics()
    features = meta.get("features", {"num": ["age","sibsp","parch","fare"], "cat": ["pclass","sex","embarked"]})
    return {
        "dataset": "Titanic (seaborn)",
        "target": "survived (0/1)",
        "num_features": features.get("num", []),
        "cat_features": features.get("cat", []),
        "preprocessing": "Imputación (mediana/moda), OHE en categóricas, escalado en numéricas",
        "baseline_model": "Logistic Regression",
        "improved_model": "XGBoost (fallback RandomForest)",
        "best_model": meta.get("model_name","baseline"),
        "seed": SEED,
        "valid_metrics": meta.get("valid_metrics", {}),
        "test_metrics": testm,
        "artifacts": {
            "pipeline": "artifacts/ml/pipeline.joblib",
            "metadata": "artifacts/ml/metadata.json",
            "test_csv": "eval/test.csv",
            "confusion_png": "eval/reports/confusion_matrix.png"
        },
        "split": "train/valid/test estratificado con semilla fija (~70/15/15)",
        "metric_primary": "F1 (por desbalance); adicional AUC/ROC",
        "leakage": "Se evita con split y revisión de features",
        "nulls": "age→mediana, embarked→moda; numéricas escaladas",
        "api": {
            "predict": "POST /predict {features:{...}} → {prediction, probability}",
            "llm": "POST /llm/ask {question} → {answer, citations,...}"
        }
    }

def _answer_from_kb(question: str, kb: Dict[str, Any]) -> Dict[str, Any]:
    q = question.strip().lower()
    patterns = [
        (r"\bm[eé]trica\b|\bf1\b|auc|roc", lambda: f"Métrica principal: {kb['metric_primary']}. Valid: {kb['valid_metrics'] or 'N/D'}. Test: {kb['test_metrics'] or 'N/D'}"),
        (r"dataset|conjunto de datos|datos", lambda: f"Dataset: {kb['dataset']}. Target: {kb['target']}."),
        (r"feature|caracter[íi]stica|variables", lambda: f"Numéricas: {', '.join(kb['num_features'])}. Categóricas: {', '.join(kb['cat_features'])}."),
        (r"preproces|imputaci[óo]n|ohe|escal", lambda: f"Preprocesamiento: {kb['preprocessing']}."),
        (r"baseline|modelo base|regresi[óo]n log[íi]stica", lambda: f"Baseline: {kb['baseline_model']}."),
        (r"mejorado|xgboost|random forest|randomforest", lambda: f"Modelo mejorado: {kb['improved_model']}. Mejor seleccionado: {kb['best_model']} (por F1 en valid)."),
        (r"umbral|threshold", lambda: "Umbral actual: 0.5 (puedes optimizarlo por F1 en valid y guardarlo en metadata)."),
        (r"seed|semilla", lambda: f"Seed: {kb['seed']}."),
        (r"split|train|valid|test|estratific", lambda: f"Split: {kb['split']}."),
        (r"leakage|fuga|filtraci[óo]n", lambda: f"Leakage: {kb['leakage']}."),
        (r"nulos|faltantes|missing", lambda: f"Nulos: {kb['nulls']}."),
        (r"guard[a|e]|d[óo]nde.*pipeline|artefact|metadata", lambda: f"Artefactos: pipeline={kb['artifacts']['pipeline']}, metadata={kb['artifacts']['metadata']}."),
        (r"api|endpoint|predict|llm|docs", lambda: f"API: {kb['api']['predict']} ; {kb['api']['llm']}. Swagger en /docs."),
    ]
    for pat, fn in patterns:
        if re.search(pat, q):
            ans = fn()
            return {"answer": ans, "citations":[kb["artifacts"]["metadata"], kb["artifacts"]["test_csv"], kb["artifacts"]["confusion_png"]], "confidence": 0.9}
    return {"answer": "Puedo responder sobre dataset, métricas (F1/AUC), features, baseline/mejorado, split, leakage, nulos, artefactos y API. ¿Qué te gustaría saber?", "citations": [], "confidence": 0.6}

def ask_llm(question: str, mode: str = "prompting", provider: Optional[str] = None, **gen_kwargs) -> Dict[str, Any]:
    t0 = time.time()

    # Guardrails
    if _is_ood(question):
        return {"answer":"Lo siento, no puedo ayudar con esa solicitud.", "citations":[], "latency_ms":int((time.time()-t0)*1000), "token_usage":{"prompt":0,"completion":0,"total":0}, "mode":mode, "confidence":0.0, "guardrail":"OOD"}
    if _low_confidence(question):
        return {"answer":"¿Podrías reformular tu pregunta con más contexto del proyecto?", "citations":[], "latency_ms":int((time.time()-t0)*1000), "token_usage":{"prompt":0,"completion":0,"total":0}, "mode":mode, "confidence":0.3, "guardrail":"LOW_CONFIDENCE"}

    # 1) Intentar OpenAI si hay API key
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            kb = _kb()
            system = "Eres un asistente del proyecto ML+LLM. Responde conciso y factualmente. Si algo no está en el contexto, dilo."
            context = json.dumps(kb, ensure_ascii=False)
            messages = [
                {"role":"system","content": system},
                {"role":"user","content": f"Contexto (KB): {context}\n\nPregunta: {question}"}
            ]
            # usa un modelo económico y rápido
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=300
            )
            answer = resp.choices[0].message.content
            usage_total = getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else 0
            return {
                "answer": answer,
                "citations": [kb["artifacts"]["metadata"]],
                "latency_ms": int((time.time()-t0)*1000),
                "token_usage": {"prompt": 0, "completion": usage_total, "total": usage_total},
                "mode": "openai",
                "confidence": 0.8,
                "guardrail": None
            }
        except Exception as e:
            err = f"(Fallo OpenAI: {type(e).__name__})"

    # 2) KB local (sin red)
    kb = _kb()
    local = _answer_from_kb(question, kb)
    local["latency_ms"] = int((time.time() - t0) * 1000)
    local["token_usage"] = {"prompt": len(question.split()), "completion": len(local["answer"].split()), "total": len(question.split()) + len(local["answer"].split())}
    local["mode"] = "kb"
    local["guardrail"] = None
    return local
