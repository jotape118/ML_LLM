"""
Prompting estructurado con guardrails simples.
- ask_llm(question, mode="prompting"): retorna dict con answer, citations, latency_ms, token_usage, mode.
- Si no hay OPENAI_API_KEY, devuelve una respuesta stub (útil p/ tests).
"""
import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente técnico del proyecto ML+LLM. Responde de forma concisa y en JSON cuando se solicite."
)

FEW_SHOTS = [
    # pares (q, a) para guiar estilo; mantener cortos
    ("¿Qué métrica principal usan?", "Usamos F1 como métrica principal por desbalance."),
    ("¿Cómo evitan leakage?", "Separando train/valid/test con semilla y revisando features causales."),
]

OOD_TRIGGERS = ["contraseña", "tarjeta de crédito", "PII", "hackear", "exploit"]

def _is_ood(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in [t.lower() for t in OOD_TRIGGERS])

def _low_confidence(q: str) -> bool:
    return len(q.strip()) < 6  # ejemplo simple

def ask_llm(question: str, mode: str = "prompting", **gen_kwargs) -> Dict[str, Any]:
    t0 = time.time()

    # Guardrails
    if _is_ood(question):
        return {
            "answer": "Lo siento, no puedo ayudar con esa solicitud.",
            "citations": [],
            "latency_ms": int((time.time() - t0) * 1000),
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "mode": mode,
            "confidence": 0.0,
            "guardrail": "OOD",
        }
    if _low_confidence(question):
        return {
            "answer": "Podrías reformular tu pregunta con más contexto del proyecto?",
            "citations": [],
            "latency_ms": int((time.time() - t0) * 1000),
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "mode": mode,
            "confidence": 0.3,
            "guardrail": "LOW_CONFIDENCE",
        }

    # Stub local si no hay API key (evita dependencias para tests)
    if not OPENAI_API_KEY:
        answer = "Este es un stub sin LLM externo. Ejemplo: usamos F1 y guardamos pipeline en artifacts/ml."
        return {
            "answer": answer,
            "citations": [],
            "latency_ms": int((time.time() - t0) * 1000),
            "token_usage": {"prompt": 0, "completion": len(answer.split()), "total": len(answer.split())},
            "mode": mode,
            "confidence": 0.7,
        }

    # Ejemplo de integración (rellenar según tu proveedor)
    # from openai import OpenAI
    # client = OpenAI(api_key=OPENAI_API_KEY)
    # prompt = _compose_prompt(question)
    # resp = client.chat.completions.create(model="gpt-4o-mini", messages=[...], temperature=0.2, max_tokens=300)
    # answer = resp.choices[0].message.content
    # usage = resp.usage.total_tokens
    # return {...}

    # Temporal mientras rellenas integración real:
    answer = "LLM conectado (placeholder). Aquí iría la respuesta generada."
    return {
        "answer": answer,
        "citations": [],  # si usas RAG, añade pasajes
        "latency_ms": int((time.time() - t0) * 1000),
        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        "mode": mode,
        "confidence": 0.8,
    }

def _compose_prompt(q: str) -> str:
    few = "\n".join([f"Q: {a}\nA: {b}" for a, b in FEW_SHOTS])
    return f"{SYSTEM_INSTRUCTIONS}\n\n{few}\n\nUser: {q}\nAssistant:"
