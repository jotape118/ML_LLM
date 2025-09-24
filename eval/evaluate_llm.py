"""
Evaluación automática simple del LLM:
- Exact-match/F1 sobre un pequeño set de Q&A (ground truth).
- Si luego haces RAG, puedes medir "answer contains citation span".
"""
from typing import List, Tuple
from src.llm import ask_llm

QA: List[Tuple[str, str]] = [
    ("¿Qué métrica principal usan para el modelo?", "F1"),
    ("¿Dónde se guarda el pipeline?", "artifacts/ml"),
]

def normalize(s: str) -> str:
    return s.strip().lower()

def main():
    correct = 0
    for q, gold in QA:
        res = ask_llm(q, mode="prompting")
        ans = normalize(res.get("answer", ""))
        if normalize(gold) in ans:
            correct += 1
        print(f"Q: {q}\nA: {res.get('answer')}\nGOLD: {gold}\n---")
    acc = correct / len(QA)
    print(f"Auto-EM accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
