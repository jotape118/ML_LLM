# ML + LLM End-to-End (Prueba Técnica)

Proyecto compacto que cumple: (A) ML supervisado con EDA, (B) módulo LLM (prompting; RAG opcional), (C) API con FastAPI, (D) documentación y evaluación.

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`
- Asegurarse de tenerl el `.env` y (opcional) poner tu `OPENAI_API_KEY`.

## Flujo recomendado
1) `python data/get_data.py`  # descarga/prepara dataset
2) `python -m src.train`      # entrena, guarda pipeline en `artifacts/`
3) `python -m eval.evaluate_ml`  # métricas F1/AUC + matriz de confusión
4) `uvicorn api.main:app --reload`  # API: POST /predict y POST /llm/ask
5) `python -m eval.evaluate_llm` # evaluación automática simple de LLM
6) Completar `MODEL_CARD.md` y `LLM_REPORT.md`.

## API rápida
- `POST /predict`
```json
{ "features": { "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S" } }
