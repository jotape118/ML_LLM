# LLM Report

## Modo
- **Prompting estructurado** (RAG opcional: pendiente/añadir).

## Prompt(s)
- Instrucciones, formato de salida, few-shot (en `src/llm.py`).

## Evaluación
- Automática: exact-match / similitud embeddings en `eval/evaluate_llm.py`.
- Mini human eval (10 Q&A): criterios (Relevancia, Corrección, Trazabilidad, Estilo). Resultados resumidos.

## Guardrails
- Rechazo OOD, umbral de confianza, formato controlado.

## Costos y latencia
- Si se usa API externa, estimación por pregunta (tokens) y latencia promedio.

## Seguridad
- Manejo de PII (no subir datos sensibles), prevención básica de jailbreaks (instrucciones negativas).


## Mini Human Evaluation

| Pregunta | Respuesta esperada | Respuesta obtenida |
|----------|------------------- |--------------------|
|¿Qué métrica principal usan para el modelo?|F1|Métrica principal: F1 (por desbalance); adicional AUC/ROC. Valid: {'f1': 0.7959183673469388, 'auc': 0.8932926829268293}. Test: {'test_f1': 0.66, 'test_auc': 0.8048665249232223}|
|¿Dónde se guarda el pipeline entrenado?|artifacts/ml/pipeline.joblib|Artefactos: pipeline=artifacts/ml/pipeline.joblib, metadata=artifacts/ml/metadata.json.|
|¿Qué dataset se utilizó para entrenar el modelo?| Titanic (seaborn)| Dataset: Titanic (seaborn). Target: survived (0/1). |
|¿Qué features categóricas se usan en el pipeline?| pclass, sex, embarked | Numéricas: age, sibsp, parch, fare. Categóricas: pclass, sex, embarked. |
|¿Qué modelo baseline se probó?| Logistic Regression | Baseline: Logistic Regression. |

### Latencia
- Latencia promedio medida en 5 queries (modo KB local): ~2.4 segundos.
- Este valor corresponde al tiempo de procesamiento local (lectura de archivos y reglas).
- Con un proveedor externo como OpenAI (gpt-4o-mini) la latencia promedio está entre 200–500 ms.

### Tokens
- En modo KB local, el sistema simuló 9 tokens (6 de entrada + 3 de salida).
- En modo OpenAI, el conteo es más preciso y suele estar entre 50–150 tokens por pregunta.

### Costo
- Modo KB local: no hay costo.
- Modo OpenAI (gpt-4o-mini):
  - Precio: $0.15 por millón de tokens.
  - Tokens promedio por query: ~100.
  - Costo por query: ~$0.000015 USD.
  - Costo por 1000 queries: ~$0.015 USD.