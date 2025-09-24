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
