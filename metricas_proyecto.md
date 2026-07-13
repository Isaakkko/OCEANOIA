# OceanoIA — Contexto y métricas originales del proyecto

> Este documento resume la guía original del profesor/rúbrica del curso.
> Es un **encaminado**, no una receta exacta: si el código real del repo se
> desvía de esto, priorizá lo que el código hace de verdad y señalá la
> diferencia en vez de forzar que todo calce.

## Objetivo general

Sistema de apoyo a INCOPESCA, MarViva, Guardacostas y pescadores artesanales
para: identificar especies marinas, predecir condiciones oceanográficas y
recomendar zonas de pesca sostenible.

## Módulo 1 — CNN (identificación de especies)

- Clasificación de imágenes multiclase.
- Dataset sugerido: Large-Scale Fish Dataset (Kaggle, +9,000 imágenes).
- Métrica objetivo: Accuracy / F1-score, umbral sugerido ≥ 90%.
- Clases: dorado, atún aleta amarilla, pargo mancha, corvina reina,
  marlín/pez vela (protegida), tortuga marina (protegida), tiburón martillo
  (veda).
- Arquitectura sugerida: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool →
  Conv2D(128) → Flatten → Dense(256) → Dropout(0.5) → Softmax.

## Módulo 2 — RNN/LSTM (predicción oceanográfica)

- Series temporales multivariadas: oleaje, viento, temperatura superficial
  del mar, marea, presión, fase lunar.
- Fuentes sugeridas: Open-Meteo Marine API (sin API key), NOAA ERDDAP /
  Copernicus Marine, IMN Costa Rica.
- Objetivo: predecir condiciones de las próximas 24–72h por zona.
- Métrica objetivo: RMSE y MAE, ventana de 30 días.
- Arquitectura sugerida: input t-30…t → LSTM(100) → LSTM(50) →
  Dropout(0.2) → Dense(72).

## Estructura de carpetas sugerida (referencia, no obligatoria)

```
OceanoIA/
├── README.md
├── requirements.txt
├── data/ (raw/, processed/)
├── notebooks/ (01_EDA, 02_CNN_Especies, 03_RNN_Oceanografia, 04_ANN_Recomendacion)
├── src/ (data_prep.py, marine_api.py, train/cnn.py, train/rnn.py, train/ann.py)
├── models/ (.h5 / .keras)
├── api/main.py (FastAPI, opcional)
└── app/Home.py (Streamlit)
```

## Stack tecnológico esperado

Python, TensorFlow/Keras, OpenCV/Pillow, Matplotlib/Seaborn/Plotly, Folium
(mapa interactivo), Streamlit o Gradio, FastAPI (opcional), Docker (extra).

## Entregables esperados

- README.md con propósito, instalación y uso.
- `/notebooks`: EDA + entrenamiento documentado de CNN y RNN.
- `/models`: modelos entrenados (.h5 / .keras).
- `/app`: interfaz Streamlit — identificador de especie, pronóstico, mapa,
  dashboard.
- `/api` (opcional): endpoints `/predict/especie`, `/predict/oceano`,
  `/predict/accion`.
- Informe técnico con diseño, resultados, métricas y conclusiones.
- Código fuente en GitHub.

## Rúbrica

| Componente     | % | Qué evalúa |
|---|---|---|
| Modelo          | 40% | Accuracy, RMSE, F1 sobre datos reales |
| Producto        | 30% | App funcional, demo en vivo |
| Documentación   | 20% | Informe técnico y README |
| Innovación      | 10% | Funcionalidades extra |

**Fecha de entrega: 13 de julio.**

## Ideas de innovación (10%) — opcionales, no obligatorias

- Integración en tiempo real con Open-Meteo Marine API.
- Alerta de especie protegida ("devolver al mar" si detecta tortuga/tiburón).
- Sincronización con vedas vigentes de INCOPESCA.
- Bot de Telegram/WhatsApp (extra, no core).

---

**Nota para Claude Code:** priorizá que lo que ya existe en el repo funcione
de punta a punta (aunque sea con métricas modestas) antes que perseguir el
umbral ideal (ej. accuracy ≥ 90%) si el tiempo no alcanza. Un producto
funcional y honesto sobre sus limitaciones vale más, dado el tiempo
disponible, que perseguir un número que no se llega a validar bien.
