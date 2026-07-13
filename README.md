# 🌊 OceanoIA

Sistema de inteligencia artificial de apoyo a INCOPESCA, MarViva, Guardacostas y pescadores artesanales para identificación de especies marinas, predicción de condiciones oceanográficas y recomendación de zonas de pesca sostenible.

**Colegio Universitario de Cartago — Costa Rica**

## Integrantes

- Isaac Ulloa Calvo
- Jeffrey Jiménez Cordero
- Felipe Montenegro Artavia
- Jefferson Granados Rodríguez

## Descripción

OceanoIA integra dos modelos de deep learning y una capa de reglas de negocio para apoyar decisiones de pesca sostenible en la costa de Costa Rica:

- Un modelo **CNN** que identifica la especie de un pez a partir de una fotografía.
- Un modelo **RNN/LSTM** que pronostica condiciones oceanográficas (oleaje, viento, temperatura del mar, marea y presión) para las próximas 72 horas.
- Un motor de **reglas por umbral** que recomienda si pescar, cambiar de zona o regresar a puerto según las condiciones actuales.

El sistema expone los dos modelos mediante una **API REST con FastAPI** y los consume desde una **aplicación web con Streamlit**, que además incluye un mapa interactivo de zonas costeras con Folium.

## Fuente de datos

### Identificación de especies (CNN)

**Dataset:** [A Large Scale Fish Dataset — Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) (`crowww/a-large-scale-fish-dataset`, descargado vía `kagglehub`).

9,000 imágenes balanceadas en 9 clases (1,000 por clase): Black Sea Sprat, Gilt-Head Bream, Hourse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet y Trout.

> **Limitación conocida:** este dataset corresponde a especies de mercados/aguas de Turquía, no a las especies costarricenses mencionadas en el planteamiento original del proyecto (dorado, atún aleta amarilla, pargo mancha, corvina reina, marlín, tortuga marina, tiburón martillo). El modelo **no** puede identificar esas especies. Se documenta como limitación explícita en lugar de ocultarla — ver sección "Uso académico".

### Pronóstico oceanográfico (RNN)

**Fuentes:** [Open-Meteo Marine API](https://marine-api.open-meteo.com) y [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) (sin API key).

Zona de entrenamiento: Golfo de Nicoya, Costa Rica (lat `9.97`, lon `-84.90`). Se descargaron aproximadamente 760 días de historial horario (~2 años) de oleaje, dirección de oleaje, período, temperatura superficial del mar, marea, viento y presión.

Los archivos originales no se versionan en `DATA/RAW` — se descargan directamente desde las APIs al correr los notebooks.

## Modelos predictivos

### CNN — Identificación de especies

Arquitectura secuencial: `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(256) → Dropout(0.5) → Dense(9, softmax)`, entrenada con `ImageDataGenerator` (rotación, desplazamiento, zoom, flip horizontal) sobre imágenes de 224×224.

División 80/20 estratificada (7,200 train / 1,800 test). Resultado sobre el conjunto de test:

| Métrica | Valor |
|---|---|
| Accuracy | 94.9% |
| Loss | 0.139 |

Desarrollo completo en [`NOTEBOOKS/02_CNN_Especies.ipynb`](NOTEBOOKS/02_CNN_Especies.ipynb). Modelo guardado en `NOTEBOOKS/modelo_CNN_espe.keras`.

### RNN/LSTM — Pronóstico oceanográfico

Arquitectura encoder-decoder: `LSTM(128) → LSTM(64) → RepeatVector(72) → LSTM(64)`, con una cabeza `TimeDistributed(Dense)` independiente por cada variable objetivo. Recibe 72 horas de contexto (oleaje, viento, SST, marea, presión, más variables cíclicas de dirección, fase lunar, hora del día y marea semidiurna/lunar) y predice 72 horas hacia adelante.

RMSE / MAE sobre el conjunto de test (30 días separados del entrenamiento):

| Variable | RMSE | MAE |
|---|---|---|
| Oleaje (m) | 0.044 | 0.036 |
| Viento (km/h) | 3.96 | 3.16 |
| SST (°C) | 0.555 | 0.532 |
| Marea (m) | 0.230 | 0.181 |
| Presión (hPa) | 1.74 | 1.46 |

Desarrollo completo en [`NOTEBOOKS/03_RNN_Oceanografia.ipynb`](NOTEBOOKS/03_RNN_Oceanografia.ipynb). Modelo guardado en `MODELS/modelo_forecast_oceanografico.keras`, scaler en `MODELS/scaler_forecast_oceanografico.pkl`.

### Recomendación de pesca

No es un modelo entrenado — es un motor de reglas fijas por umbral (oleaje, viento, veda, área protegida, distancia a costa, temperatura del mar) implementado directamente en la aplicación Streamlit.

## API REST

Desarrollada con FastAPI en [`API/main.py`](API/main.py). Documentación interactiva disponible en `/docs` una vez levantada.

### Endpoint raíz

`GET /`

Lista los endpoints disponibles.

### Pronóstico oceanográfico

`GET /oceanografia/predict?lat=9.97&lon=-84.90`

Descarga las últimas 72h reales de Open-Meteo y devuelve el pronóstico de las próximas 72h.

Ejemplo de respuesta:

```json
{
  "zona": { "lat": 9.97, "lon": -84.9 },
  "generado_desde": "2026-07-12T23:00:00",
  "horizonte_horas": 72,
  "forecast": [
    {
      "timestamp": "2026-07-12T23:00:00",
      "wave_height": 0.163,
      "wind_speed_10m": 6.46,
      "sea_surface_temperature": 30.64,
      "sea_level_height_msl": 0.88,
      "surface_pressure": 1010.6
    }
  ]
}
```

### Identificación de especies

`POST /especies/predict` (multipart form, campo `imagen`)

Recibe una foto (jpg/jpeg/png/webp) y devuelve la especie identificada.

Ejemplo de respuesta:

```json
{
  "especie_en": "Sea Bass",
  "especie_es": "Róbalo Europeo",
  "nombre_cientifico": "Dicentrarchus labrax",
  "confianza": 0.9995,
  "estado_legal": "PLACEHOLDER — verificar regulación local",
  "recomendacion": "PLACEHOLDER — verificar talla mínima y veda antes de publicar",
  "top3": [
    { "especie_en": "Sea Bass", "especie_es": "Róbalo Europeo", "confianza": 0.9995 }
  ]
}
```

`estado_legal` y `recomendacion` son placeholders — ver "Uso académico".

## Aplicación web

Desarrollada con Streamlit en [`APP/Home.py`](APP/Home.py). Consume la API REST y muestra un indicador de conexión en el sidebar.

- **📷 Identificar Especie** — sube o toma una foto, la envía a `/especies/predict` y muestra la especie, confianza y top 3.
- **🌊 Pronóstico Oceánico** — elige una zona costera y consulta `/oceanografia/predict`, con gráficos Plotly de oleaje, viento, SST, marea y presión.
- **🗺️ Mapa de Recomendaciones** — mapa Folium con las zonas costeras de Costa Rica y el motor de reglas de recomendación.
- **📊 Dashboard** — resumen general del proyecto.

## Estructura del proyecto

```text
OceanoIA/
│
├── API/
│   └── main.py
│
├── APP/
│   ├── ASSENTS/
│   └── Home.py
│
├── DATA/
│   ├── PROCESSED/
│   └── RAW/
│
├── MODELS/
│   ├── modelo_forecast_oceanografico.keras
│   └── scaler_forecast_oceanografico.pkl
│
├── NOTEBOOKS/
│   ├── 01_EDA.ipynb
│   ├── 02_CNN_Especies.ipynb
│   ├── 03_RNN_Oceanografia.ipynb
│   ├── modelo_CNN_espe.h5
│   └── modelo_CNN_espe.keras
│
├── SRC/
│   ├── data_prep.py
│   ├── marine_api.py
│   └── TRAIN/
│       ├── cnn.py
│       └── rnn.py
│
├── requirements.txt
└── README.md
```

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Streamlit
- Plotly
- Folium / streamlit-folium
- pandas
- NumPy
- scikit-learn
- OpenCV
- Requests
- Jupyter Notebook

## Instalación

Se recomienda utilizar Python 3.11.

Clonar el repositorio:

```bash
git clone https://github.com/Isaakkko/OCEANOIA.git
cd OCEANOIA
```

Crear el entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Ejecución de la API

Desde la raíz del repositorio:

```bash
python -m uvicorn API.main:app --reload
```

La documentación interactiva de FastAPI estará disponible normalmente en:

`http://127.0.0.1:8000/docs`

## Ejecución de la aplicación Streamlit

La API debe estar corriendo primero. Desde la raíz del repositorio:

```bash
python -m streamlit run APP/Home.py
```

La aplicación estará disponible normalmente en:

`http://localhost:8501`

## Objetivo del proyecto

El objetivo es demostrar la aplicación de redes neuronales convolucionales, redes recurrentes y sistemas de reglas en un caso real de monitoreo costero, integrando fuentes de datos abiertas, un modelo de clasificación de imágenes y un modelo de series temporales dentro de un producto funcional de punta a punta.

## Uso académico

Proyecto desarrollado con fines académicos para el curso de Inteligencia Artificial, CUC. Las predicciones generadas deben interpretarse como resultados de modelos académicos y no como una herramienta oficial de regulación pesquera.

En particular:

- El modelo CNN identifica especies del dataset de Turquía usado para entrenarlo, no las especies protegidas o reguladas de Costa Rica mencionadas en el planteamiento original del proyecto.
- Los campos `estado_legal` y `recomendacion` que devuelve la API son placeholders sin validar contra la regulación vigente de INCOPESCA — no deben usarse como base real para decidir si pescar o devolver una especie al mar.
