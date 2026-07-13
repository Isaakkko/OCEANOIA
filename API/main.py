"""
API de Predicción Oceanográfica y Clasificación de Especies
=============================================================
Todo en un solo archivo: carga de modelos, preprocesamiento, y los dos endpoints.

Modelos que necesita, en la carpeta models/ (junto a este archivo):
    - modelo_forecast_oceanografico.keras
    - scaler_forecast_oceanografico.pkl
    - modelo_CNN_espe.keras

Correr con (desde la raíz del repo):
    uvicorn API.main:app --reload
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from tensorflow.keras.models import load_model


# =====================================================================
# CONFIGURACIÓN GENERAL
# =====================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "MODELS")
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "NOTEBOOKS")

RNN_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_forecast_oceanografico.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_forecast_oceanografico.pkl")
CNN_MODEL_PATH = os.path.join(NOTEBOOKS_DIR, "modelo_CNN_espe.keras")

# Variables que usa el modelo oceanográfico. El orden importa: debe coincidir
# exactamente con el orden usado al entrenar, o el scaler/modelo van a recibir
# las columnas mezcladas y las predicciones no van a tener sentido.
TARGET_COLS = [
    "wave_height",
    "wind_speed_10m",
    "sea_surface_temperature",
    "sea_level_height_msl",
    "surface_pressure",
]

FEATURE_COLS = [
    "wave_height", "wave_period", "sea_surface_temperature", "sea_level_height_msl",
    "wind_speed_10m", "surface_pressure",
    "wave_dir_sin", "wave_dir_cos", "wind_dir_sin", "wind_dir_cos",
    "moon_phase_sin", "moon_phase_cos",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "tide_semidiurnal_sin", "tide_semidiurnal_cos", "tide_lunarday_sin", "tide_lunarday_cos",
]

TARGET_IDX = [FEATURE_COLS.index(c) for c in TARGET_COLS]

INPUT_HOURS = 72
OUTPUT_HOURS = 72

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
MARINE_VARS = ["wave_height", "wave_direction", "wave_period", "sea_surface_temperature", "sea_level_height_msl"]

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"  # API en vivo: sirve para datos recientes
WEATHER_VARS = ["wind_speed_10m", "wind_direction_10m", "surface_pressure"]

SYNODIC_MONTH = 29.530588853
REFERENCE_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)

IMAGE_SIZE = (224, 224)  # tamaño usado al entrenar la CNN


# =====================================================================
# TABLA DE ESPECIES: traducción + estado legal + recomendación
# =====================================================================

CLASES_ORDENADAS = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout",
]
# IMPORTANTE
# la gran mayoría no existen de forma silvestre en las costas de Costa Rica por ser propias del mar Mediterráneo, el Atlántico europeo o zonas templadas.

ESPECIES_INFO = {
    "Black Sea Sprat": {
        "nombre_es": "Espadín del Mar Negro", "nombre_cientifico": "Clupeonella cultriventris",
        "estado_legal": "pesca deportiva y con devolución obligatoria",
        "recomendacion": "Extraerlos del agua para consumo o comercialización está prohibido por ley",
    },
    "Gilt-Head Bream": {
        "nombre_es": "Dorada", "nombre_cientifico": "Sparus aurata",
        "estado_legal": "Pezclable con licencia de pesca obligatoria",
        "recomendacion": "Pezcar en Pacífico Norte, Central y Sur",
    },
    "Hourse Mackerel": {
        "nombre_es": "Chicharro / Jurel", "nombre_cientifico": "Trachurus trachurus",
        "estado_legal": "No existe en las aguas de Costa Rica",
        "recomendacion": "Especie considerada de consumo, prohibido vender los filetes si fueron capturados con licencia deportiva",
    },
    "Red Mullet": {
        "nombre_es": "Salmonete de Roca", "nombre_cientifico": "Mullus barbatus",
        "estado_legal": "No existe en las aguas de Costa Rica",
        "recomendacion": "Totalmente legal.",
    },
    "Red Sea Bream": {
        "nombre_es": "Besugo", "nombre_cientifico": "Pagellus bogaraveo",
        "estado_legal": "No existe en las aguas de Costa Rica",
        "recomendacion": "Totalmente legal.",
    },
    "Sea Bass": {
        "nombre_es": "Róbalo Europeo", "nombre_cientifico": "Dicentrarchus labrax",
        "estado_legal": "No existe en las aguas de Costa Rica",
        "recomendacion": "Generalmente no se permite retener ejemplares menores a los 50-60 centímetros",
    },
    "Shrimp": {
        "nombre_es": "Camarón", "nombre_cientifico": "Penaeus spp.",
        "estado_legal": " legal únicamente bajo regulaciones comerciales estrictas(INCOPESCA)",
        "recomendacion": "Pezcar en Pacífico Central, Sur y Caribe Norte",
    },
    "Striped Red Mullet": {
        "nombre_es": "Salmonete de Fango", "nombre_cientifico": "Mullus surmuletus",
        "estado_legal": "No existe en las aguas de Costa Rica",
        "recomendacion": "Totalmente legal.",
    },
    "Trout": {
        "nombre_es": "Trucha", "nombre_cientifico": "Oncorhynchus mykiss / Salmo trutta",
        "estado_legal": "5 piezas por persona con una longitud mínima de 25 cm",
        "recomendacion": " Arroyos y ríos de Cerro de la Muerte, especialmente en la zona de San Gerardo de Dota",
    },
}


def get_info_especie(nombre_clase: str) -> dict:
    return ESPECIES_INFO.get(nombre_clase, {
        "nombre_es": nombre_clase, "nombre_cientifico": "Desconocido",
        "estado_legal": "Sin información registrada", "recomendacion": "Sin información registrada",
    })


# =====================================================================
# MODELOS EN MEMORIA (se cargan una sola vez, al arrancar el servidor)
# =====================================================================

_rnn_model = None
_scaler = None
_cnn_model = None


def cargar_modelos():
    global _rnn_model, _scaler, _cnn_model

    if not os.path.exists(RNN_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo oceanográfico en {RNN_MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"No se encontró el scaler en {SCALER_PATH}")
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo de especies en {CNN_MODEL_PATH}")

    print("Cargando modelo oceanográfico (RNN)...")
    _rnn_model = load_model(RNN_MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

    print("Cargando modelo de especies (CNN)...")
    _cnn_model = load_model(CNN_MODEL_PATH)

    print("Modelos cargados. Listo para recibir peticiones.")


# =====================================================================
# LÓGICA DEL MODELO OCEANOGRÁFICO (RNN)
# =====================================================================

def _fetch_marine(lat, lon, past_days=5, forecast_days=1, timezone_="auto"):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(MARINE_VARS),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": timezone_,
    }
    r = requests.get(MARINE_URL, params=params, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Error {r.status_code} en Marine API: {r.text}")
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _fetch_weather(lat, lon, past_days=5, forecast_days=1, timezone_="auto"):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": timezone_,
    }
    r = requests.get(WEATHER_URL, params=params, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Error {r.status_code} en Weather API: {r.text}")
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _moon_phase_fraction(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days_since = (dt - REFERENCE_NEW_MOON).total_seconds() / 86400.0
    return (days_since % SYNODIC_MONTH) / SYNODIC_MONTH


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replica exactamente el preprocesamiento del notebook de entrenamiento."""
    df = df.copy()

    df["wave_dir_sin"] = np.sin(np.deg2rad(df["wave_direction"]))
    df["wave_dir_cos"] = np.cos(np.deg2rad(df["wave_direction"]))
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction_10m"]))
    df = df.drop(columns=["wave_direction", "wind_direction_10m"])

    phase = df.index.to_series().apply(_moon_phase_fraction)
    df["moon_phase_sin"] = np.sin(2 * np.pi * phase)
    df["moon_phase_cos"] = np.cos(2 * np.pi * phase)

    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    hours_elapsed = (df.index - df.index[0]).total_seconds() / 3600.0
    df["tide_semidiurnal_sin"] = np.sin(2 * np.pi * hours_elapsed / 12.4206)
    df["tide_semidiurnal_cos"] = np.cos(2 * np.pi * hours_elapsed / 12.4206)
    df["tide_lunarday_sin"] = np.sin(2 * np.pi * hours_elapsed / 24.8412)
    df["tide_lunarday_cos"] = np.cos(2 * np.pi * hours_elapsed / 24.8412)

    return df.interpolate(method="time").ffill().bfill()


def _inverse_transform(scaled_targets, scaler, target_idx, n_features):
    n_samples, out_h, n_t = scaled_targets.shape
    flat = scaled_targets.reshape(-1, n_t)
    dummy = np.zeros((flat.shape[0], n_features))
    dummy[:, target_idx] = flat
    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(n_samples, out_h, n_t)


def predecir_oceanografia(lat: float, lon: float, timezone_: str = "auto") -> dict:
    df_marine = _fetch_marine(lat, lon, timezone_=timezone_)
    df_weather = _fetch_weather(lat, lon, timezone_=timezone_)
    df = df_marine.join(df_weather, how="inner")
    df = _build_features(df)

    ahora = pd.Timestamp.now(tz=df.index.tz).floor("h") if df.index.tz else pd.Timestamp.now().floor("h")
    df_pasado = df[df.index <= ahora]

    if len(df_pasado) < INPUT_HOURS:
        raise ValueError(
            f"Solo hay {len(df_pasado)} horas de datos disponibles antes de ahora, "
            f"se necesitan {INPUT_HOURS}. Prueba con otra zona o revisa la cobertura de Open-Meteo ahí."
        )

    df_recientes = df_pasado.tail(INPUT_HOURS)

    ultimo_batch = df_recientes[FEATURE_COLS].values
    ultimo_batch_scaled = _scaler.transform(ultimo_batch)
    ultimo_batch_scaled = ultimo_batch_scaled.reshape((1, INPUT_HOURS, len(FEATURE_COLS)))

    pred_scaled_list = _rnn_model.predict(ultimo_batch_scaled, verbose=0)
    pred_scaled = np.concatenate(pred_scaled_list, axis=-1)
    pred_real = _inverse_transform(pred_scaled, _scaler, TARGET_IDX, len(FEATURE_COLS))[0]

    ultima_hora = df_recientes.index[-1]
    horas_forecast = pd.date_range(start=ultima_hora + pd.Timedelta(hours=1), periods=OUTPUT_HOURS, freq="h")

    forecast = []
    for i, ts in enumerate(horas_forecast):
        punto = {"timestamp": ts.isoformat()}
        for j, col in enumerate(TARGET_COLS):
            punto[col] = round(float(pred_real[i, j]), 4)
        forecast.append(punto)

    return {
        "zona": {"lat": lat, "lon": lon},
        "generado_desde": ultima_hora.isoformat(),
        "horizonte_horas": OUTPUT_HOURS,
        "forecast": forecast,
    }


# =====================================================================
# LÓGICA DEL MODELO DE ESPECIES (CNN)
# =====================================================================

def preprocesar_imagen(image_bytes: bytes) -> np.ndarray:
    """
    el modelo se entrenó con cv2.imread(), que lee en BGR (no RGB),
    y nunca se convirtió a RGB antes de entrenar. Por eso aquí también usamos cv2
    para mantener el mismo orden de canales.

    si se usara RGB el modelo vería los colores "invertidos" respecto a lo que aprendió.

    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    imagen = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # decodifica en BGR
    if imagen is None:
        raise ValueError("No se pudo leer la imagen. Verifica que el archivo sea una imagen válida (jpg, png, etc.)")

    imagen = cv2.resize(imagen, IMAGE_SIZE)
    imagen = imagen.astype("float32") / 255.0
    return np.expand_dims(imagen, axis=0)  # (1, 224, 224, 3)


def predecir_especie(image_bytes: bytes) -> dict:
    imagen = preprocesar_imagen(image_bytes)

    probabilidades = _cnn_model.predict(imagen, verbose=0)[0]
    idx_predicho = int(np.argmax(probabilidades))
    clase_predicha = CLASES_ORDENADAS[idx_predicho]
    confianza = float(probabilidades[idx_predicho])
    info = get_info_especie(clase_predicha)

    top3_idx = np.argsort(probabilidades)[::-1][:3]
    top3 = [
        {"especie_en": CLASES_ORDENADAS[i], "especie_es": get_info_especie(CLASES_ORDENADAS[i])["nombre_es"],
         "confianza": round(float(probabilidades[i]), 4)}
        for i in top3_idx
    ]

    return {
        "especie_en": clase_predicha,
        "especie_es": info["nombre_es"],
        "nombre_cientifico": info["nombre_cientifico"],
        "confianza": round(confianza, 4),
        "estado_legal": info["estado_legal"],
        "recomendacion": info.get("recomendacion", "Sin información registrada"),
        "top3": top3,
    }


# =====================================================================
# APLICACIÓN FASTAPI
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelos()  # se ejecuta una sola vez, al arrancar el servidor
    yield


app = FastAPI(
    title="API de Predicción Oceanográfica y Clasificación de Especies",
    description="Combina un modelo RNN (pronóstico oceanográfico 72h) y una CNN (clasificación de especies de peces).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "mensaje": "API de Predicción Oceanográfica y Clasificación de Especies",
        "endpoints": {
            "oceanografia": "GET /oceanografia/predict?lat=..&lon=..",
            "especies": "POST /especies/predict (multipart form, campo 'imagen')",
            "documentacion": "/docs",
        },
    }


@app.get("/oceanografia/predict")
def predict_oceanografia(
    lat: float = Query(..., description="Latitud de la zona, ej. 9.97"),
    lon: float = Query(..., description="Longitud de la zona, ej. -84.90"),
    timezone: str = Query("auto", description="Timezone para Open-Meteo, 'auto' detecta según lat/lon"),
):
    """Descarga las últimas 72h reales de datos y devuelve el pronóstico de las próximas 72h."""
    try:
        return predecir_oceanografia(lat, lon, timezone)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Error consultando Open-Meteo: {e}")


CONTENT_TYPES_VALIDOS = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@app.post("/especies/predict")
async def predict_especie(imagen: UploadFile = File(..., description="Foto del pez a identificar")):
    """Predice la especie de la foto y devuelve nombre traducido, estado legal y recomendación."""
    if imagen.content_type not in CONTENT_TYPES_VALIDOS:
        raise HTTPException(status_code=415, detail=f"Tipo de archivo no soportado: {imagen.content_type}")

    image_bytes = await imagen.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío")

    try:
        return predecir_especie(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
