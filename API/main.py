"""
main.py
=======
API REST OceanoIA con FastAPI.

Endpoints:
  - POST /predict/especie  (imagen del pez)
  - POST /predict/oceano   (serie temporal de 30 días)
  - POST /predict/accion   (features tabulares)

Ejecutar:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="OceanoIA API",
    description="API REST para identificación de especies, pronóstico oceánico y recomendación de pesca.",
    version="1.0.0",
)

# ===================== Carga perezosa de modelos =====================
_models = {}

def _load(key: str, path: str):
    """Carga modelos solo cuando se necesitan."""
    if key not in _models:
        import tensorflow as tf
        try:
            _models[key] = tf.keras.models.load_model(path)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo no disponible: {path}. Entrénalo primero. ({e})",
            )
    return _models[key]


# ===================== Health =====================
@app.get("/")
def root():
    return {
        "name": "OceanoIA API",
        "endpoints": ["/predict/especie", "/predict/oceano", "/predict/accion"],
        "status": "ok",
    }


# ===================== CNN: especie =====================
@app.post("/predict/especie")
async def predict_especie(file: UploadFile = File(...)):
    """Recibe una imagen y devuelve la especie con probabilidad."""
    try:
        from PIL import Image
        model = _load("cnn", "models/cnn_especies.keras")

        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128, 128))
        arr = np.expand_dims(np.array(img) / 255.0, 0).astype("float32")

        preds = model.predict(arr, verbose=0)[0]

        clases = [
            "dorado", "atun_aleta_amarilla", "pargo_mancha", "corvina_reina",
            "marlin_pez_vela", "tortuga_marina", "tiburon_martillo", "otros",
        ]
        idx = int(np.argmax(preds))

        return {
            "especie":    clases[idx],
            "confianza":  float(preds[idx]),
            "protegida":  clases[idx] in {"marlin_pez_vela", "tortuga_marina", "tiburon_martillo"},
            "todas":      {c: float(p) for c, p in zip(clases, preds)},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== RNN: pronóstico =====================
class OceanoSeriesIn(BaseModel):
    """30 días × 4 features (wave_height, wave_period, SST, wave_direction) ya escalados."""
    serie: List[List[float]]

@app.post("/predict/oceano")
def predict_oceano(payload: OceanoSeriesIn):
    """Recibe la ventana de 30 días y predice las próximas 24 h."""
    try:
        model = _load("rnn", "models/rnn_oceano.keras")
        arr = np.array(payload.serie, dtype="float32")
        if arr.ndim != 2:
            raise HTTPException(status_code=400, detail="Se esperaba matriz 2D (timesteps, features)")
        arr = np.expand_dims(arr, 0)
        pred = model.predict(arr, verbose=0)[0]
        return {"prediccion_24h": pred.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== ANN: recomendación =====================
class AccionIn(BaseModel):
    altura_oleaje: float
    viento_kmh: float
    sst: float
    fase_lunar: int
    dist_costa: float
    mes: int
    zona_amp: int
    especie: str
    veda: int

@app.post("/predict/accion")
def predict_accion(payload: AccionIn):
    """Recomienda la acción óptima del pescador."""
    try:
        import joblib
        import pandas as pd

        model = _load("ann", "models/ann_recomendacion.keras")
        scaler     = joblib.load("models/ann_scaler.pkl")
        target_enc = joblib.load("models/ann_target_encoder.pkl")

        with open("models/ann_features.txt") as f:
            cols = f.read().splitlines()

        df = pd.DataFrame([payload.dict()])
        df = pd.get_dummies(df, columns=["especie"], prefix="esp")
        df = df.reindex(columns=cols, fill_value=0)
        X = scaler.transform(df.values.astype("float32"))

        pred = model.predict(X, verbose=0)[0]
        idx = int(np.argmax(pred))
        return {
            "recomendacion": str(target_enc.classes_[idx]),
            "confianza":     float(pred[idx]),
            "todas":         {c: float(p) for c, p in zip(target_enc.classes_, pred)},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
