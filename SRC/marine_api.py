"""
marine_api.py
=============
Utilidades para consultar la Marine API de Open-Meteo (sin API key) para las
zonas costeras de Costa Rica usadas en la app de Streamlit (APP/Home.py).
"""

import pandas as pd
import requests

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
MARINE_VARS = ["wave_height", "wave_period", "sea_surface_temperature"]

# Coordenadas aproximadas (lat, lon) de cada zona costera. Sirven para ubicar
# los marcadores del mapa y como punto de consulta a la Marine API — no son
# límites oficiales de zona.
ZONAS_CR = {
    "golfo_nicoya": (9.9, -84.9),
    "golfo_dulce": (8.7, -83.3),
    "pacifico_norte": (10.6, -85.7),
    "pacifico_central": (9.6, -84.6),
    "pacifico_sur": (8.4, -83.3),
    "caribe_norte": (10.5, -83.5),
    "caribe_sur": (9.6, -82.8),
}


def get_zone_forecast(zona: str, days: int = 3) -> pd.DataFrame:
    """Descarga el pronóstico de oleaje/SST de Open-Meteo Marine para una zona de ZONAS_CR."""
    if zona not in ZONAS_CR:
        raise ValueError(f"Zona desconocida: {zona}. Opciones: {list(ZONAS_CR)}")

    lat, lon = ZONAS_CR[zona]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(MARINE_VARS),
        "forecast_days": days,
        "timezone": "auto",
    }
    r = requests.get(MARINE_URL, params=params, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Error {r.status_code} en Marine API: {r.text}")

    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df
