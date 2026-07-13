"""
Home.py
=======
App Streamlit de OceanoIA con cuatro pestañas:
  1. 📷 Identificador de especies (CNN, vía API)
  2. 🌊 Pronóstico oceánico (RNN, vía API)
  3. 🗺️ Mapa de recomendaciones (reglas + Folium)
  4. 📊 Dashboard combinado

Ejecutar:
    streamlit run APP/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from SRC.marine_api import ZONAS_CR

# ===================== Config general =====================
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OceanoIA · Pesca Sostenible",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def llamar_api(metodo: str, path: str, **kwargs):
    """Llama a la API y devuelve (response, None) o (None, mensaje_de_error).

    No lanza excepciones: los tabs solo tienen que chequear si error is None.
    """
    kwargs.setdefault("timeout", 30)
    try:
        resp = requests.request(metodo, f"{API_BASE_URL}{path}", **kwargs)
        return resp, None
    except requests.exceptions.ConnectionError:
        return None, (
            f"🔌 No se pudo conectar con la API en `{API_BASE_URL}`. "
            "¿Está corriendo `uvicorn API.main:app` desde la raíz del repo?"
        )
    except requests.exceptions.Timeout:
        return None, f"⏱️ La API en `{API_BASE_URL}` tardó demasiado en responder. Probá de nuevo."
    except requests.exceptions.RequestException as e:
        return None, f"⚠️ Error inesperado al conectar con la API: {e}"


def mostrar_error_api(resp: requests.Response) -> None:
    """Muestra el detalle de un error HTTP de la API (4xx/5xx) de forma legible."""
    try:
        detalle = resp.json().get("detail", resp.text)
    except ValueError:
        detalle = resp.text
    st.error(f"Error {resp.status_code} de la API: {detalle}")


@st.cache_data(ttl=10, show_spinner=False)
def api_esta_arriba() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/", timeout=2).ok
    except requests.exceptions.RequestException:
        return False


# ===================== Sidebar: estado de la API =====================
if api_esta_arriba():
    st.sidebar.success(f"🟢 API conectada\n\n`{API_BASE_URL}`")
else:
    st.sidebar.error(
        f"🔴 API no disponible\n\n`{API_BASE_URL}`\n\n"
        "Los tabs de identificación de especie y pronóstico no van a funcionar "
        "hasta que corras `uvicorn API.main:app` desde la raíz del repo."
    )

# ===================== Header principal =====================
st.title("🌊 OceanoIA")
st.markdown(
    "#### Asistente Inteligente para **Monitoreo Costero y Pesca Sostenible**"
)
st.markdown("---")

# ===================== Tabs =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Identificar Especie",
    "🌊 Pronóstico Oceánico",
    "🗺️ Mapa de Recomendaciones",
    "📊 Dashboard",
])

# ===================== TAB 1: CNN Identificación =====================
with tab1:
    st.header("📷 Identificación de especies marinas (CNN)")
    st.markdown(
        "Subí una foto o tomala con la cámara. El modelo identifica la especie "
        "más parecida entre las que conoce."
    )

    metodo = st.radio("Método de captura", ["Subir imagen", "Tomar foto"], horizontal=True)
    if metodo == "Subir imagen":
        imagen_subida = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    else:
        imagen_subida = st.camera_input("Tomá una foto del pez")

    if imagen_subida is not None:
        col_img, col_res = st.columns(2)
        with col_img:
            st.image(imagen_subida, caption="Imagen a analizar", use_container_width=True)

        with col_res:
            if st.button("🔍 Identificar especie"):
                with st.spinner("Consultando el modelo..."):
                    resp, error = llamar_api(
                        "POST", "/especies/predict",
                        files={"imagen": (imagen_subida.name, imagen_subida.getvalue(), imagen_subida.type)},
                        timeout=30,
                    )

                if error is not None:
                    st.error(error)
                else:
                    if resp.ok:
                        data = resp.json()

                        st.success(f"**Especie detectada:** {data['especie_es']}")
                        st.caption(f"*{data['nombre_cientifico']}* ({data['especie_en']})")
                        st.metric("Confianza", f"{data['confianza'] * 100:.1f}%")

                        st.warning(
                            "⚠️ **Estado legal y recomendación — dato PLACEHOLDER, sin validar.** "
                            "El modelo solo predice la especie; el texto de abajo todavía no "
                            "está verificado contra la regulación real de INCOPESCA. No lo uses "
                            "como base para decidir si pescar o devolver el pez.\n\n"
                            f"- **Estado legal (placeholder):** {data['estado_legal']}\n"
                            f"- **Recomendación (placeholder):** {data['recomendacion']}"
                        )

                        st.markdown("**Top 3 especies más probables:**")
                        df_top3 = pd.DataFrame(data["top3"]).rename(
                            columns={"especie_es": "Especie", "confianza": "Confianza"}
                        )
                        st.bar_chart(df_top3.set_index("Especie")["Confianza"])
                    else:
                        mostrar_error_api(resp)

# ===================== TAB 2: RNN Pronóstico =====================
VARIABLES_FORECAST = [
    ("wave_height", "Oleaje (m)"),
    ("wind_speed_10m", "Viento (km/h)"),
    ("sea_surface_temperature", "SST (°C)"),
    ("sea_level_height_msl", "Marea (m)"),
    ("surface_pressure", "Presión (hPa)"),
]

with tab2:
    st.header("🌊 Pronóstico oceánico")
    st.markdown(
        "Predicción de oleaje, viento, temperatura del mar, marea y presión con un modelo LSTM. "
        "El modelo siempre parte de las últimas 72h reales disponibles y pronostica hacia adelante "
        "desde ahí — no admite elegir una fecha pasada arbitraria."
    )

    col_zona, col_horiz = st.columns(2)
    with col_zona:
        zona = st.selectbox(
            "Zona costera",
            options=list(ZONAS_CR.keys()),
            format_func=lambda z: z.replace("_", " ").title(),
        )
    with col_horiz:
        horizonte = st.radio("Horizonte a mostrar", [24, 48, 72], index=2, horizontal=True,
                              format_func=lambda h: f"{h}h")

    if st.button("🌊 Obtener pronóstico"):
        lat, lon = ZONAS_CR[zona]
        with st.spinner("Consultando el modelo..."):
            resp, error = llamar_api(
                "GET", "/oceanografia/predict",
                params={"lat": lat, "lon": lon, "timezone": "auto"},
                timeout=60,
            )

        if error is not None:
            st.error(error)
        else:
            if resp.ok:
                data = resp.json()
                st.caption(f"Generado desde: {data['generado_desde']} · lat={lat}, lon={lon}")

                df = pd.DataFrame(data["forecast"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").iloc[:horizonte]

                col1, col2, col3 = st.columns(3)
                col1.metric(f"Oleaje máx. {horizonte}h", f"{df['wave_height'].max():.2f} m")
                col2.metric(f"SST promedio {horizonte}h", f"{df['sea_surface_temperature'].mean():.1f} °C")
                col3.metric(f"Viento máx. {horizonte}h", f"{df['wind_speed_10m'].max():.1f} km/h")

                fig = make_subplots(
                    rows=len(VARIABLES_FORECAST), cols=1, shared_xaxes=True,
                    subplot_titles=[label for _, label in VARIABLES_FORECAST],
                    vertical_spacing=0.05,
                )
                for i, (col, label) in enumerate(VARIABLES_FORECAST, start=1):
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df[col], mode="lines", name=label),
                        row=i, col=1,
                    )
                fig.update_layout(
                    height=200 * len(VARIABLES_FORECAST),
                    showlegend=False,
                    title=f"Pronóstico — {zona.replace('_', ' ').title()} ({horizonte}h)",
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Ver datos crudos"):
                    st.dataframe(df)
            else:
                mostrar_error_api(resp)

# ===================== TAB 3: Mapa + recomendación =====================
with tab3:
    st.header("🗺️ Recomendación de pesca + mapa interactivo")
    st.caption(
        "La recomendación usa reglas fijas por umbral, no un modelo entrenado — "
        "todavía no existe un endpoint de recomendación en la API."
    )

    col_reglas, col_mapa = st.columns([1, 2])

    with col_reglas:
        st.markdown("**Ingresá las condiciones actuales:**")
        altura = st.slider("Altura del oleaje (m)", 0.0, 5.0, 1.5, 0.1)
        viento = st.slider("Viento (km/h)", 0, 80, 20)
        sst = st.slider("Temperatura del mar (°C)", 22.0, 32.0, 28.0, 0.1)
        dist = st.slider("Distancia a costa (km)", 1, 80, 20)
        especie = st.selectbox("Especie objetivo", ["dorado", "atun", "pargo", "corvina", "otro"])
        veda = st.checkbox("¿Hay veda activa?")
        amp = st.checkbox("¿Es Área Marina Protegida?")

        if st.button("Obtener recomendación"):
            if altura > 2.5 or viento > 35:
                rec, color, msg = "REGRESAR A PUERTO", "⚫", "Alerta meteorológica."
            elif veda or amp:
                rec, color, msg = "NO PESCAR", "🔴", "Veda activa o área protegida."
            elif altura > 1.8 or viento > 25:
                rec, color, msg = "PESCA CON PRECAUCIÓN", "🟡", "Restricciones leves."
            elif dist > 60:
                rec, color, msg = "CAMBIAR ZONA", "🔵", "Hay zona alternativa más cercana."
            elif 26 <= sst <= 30 and altura < 1.5:
                rec, color, msg = "PESCA RECOMENDADA", "🟢", "Condiciones óptimas."
            else:
                rec, color, msg = "PESCA CON PRECAUCIÓN", "🟡", "Condiciones aceptables."

            st.markdown(f"### {color} **{rec}**")
            st.info(msg)

    with col_mapa:
        mapa = folium.Map(location=[9.7, -84.0], zoom_start=7, tiles="CartoDB positron")
        for nombre, (lat, lon) in ZONAS_CR.items():
            folium.Marker(
                [lat, lon],
                popup=f"<b>{nombre.replace('_', ' ').title()}</b>",
                icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
            ).add_to(mapa)
        st_folium(mapa, width=700, height=500)

# ===================== TAB 4: Dashboard =====================
with tab4:
    st.header("📊 Dashboard integrado")
