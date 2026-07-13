"""
Home.py
=======
App Streamlit de OceanoIA con cuatro pestañas:
  1. 📷 Identificador de especies (CNN, vía API)
  2. 🌊 Pronóstico oceánico (RNN, vía API)
  3. 🗺️ Mapa de recomendaciones (reglas + Folium)
  4. 📊 Dashboard combinado

Ejecutar (desde la raíz del repo):
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
    """
    Llama a la API y devuelve (response, None) o (None, mensaje_de_error).
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
    """
    Muestra el detalle de un error HTTP de la API de forma legible.
    """
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


@st.cache_data(ttl=600, show_spinner=False)
def obtener_comparativo_zonas():
    """
    Consulta /oceanografia/predict para las 7 zonas
    """
    filas, errores = [], []
    for zona, (lat, lon) in ZONAS_CR.items():
        resp, error = llamar_api(
            "GET", "/oceanografia/predict",
            params={"lat": lat, "lon": lon, "timezone": "auto"},
            timeout=60,
        )
        if error is not None or not resp.ok:
            errores.append(zona)
            continue
        primero = resp.json()["forecast"][0]
        filas.append({
            "Zona": zona.replace("_", " ").title(),
            "zona_id": zona,
            "lat": lat,
            "lon": lon,
            "Oleaje (m)": primero["wave_height"],
            "Viento (km/h)": primero["wind_speed_10m"],
            "SST (°C)": primero["sea_surface_temperature"],
        })
    return pd.DataFrame(filas), errores


def color_oleaje(m: float) -> str:
    if m > 1.8:
        return "red"
    if m > 1.0:
        return "orange"
    return "green"


# ===================== Historial de sesión =====================
if "historial_especies" not in st.session_state:
    st.session_state.historial_especies = []
if "historial_pronosticos" not in st.session_state:
    st.session_state.historial_pronosticos = []


# ===================== Especies de referencia por zona =====================
# Conocimiento público sobre pesca en Costa Rica -- NO sale de la CNN (que
# identifica especies de un dataset distinto, no nativas de CR)

# Es contenido de referencia: verificar con INCOPESCA

ESPECIES_POR_ZONA = {
    "golfo_nicoya": ["Corvina reina", "Pargo mancha", "Tortuga marina (protegida)"],
    "golfo_dulce": ["Pargo mancha", "Corvina reina", "Tiburón martillo (veda)"],
    "pacifico_norte": ["Dorado", "Atún aleta amarilla"],
    "pacifico_central": ["Dorado", "Atún aleta amarilla", "Marlín (pesca deportiva)"],
    "pacifico_sur": ["Marlín (pesca deportiva)", "Atún aleta amarilla", "Dorado"],
    "caribe_norte": ["Tortuga marina (protegida, zona de anidación)"],
    "caribe_sur": ["Corvina reina", "Pargo mancha"],
}


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
                        st.session_state.historial_especies.append({
                            "Especie": data["especie_es"],
                            "Confianza": f"{data['confianza'] * 100:.1f}%",
                        })

                        st.success(f"**Especie detectada:** {data['especie_es']}")
                        st.caption(f"*{data['nombre_cientifico']}* ({data['especie_en']})")
                        st.metric("Confianza", f"{data['confianza'] * 100:.1f}%")

                        st.info(
                            f"- **Estado legal:** {data['estado_legal']}\n"
                            f"- **Recomendación:** {data['recomendacion']}"
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
                st.session_state.historial_pronosticos.append({
                    "Zona": zona.replace("_", " ").title(),
                    "Horizonte": f"{horizonte}h",
                    "Generado desde": data["generado_desde"],
                })
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

# ===================== TAB 3: Mapa =====================
with tab3:
    st.header("🗺️ Mapa de recomendaciones")
    st.caption(
        "La recomendación usa el pronóstico real del RNN para la zona elegida, más reglas fijas por umbral. "
    )

    col_reglas, col_mapa = st.columns([1, 2])

    with col_reglas:
        zona_rec = st.selectbox(
            "Zona costera",
            options=list(ZONAS_CR.keys()),
            format_func=lambda z: z.replace("_", " ").title(),
            key="zona_recomendacion",
        )
        dist = st.slider("Distancia a costa (km)", 1, 80, 20)
        especie = st.selectbox("Especie objetivo", ["dorado", "atun", "pargo", "corvina", "otros"])

        if st.button("Obtener recomendación"):
            lat_rec, lon_rec = ZONAS_CR[zona_rec]
            with st.spinner("Consultando condiciones actuales..."):
                resp, error = llamar_api(
                    "GET", "/oceanografia/predict",
                    params={"lat": lat_rec, "lon": lon_rec, "timezone": "auto"},
                    timeout=60,
                )

            if error is not None:
                st.error(error)
            elif not resp.ok:
                mostrar_error_api(resp)
            else:
                actual = resp.json()["forecast"][0]
                altura = actual["wave_height"]
                viento = actual["wind_speed_10m"]
                sst = actual["sea_surface_temperature"]

                if altura > 2.5 or viento > 35:
                    rec, color, msg = "REGRESAR A PUERTO", "⚫", "Alerta meteorológica."
                elif altura > 1.8 or viento > 25:
                    rec, color, msg = "PESCA CON PRECAUCIÓN", "🟡", "Restricciones leves."
                elif dist > 60:
                    rec, color, msg = "CAMBIAR ZONA", "🔵", "Hay zona alternativa más cercana."
                elif 26 <= sst <= 30 and altura < 1.5:
                    rec, color, msg = "PESCA RECOMENDADA", "🟢", "Condiciones óptimas."
                else:
                    rec, color, msg = "PESCA CON PRECAUCIÓN", "🟡", "Condiciones aceptables."

                st.markdown(f"### {color} **{rec}** — {zona_rec.replace('_', ' ').title()}")
                st.caption(f"Oleaje: {altura:.2f} m · Viento: {viento:.1f} km/h · SST: {sst:.1f} °C")
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

    # ---- 1. Comparativo de las 7 zonas ----
    st.subheader("🌊 Condiciones actuales por zona")
    st.caption("Consulta para las 7 zonas costeras y muestra el pronóstico de cada una.")

    if st.button("🔄 Actualizar comparativo de zonas"):
        obtener_comparativo_zonas.clear()

    with st.spinner("Consultando el pronóstico de las 7 zonas..."):
        df_zonas, zonas_con_error = obtener_comparativo_zonas()

    if zonas_con_error:
        st.warning(
            "No se pudo consultar: "
            + ", ".join(z.replace("_", " ").title() for z in zonas_con_error)
        )

    if df_zonas.empty:
        st.error("No se pudo consultar ninguna zona. ¿Está la API arriba?")
    else:
        st.dataframe(
            df_zonas[["Zona", "Oleaje (m)", "Viento (km/h)", "SST (°C)"]],
            use_container_width=True, hide_index=True,
        )

        # ---- 2. Mapa semáforo ----
        st.subheader("🗺️ Mapa semáforo (oleaje actual)")
        st.caption("🟢 oleaje ≤ (menor a) 1.0 m · 🟠 1.0m y 1.8 m · 🔴 > (mayor a) 1.8 m")

        mapa_dashboard = folium.Map(location=[9.7, -84.0], zoom_start=7, tiles="CartoDB positron")
        for _, fila in df_zonas.iterrows():
            folium.CircleMarker(
                [fila["lat"], fila["lon"]],
                radius=14,
                color=color_oleaje(fila["Oleaje (m)"]),
                fill=True,
                fill_color=color_oleaje(fila["Oleaje (m)"]),
                fill_opacity=0.8,
                popup=(
                    f"<b>{fila['Zona']}</b><br>"
                    f"Oleaje: {fila['Oleaje (m)']:.2f} m<br>"
                    f"Viento: {fila['Viento (km/h)']:.1f} km/h<br>"
                    f"SST: {fila['SST (°C)']:.1f} °C"
                ),
            ).add_to(mapa_dashboard)
        st_folium(mapa_dashboard, width=900, height=450, key="mapa_dashboard")

    st.markdown("---")

    # ---- 3. Historial de esta sesión ----
    st.subheader("🕓 Actividad de esta sesión")
    col_hist_especies, col_hist_pronosticos = st.columns(2)

    with col_hist_especies:
        st.markdown("**Identificaciones de especie:**")
        if st.session_state.historial_especies:
            st.dataframe(
                pd.DataFrame(st.session_state.historial_especies),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("Todavía no identificaste ninguna especie en esta sesión (Tab 1).")

    with col_hist_pronosticos:
        st.markdown("**Pronósticos consultados:**")
        if st.session_state.historial_pronosticos:
            st.dataframe(
                pd.DataFrame(st.session_state.historial_pronosticos),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("Todavía no consultaste ningún pronóstico en esta sesión (Tab 2).")

    st.markdown("---")

    # ---- 4. Especies por zona (referencia) ----
    st.subheader("🐟 Especies por zona")
    st.caption(
        "Especies típicamente asociadas a cada zona costera, según conocimiento público de pesca en "
        "Costa Rica"
    )

    col_lista, col_mapa_especies = st.columns([1, 2])

    with col_lista:
        zona_dash = st.selectbox(
            "Zona costera",
            options=list(ZONAS_CR.keys()),
            format_func=lambda z: z.replace("_", " ").title(),
            key="zona_dashboard",
        )
        st.markdown(f"**Especies en {zona_dash.replace('_', ' ').title()}:**")
        for especie_zona in ESPECIES_POR_ZONA.get(zona_dash, []):
            st.markdown(f"- {especie_zona}")

    with col_mapa_especies:
        mapa_especies = folium.Map(location=[9.7, -84.0], zoom_start=7, tiles="CartoDB positron")
        for zona_id, (lat, lon) in ZONAS_CR.items():
            especies_zona = ESPECIES_POR_ZONA.get(zona_id, [])
            popup_html = f"<b>{zona_id.replace('_', ' ').title()}</b><br>" + "<br>".join(especies_zona)
            folium.Marker(
                [lat, lon],
                popup=popup_html,
                icon=folium.Icon(color="green", icon="fish", prefix="fa"),
            ).add_to(mapa_especies)
        st_folium(mapa_especies, width=900, height=500, key="mapa_especies_dashboard")
