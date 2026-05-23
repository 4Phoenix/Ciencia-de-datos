import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="PredictPro Dashboard", page_icon="📊", layout="centered")

#ESTILOS CSS

st.markdown("""
<style>
    .main {
        background-color: #07111f;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #132235, #0d1828);
        border: 1px solid #26384f;
        border-radius: 12px;
        padding: 18px;
        color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }

    .metric-title {
        font-size: 14px;
        color: #b8c7d9;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: bold;
        color: #ffffff;
    }

    .metric-unit {
        font-size: 13px;
        color: #9caec4;
        margin-left: 4px;
    }

    .panel {
        background: linear-gradient(145deg, #132235, #0d1828);
        border: 1px solid #26384f;
        border-radius: 12px;
        padding: 18px;
        color: white;
        height: 100%;
    }

    .warning {
        color: #ff9f1c;
        font-weight: bold;
    }

    .normal {
        color: #59d14f;
        font-weight: bold;
    }

    .danger {
        color: #ff5c33;
        font-weight: bold;
    }

    .small-text {
        font-size: 13px;
        color: #b8c7d9;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06233f, #02111f);
    }
</style>
""", unsafe_allow_html=True)

#CARGA DE DATOS

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.joblib")

df = load_data()

try:
    model = load_model()
except:
    model = None

# VARIABLE OBJETIVO
# =========================

def define_fault(row):
    if row["bearings"] == "Noisy":
        return "Rodamientos ruidosos"
    elif row["wpump"] == "Noisy":
        return "Bomba de agua ruidosa"
    elif row["radiator"] == "Dirty":
        return "Radiador sucio"
    elif row["exvalve"] == "Dirty":
        return "Válvula de escape sucia"
    else:
        return "Normal"

if "fault_type" not in df.columns:
    df["fault_type"] = df.apply(define_fault, axis=1)

# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("## ⚙️ PredictPro")
st.sidebar.markdown("### INDUSTRIAL")

menu = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "Monitoreo",
        "Diagnóstico",
        "Alertas",
        "Histórico",
        "Mantenimiento",
        "Configuración"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("Sistema conectado")
st.sidebar.caption(f"Última actualización: {dt.datetime.now().strftime('%H:%M:%S')}")

# =========================
# HEADER
# =========================

col_title, col_time = st.columns([3, 1])

with col_title:
    st.title("Sistema de Monitoreo y Diagnóstico Predictivo")
    st.caption("Panel operativo en tiempo real")

with col_time:
    st.write("")
    st.write(dt.datetime.now().strftime("%d/%m/%Y"))
    st.write(dt.datetime.now().strftime("%H:%M:%S"))



# =========================
# SELECCIÓN DE REGISTRO
# =========================

st.sidebar.markdown("### Registro de datos")
selected_index = st.sidebar.slider(
    "Selecciona una lectura",
    min_value=0,
    max_value=len(df) - 1,
    value=len(df) - 1
)

current = df.iloc[selected_index]

# =========================
# COLUMNAS NUMÉRICAS
# =========================

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if "id" in numerical_cols:
    numerical_cols.remove("id")

X_current = current[numerical_cols].to_frame().T

# =========================
# PREDICCIÓN
# =========================

if model is not None:
    prediction = model.predict(X_current)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_current)[0]
        classes = model.classes_
        prob_df = pd.DataFrame({
            "Falla": classes,
            "Probabilidad": probabilities
        }).sort_values("Probabilidad", ascending=False)
    else:
        prob_df = None
else:
    prediction = current["fault_type"]
    prob_df = None

# =========================
# TARJETAS SUPERIORES
# =========================

def get_value(pos, default=0):
    if pos < len(numerical_cols):
        return current[numerical_cols[pos]]
    return default

rpm = get_value(0)
motor_power = get_value(1)
torque = get_value(2)
pressure = get_value(3)
air_flow = get_value(4)
water_flow = get_value(5)
outlet_temp = get_value(6)
noise = get_value(7)

metrics = [
    ("RPM", rpm, "rpm", "🌀"),
    ("Potencia del motor", motor_power, "kW", "⚡"),
    ("Torque", torque, "Nm", "🔧"),
    ("Presión de salida", pressure, "bar", "🌡️"),
    ("Flujo de aire", air_flow, "L/min", "💨"),
    ("Flujo de agua", water_flow, "L/min", "💧"),
    ("Temperatura de salida", outlet_temp, "°C", "🌡️"),
    ("Ruido", noise, "dB", "🔊"),
]

metric_cols = st.columns(4)

for i, (title, value, unit, icon) in enumerate(metrics[:4]):
    with metric_cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{icon} {title}</div>
            <span class="metric-value">{value:,.2f}</span>
            <span class="metric-unit">{unit}</span>
        </div>
        """, unsafe_allow_html=True)

metric_cols_2 = st.columns(4)

for i, (title, value, unit, icon) in enumerate(metrics[4:]):
    with metric_cols_2[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{icon} {title}</div>
            <span class="metric-value">{value:,.2f}</span>
            <span class="metric-unit">{unit}</span>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# =========================
# GRÁFICA DE TENDENCIAS
# =========================

left, right = st.columns([2.1, 1.3])

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Tendencias en tiempo real")

    window = st.slider(
        "Ventana de registros",
        min_value=10,
        max_value=min(200, len(df)),
        value=min(50, len(df))
    )

    start = max(0, selected_index - window)
    trend_df = df.iloc[start:selected_index + 1].copy()

    fig = go.Figure()

    if len(numerical_cols) >= 1:
        fig.add_trace(go.Scatter(
            y=trend_df[numerical_cols[0]],
            mode="lines+markers",
            name="RPM"
        ))

    if len(numerical_cols) >= 7:
        fig.add_trace(go.Scatter(
            y=trend_df[numerical_cols[6]],
            mode="lines+markers",
            name="Temperatura de salida"
        ))

    if len(numerical_cols) >= 6:
        fig.add_trace(go.Scatter(
            y=trend_df[numerical_cols[5]],
            mode="lines+markers",
            name="Flujo de agua"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=430,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Diagnóstico actual")

    if prediction == "Normal":
        st.markdown("## ✅ Normal")
        st.markdown('<p class="normal">Estado del sistema estable</p>', unsafe_allow_html=True)
    else:
        st.markdown("## ⚠️ Advertencia")
        st.markdown(f'<p class="warning">{prediction}</p>', unsafe_allow_html=True)

    if prob_df is not None:
        st.write("Probabilidad de fallas")

        for _, row in prob_df.iterrows():
            st.write(f"{row['Falla']}: {row['Probabilidad']:.1%}")
            st.progress(float(row["Probabilidad"]))

    else:
        st.info("Modelo cargado sin probabilidades. Se muestra solo la predicción final.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# COMPONENTES Y ALERTAS
# =========================

col_alerts, col_components = st.columns([1.3, 1])

with col_alerts:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Alertas y recomendaciones")

    alerts = []

    if prediction != "Normal":
        alerts.append(f"Diagnóstico detectado: {prediction}")

    if len(numerical_cols) >= 7 and outlet_temp > df[numerical_cols[6]].quantile(0.75):
        alerts.append("Aumento de temperatura de salida detectado")

    if len(numerical_cols) >= 6 and water_flow < df[numerical_cols[5]].quantile(0.25):
        alerts.append("Caída del flujo de agua detectada")

    if len(alerts) == 0:
        st.success("No hay alertas críticas en este momento.")
    else:
        for alert in alerts:
            st.warning(alert)

    if prediction == "Radiador sucio":
        st.info("Recomendación: inspeccionar y limpiar el radiador.")
    elif prediction == "Bomba de agua ruidosa":
        st.info("Recomendación: verificar bomba de agua y líneas de suministro.")
    elif prediction == "Rodamientos ruidosos":
        st.info("Recomendación: revisar lubricación y estado de rodamientos.")
    elif prediction == "Válvula de escape sucia":
        st.info("Recomendación: inspeccionar y limpiar válvula de escape.")
    else:
        st.info("Sistema en operación normal.")

    st.markdown('</div>', unsafe_allow_html=True)

with col_components:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Componentes del sistema")

    components = {
        "Rodamientos": current.get("bearings", "Normal"),
        "Bomba de agua": current.get("wpump", "Normal"),
        "Radiador": current.get("radiator", "Normal"),
        "Válvula de escape": current.get("exvalve", "Normal"),
        "Motor AC": current.get("motor", "Normal")
    }

    for component, status in components.items():
        if status in ["Normal", "Clean", "Stable"]:
            st.markdown(f"🟢 **{component}** — Normal")
        elif status in ["Noisy", "Dirty"]:
            st.markdown(f"🟠 **{component}** — Advertencia")
        else:
            st.markdown(f"🔵 **{component}** — {status}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# VIBRACIÓN / ACELERACIÓN
# =========================

st.write("")
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Vibración / Aceleración")

vib_cols = [col for col in numerical_cols if "acc" in col.lower() or "gacc" in col.lower() or "hacc" in col.lower()]

if len(vib_cols) > 0:
    vib_layout = st.columns(min(6, len(vib_cols)))

    for i, col in enumerate(vib_cols[:6]):
        with vib_layout[i]:
            value = current[col]
            st.metric(col, f"{value:.3f} g")
            st.progress(min(float(abs(value)) / 0.1, 1.0))
else:
    st.caption("No se encontraron columnas de vibración/aceleración en el dataset.")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HISTORIAL DE EVENTOS
# =========================

st.write("")
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Historial de eventos")

history = pd.DataFrame({
    "Fecha / Hora": [
        dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ],
    "Evento": [
        f"Diagnóstico: {prediction}",
        "Lectura de sensores registrada",
        "Sistema actualizado"
    ],
    "Severidad": [
        "Advertencia" if prediction != "Normal" else "Información",
        "Información",
        "Información"
    ],
    "Acción sugerida": [
        "Programar mantenimiento preventivo" if prediction != "Normal" else "Ninguna acción requerida",
        "Continuar monitoreo",
        "Sistema en operación normal"
    ]
})

st.dataframe(history, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)
