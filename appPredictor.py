import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="PredictPro Industrial",
    page_icon="⚙️",
    layout="wide"
)

# =========================================================
# ESTILOS CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #07111f;
        color: white;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-bottom: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06233f, #02111f);
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 18px;
    }

    .header-title {
        font-size: 30px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.15;
    }

    .header-subtitle {
        font-size: 14px;
        color: #9ca3af;
        margin-top: 6px;
    }

    .header-time {
        text-align: right;
        color: #ffffff;
        font-size: 14px;
        line-height: 1.8;
        min-width: 150px;
    }

    .metric-card {
        background: linear-gradient(145deg, #132235, #0d1828);
        border: 1px solid #26384f;
        border-radius: 12px;
        padding: 18px;
        color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        min-height: 120px;
    }

    .metric-title {
        font-size: 14px;
        color: #b8c7d9;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
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
        box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
    }

    .diagnostic-title {
        font-size: 30px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .diagnostic-subtitle {
        font-size: 15px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .diagnostic-card {
        background: linear-gradient(145deg, #132235, #0d1828);
        border: 1px solid #26384f;
        border-radius: 14px;
        padding: 22px;
        color: white;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.35);
        margin-bottom: 18px;
    }

    .result-normal {
        background: linear-gradient(145deg, #12351f, #0b2415);
        border: 1px solid #2ecc71;
        border-radius: 14px;
        padding: 24px;
        color: white;
        text-align: center;
        margin-bottom: 18px;
    }

    .result-warning {
        background: linear-gradient(145deg, #3a2608, #1e1404);
        border: 1px solid #ff9f1c;
        border-radius: 14px;
        padding: 24px;
        color: white;
        text-align: center;
        margin-bottom: 18px;
    }

    .result-danger {
        background: linear-gradient(145deg, #3a1111, #1f0808);
        border: 1px solid #ff4d4d;
        border-radius: 14px;
        padding: 24px;
        color: white;
        text-align: center;
        margin-bottom: 18px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-text {
        font-size: 17px;
        color: #d1d5db;
    }

    .small-text {
        font-size: 13px;
        color: #b8c7d9;
    }

    .normal-text {
        color: #59d14f;
        font-weight: bold;
    }

    .warning-text {
        color: #ff9f1c;
        font-weight: bold;
    }

    .danger-text {
        color: #ff5c33;
        font-weight: bold;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #132235, #0d1828);
        border: 1px solid #26384f;
        border-radius: 12px;
        padding: 12px;
    }

    @media (max-width: 900px) {
        .header-container {
            flex-direction: column;
        }

        .header-time {
            text-align: left;
            margin-top: 10px;
        }

        .header-title {
            font-size: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# ARCHIVOS DEL PROYECTO
# =========================================================

DATA_PATH = "data.csv"
MODEL_PATH = "best_model_pipeline.joblib"


# =========================================================
# COLUMNAS DE ENTRADA DEL MODELO
# =========================================================

MODEL_INPUTS = [
    "rpm",
    "motor_power",
    "torque",
    "outlet_pressure_bar",
    "air_flow",
    "noise_db",
    "outlet_temp",
    "wpump_outlet_press",
    "water_inlet_temp",
    "water_outlet_temp",
    "wpump_power",
    "water_flow",
    "oilpump_power",
    "oil_tank_temp",
    "gaccx",
    "gaccy",
    "gaccz",
    "haccx",
    "haccy",
    "haccz"
]

INPUT_LABELS = {
    "rpm": "RPM",
    "motor_power": "Potencia del motor",
    "torque": "Torque",
    "outlet_pressure_bar": "Presión de salida",
    "air_flow": "Flujo de aire",
    "noise_db": "Ruido",
    "outlet_temp": "Temperatura de salida",
    "wpump_outlet_press": "Presión de salida bomba de agua",
    "water_inlet_temp": "Temperatura entrada de agua",
    "water_outlet_temp": "Temperatura salida de agua",
    "wpump_power": "Potencia bomba de agua",
    "water_flow": "Flujo de agua",
    "oilpump_power": "Potencia bomba de aceite",
    "oil_tank_temp": "Temperatura tanque de aceite",
    "gaccx": "Aceleración gaccx",
    "gaccy": "Aceleración gaccy",
    "gaccz": "Aceleración gaccz",
    "haccx": "Aceleración haccx",
    "haccy": "Aceleración haccy",
    "haccz": "Aceleración haccz"
}

INPUT_UNITS = {
    "rpm": "rpm",
    "motor_power": "kW",
    "torque": "Nm",
    "outlet_pressure_bar": "bar",
    "air_flow": "L/min",
    "noise_db": "dB",
    "outlet_temp": "°C",
    "wpump_outlet_press": "bar",
    "water_inlet_temp": "°C",
    "water_outlet_temp": "°C",
    "wpump_power": "kW",
    "water_flow": "L/min",
    "oilpump_power": "kW",
    "oil_tank_temp": "°C",
    "gaccx": "g",
    "gaccy": "g",
    "gaccz": "g",
    "haccx": "g",
    "haccy": "g",
    "haccz": "g"
}


# =========================================================
# FUNCIONES
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def define_fault(row):
    if "bearings" in row and row["bearings"] == "Noisy":
        return "Rodamientos ruidosos"
    elif "wpump" in row and row["wpump"] == "Noisy":
        return "Bomba de agua ruidosa"
    elif "radiator" in row and row["radiator"] == "Dirty":
        return "Radiador sucio"
    elif "exvalve" in row and row["exvalve"] == "Dirty":
        return "Válvula de escape sucia"
    else:
        return "Normal"


def safe_value(row, column, default=0.0):
    if column in row.index:
        return row[column]
    return default


def get_prediction(model, input_data):
    prediction = model.predict(input_data)[0]

    prob_df = None
    max_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        prob_df = pd.DataFrame({
            "Tipo de falla": classes,
            "Probabilidad": probabilities
        }).sort_values("Probabilidad", ascending=False)

        max_probability = prob_df.iloc[0]["Probabilidad"]

    return prediction, prob_df, max_probability


def show_prediction_result(prediction, max_probability=None):
    if prediction == "Normal":
        st.markdown("""
        <div class="result-normal">
            <div class="result-title">✅ Sistema normal</div>
            <div class="result-text">
                El modelo predice que el compresor se encuentra en operación normal.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == "Radiador sucio":
        st.markdown("""
        <div class="result-warning">
            <div class="result-title">⚠️ Radiador sucio</div>
            <div class="result-text">
                El modelo detecta una posible falla asociada al radiador.
                Se recomienda inspeccionar y limpiar el radiador.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == "Bomba de agua ruidosa":
        st.markdown("""
        <div class="result-warning">
            <div class="result-title">⚠️ Bomba de agua ruidosa</div>
            <div class="result-text">
                El modelo detecta una posible anomalía en la bomba de agua.
                Se recomienda revisar presión, flujo y estado mecánico de la bomba.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == "Rodamientos ruidosos":
        st.markdown("""
        <div class="result-warning">
            <div class="result-title">⚠️ Rodamientos ruidosos</div>
            <div class="result-text">
                El modelo detecta una posible falla en los rodamientos.
                Se recomienda revisar vibración, ruido y lubricación.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == "Válvula de escape sucia":
        st.markdown("""
        <div class="result-warning">
            <div class="result-title">⚠️ Válvula de escape sucia</div>
            <div class="result-text">
                El modelo detecta una posible suciedad u obstrucción en la válvula de escape.
                Se recomienda inspeccionar y limpiar la válvula.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-danger">
            <div class="result-title">⚠️ {prediction}</div>
            <div class="result-text">
                El modelo detectó una condición anormal en el sistema.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if max_probability is not None:
        st.info(f"Confianza estimada del diagnóstico: {max_probability:.2%}")


def show_header():
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    hora_actual = datetime.now().strftime("%H:%M:%S")

    st.markdown(f"""
    <div class="header-container">
        <div>
            <h1 class="header-title">Sistema de Monitoreo y Diagnóstico Predictivo</h1>
            <div class="header-subtitle">Panel operativo en tiempo real</div>
        </div>
        <div class="header-time">
            <div>📅 {fecha_actual}</div>
            <div>🕒 {hora_actual}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# CARGA DE DATOS Y MODELO
# =========================================================

try:
    df = load_data()
except Exception as e:
    st.error(f"No se pudo cargar el archivo {DATA_PATH}. Error: {e}")
    st.stop()

if "fault_type" not in df.columns:
    df["fault_type"] = df.apply(define_fault, axis=1)

try:
    model = load_model()
except Exception as e:
    model = None
    model_error = str(e)
else:
    model_error = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚙️ PredictPro")
st.sidebar.markdown("### INDUSTRIAL")
st.sidebar.markdown("")

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

if model is not None:
    st.sidebar.success("Modelo cargado")
else:
    st.sidebar.error("Modelo no cargado")

st.sidebar.success("Sistema conectado")
st.sidebar.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")


# =========================================================
# VALIDACIÓN DE COLUMNAS DEL DATASET
# =========================================================

missing_cols = [col for col in MODEL_INPUTS if col not in df.columns]

if missing_cols:
    st.warning(
        "El dataset no contiene todas las columnas esperadas por el modelo. "
        f"Columnas faltantes: {missing_cols}"
    )


# =========================================================
# VENTANA: INICIO
# =========================================================

if menu == "Inicio":

    show_header()

    selected_index = st.sidebar.slider(
        "Selecciona una lectura",
        min_value=0,
        max_value=len(df) - 1,
        value=len(df) - 1
    )

    current = df.iloc[selected_index]

    available_inputs = [col for col in MODEL_INPUTS if col in df.columns]

    if len(available_inputs) == len(MODEL_INPUTS) and model is not None:
        X_current = current[MODEL_INPUTS].to_frame().T
        prediction, prob_df, max_probability = get_prediction(model, X_current)
    else:
        prediction = current.get("fault_type", "Sin diagnóstico")
        prob_df = None
        max_probability = None

    rpm = safe_value(current, "rpm")
    motor_power = safe_value(current, "motor_power")
    torque = safe_value(current, "torque")
    pressure = safe_value(current, "outlet_pressure_bar")
    air_flow = safe_value(current, "air_flow")
    water_flow = safe_value(current, "water_flow")
    outlet_temp = safe_value(current, "outlet_temp")
    noise = safe_value(current, "noise_db")

    metrics = [
        ("RPM", rpm, "rpm", "🌀"),
        ("Potencia del motor", motor_power, "kW", "⚡"),
        ("Torque", torque, "Nm", "🔧"),
        ("Presión de salida", pressure, "bar", "📈"),
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
                <span class="metric-value">{float(value):,.2f}</span>
                <span class="metric-unit">{unit}</span>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    metric_cols_2 = st.columns(4)

    for i, (title, value, unit, icon) in enumerate(metrics[4:]):
        with metric_cols_2[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{icon} {title}</div>
                <span class="metric-value">{float(value):,.2f}</span>
                <span class="metric-unit">{unit}</span>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

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

        if "rpm" in trend_df.columns:
            fig.add_trace(go.Scatter(
                y=trend_df["rpm"],
                mode="lines+markers",
                name="RPM"
            ))

        if "outlet_temp" in trend_df.columns:
            fig.add_trace(go.Scatter(
                y=trend_df["outlet_temp"],
                mode="lines+markers",
                name="Temperatura de salida"
            ))

        if "water_flow" in trend_df.columns:
            fig.add_trace(go.Scatter(
                y=trend_df["water_flow"],
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
            st.markdown('<p class="normal-text">Estado del sistema estable</p>', unsafe_allow_html=True)
        else:
            st.markdown("## ⚠️ Advertencia")
            st.markdown(f'<p class="warning-text">{prediction}</p>', unsafe_allow_html=True)

        if max_probability is not None:
            st.info(f"Confianza estimada: {max_probability:.2%}")

        if prob_df is not None:
            st.write("Probabilidad de fallas")

            for _, row in prob_df.iterrows():
                st.write(f"{row['Tipo de falla']}: {row['Probabilidad']:.1%}")
                st.progress(float(row["Probabilidad"]))

        elif model is None:
            st.error(f"No se pudo cargar el modelo: {model_error}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    col_alerts, col_components = st.columns([1.3, 1])

    with col_alerts:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Alertas y recomendaciones")

        alerts = []

        if prediction != "Normal":
            alerts.append(f"Diagnóstico detectado: {prediction}")

        if "outlet_temp" in df.columns:
            if outlet_temp > df["outlet_temp"].quantile(0.75):
                alerts.append("Aumento de temperatura de salida detectado")

        if "water_flow" in df.columns:
            if water_flow < df["water_flow"].quantile(0.25):
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
            "Motor AC": current.get("acmotor", "Normal")
        }

        for component, status in components.items():
            if status in ["Normal", "Clean", "Stable"]:
                st.markdown(f"🟢 **{component}** — Normal")
            elif status in ["Noisy", "Dirty"]:
                st.markdown(f"🟠 **{component}** — Advertencia")
            else:
                st.markdown(f"🔵 **{component}** — {status}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Vibración / Aceleración")

    vib_cols = ["gaccx", "gaccy", "gaccz", "haccx", "haccy", "haccz"]
    vib_cols = [col for col in vib_cols if col in df.columns]

    if len(vib_cols) > 0:
        vib_layout = st.columns(len(vib_cols))

        for i, col in enumerate(vib_cols):
            with vib_layout[i]:
                value = safe_value(current, col)
                st.metric(col, f"{float(value):.4f} g")
                st.progress(min(float(abs(value)) / 0.1, 1.0))
    else:
        st.caption("No se encontraron columnas de vibración/aceleración en el dataset.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Historial de eventos")

    history = pd.DataFrame({
        "Fecha / Hora": [
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
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


# =========================================================
# VENTANA: MONITOREO
# =========================================================

elif menu == "Monitoreo":

    show_header()

    st.markdown('<div class="diagnostic-title">Monitoreo de variables</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="diagnostic-subtitle">Visualización de las variables registradas por el sistema.</div>',
        unsafe_allow_html=True
    )

    available_cols = [col for col in MODEL_INPUTS if col in df.columns]

    selected_vars = st.multiselect(
        "Selecciona las variables a visualizar",
        options=available_cols,
        default=available_cols[:3]
    )

    if selected_vars:
        fig = go.Figure()

        for col in selected_vars:
            fig.add_trace(go.Scatter(
                y=df[col],
                mode="lines",
                name=col
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[available_cols].head(100), use_container_width=True)


# =========================================================
# VENTANA: DIAGNÓSTICO
# =========================================================

elif menu == "Diagnóstico":

    show_header()

    st.markdown('<div class="diagnostic-title">Diagnóstico predictivo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="diagnostic-subtitle">Ingrese las variables medidas del compresor para predecir el estado del sistema.</div>',
        unsafe_allow_html=True
    )

    default_values = {}

    for col in MODEL_INPUTS:
        if col in df.columns:
            default_values[col] = float(df[col].iloc[-1])
        else:
            default_values[col] = 0.0

    st.markdown('<div class="diagnostic-card">', unsafe_allow_html=True)
    st.subheader("Datos de entrada del modelo")

    with st.form("diagnostic_form"):

        st.markdown("### Variables principales")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rpm = st.number_input(
                f"{INPUT_LABELS['rpm']} ({INPUT_UNITS['rpm']})",
                value=default_values["rpm"],
                step=10.0
            )

            motor_power = st.number_input(
                f"{INPUT_LABELS['motor_power']} ({INPUT_UNITS['motor_power']})",
                value=default_values["motor_power"],
                step=0.1
            )

            torque = st.number_input(
                f"{INPUT_LABELS['torque']} ({INPUT_UNITS['torque']})",
                value=default_values["torque"],
                step=0.1
            )

            outlet_pressure_bar = st.number_input(
                f"{INPUT_LABELS['outlet_pressure_bar']} ({INPUT_UNITS['outlet_pressure_bar']})",
                value=default_values["outlet_pressure_bar"],
                step=0.1
            )

            air_flow = st.number_input(
                f"{INPUT_LABELS['air_flow']} ({INPUT_UNITS['air_flow']})",
                value=default_values["air_flow"],
                step=1.0
            )

        with col2:
            noise_db = st.number_input(
                f"{INPUT_LABELS['noise_db']} ({INPUT_UNITS['noise_db']})",
                value=default_values["noise_db"],
                step=0.1
            )

            outlet_temp = st.number_input(
                f"{INPUT_LABELS['outlet_temp']} ({INPUT_UNITS['outlet_temp']})",
                value=default_values["outlet_temp"],
                step=0.1
            )

            wpump_outlet_press = st.number_input(
                f"{INPUT_LABELS['wpump_outlet_press']} ({INPUT_UNITS['wpump_outlet_press']})",
                value=default_values["wpump_outlet_press"],
                step=0.1
            )

            water_inlet_temp = st.number_input(
                f"{INPUT_LABELS['water_inlet_temp']} ({INPUT_UNITS['water_inlet_temp']})",
                value=default_values["water_inlet_temp"],
                step=0.1
            )

            water_outlet_temp = st.number_input(
                f"{INPUT_LABELS['water_outlet_temp']} ({INPUT_UNITS['water_outlet_temp']})",
                value=default_values["water_outlet_temp"],
                step=0.1
            )

        with col3:
            wpump_power = st.number_input(
                f"{INPUT_LABELS['wpump_power']} ({INPUT_UNITS['wpump_power']})",
                value=default_values["wpump_power"],
                step=0.1
            )

            water_flow = st.number_input(
                f"{INPUT_LABELS['water_flow']} ({INPUT_UNITS['water_flow']})",
                value=default_values["water_flow"],
                step=0.1
            )

            oilpump_power = st.number_input(
                f"{INPUT_LABELS['oilpump_power']} ({INPUT_UNITS['oilpump_power']})",
                value=default_values["oilpump_power"],
                step=0.1
            )

            oil_tank_temp = st.number_input(
                f"{INPUT_LABELS['oil_tank_temp']} ({INPUT_UNITS['oil_tank_temp']})",
                value=default_values["oil_tank_temp"],
                step=0.1
            )

        with col4:
            gaccx = st.number_input(
                f"{INPUT_LABELS['gaccx']} ({INPUT_UNITS['gaccx']})",
                value=default_values["gaccx"],
                step=0.001,
                format="%.4f"
            )

            gaccy = st.number_input(
                f"{INPUT_LABELS['gaccy']} ({INPUT_UNITS['gaccy']})",
                value=default_values["gaccy"],
                step=0.001,
                format="%.4f"
            )

            gaccz = st.number_input(
                f"{INPUT_LABELS['gaccz']} ({INPUT_UNITS['gaccz']})",
                value=default_values["gaccz"],
                step=0.001,
                format="%.4f"
            )

            haccx = st.number_input(
                f"{INPUT_LABELS['haccx']} ({INPUT_UNITS['haccx']})",
                value=default_values["haccx"],
                step=0.001,
                format="%.4f"
            )

            haccy = st.number_input(
                f"{INPUT_LABELS['haccy']} ({INPUT_UNITS['haccy']})",
                value=default_values["haccy"],
                step=0.001,
                format="%.4f"
            )

            haccz = st.number_input(
                f"{INPUT_LABELS['haccz']} ({INPUT_UNITS['haccz']})",
                value=default_values["haccz"],
                step=0.001,
                format="%.4f"
            )

        submitted = st.form_submit_button("Generar diagnóstico")

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:

        input_data = pd.DataFrame([{
            "rpm": rpm,
            "motor_power": motor_power,
            "torque": torque,
            "outlet_pressure_bar": outlet_pressure_bar,
            "air_flow": air_flow,
            "noise_db": noise_db,
            "outlet_temp": outlet_temp,
            "wpump_outlet_press": wpump_outlet_press,
            "water_inlet_temp": water_inlet_temp,
            "water_outlet_temp": water_outlet_temp,
            "wpump_power": wpump_power,
            "water_flow": water_flow,
            "oilpump_power": oilpump_power,
            "oil_tank_temp": oil_tank_temp,
            "gaccx": gaccx,
            "gaccy": gaccy,
            "gaccz": gaccz,
            "haccx": haccx,
            "haccy": haccy,
            "haccz": haccz
        }])

        input_data = input_data[MODEL_INPUTS]

        if model is not None:

            try:
                prediction, prob_df, max_probability = get_prediction(model, input_data)

                st.markdown("## Resultado del diagnóstico")
                show_prediction_result(prediction, max_probability)

                if prob_df is not None:
                    st.markdown("### Probabilidad por tipo de falla")

                    for _, row in prob_df.iterrows():
                        st.write(f"**{row['Tipo de falla']}**: {row['Probabilidad']:.2%}")
                        st.progress(float(row["Probabilidad"]))

                    st.dataframe(
                        prob_df,
                        use_container_width=True,
                        hide_index=True
                    )

                with st.expander("Ver datos ingresados al modelo"):
                    st.dataframe(input_data, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"No se pudo generar el diagnóstico. Error: {e}")

        else:
            st.error(
                "No se pudo cargar el modelo. "
                "Verifica que el archivo best_model_pipeline.joblib esté en la carpeta del proyecto."
            )
            if model_error:
                st.code(model_error)


# =========================================================
# VENTANA: ALERTAS
# =========================================================

elif menu == "Alertas":

    show_header()

    st.markdown('<div class="diagnostic-title">Alertas del sistema</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="diagnostic-subtitle">Eventos generados por condiciones anormales del compresor.</div>',
        unsafe_allow_html=True
    )

    alerts_data = []

    for _, row in df.tail(50).iterrows():
        fault = row.get("fault_type", "Normal")

        if fault != "Normal":
            alerts_data.append({
                "Evento": f"Diagnóstico: {fault}",
                "Severidad": "Advertencia",
                "Acción sugerida": "Programar mantenimiento preventivo"
            })

    if len(alerts_data) == 0:
        st.success("No hay alertas activas.")
    else:
        alerts_df = pd.DataFrame(alerts_data)
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)


# =========================================================
# VENTANA: HISTÓRICO
# =========================================================

elif menu == "Histórico":

    show_header()

    st.markdown('<div class="diagnostic-title">Histórico de datos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="diagnostic-subtitle">Consulta de registros históricos del compresor.</div>',
        unsafe_allow_html=True
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar histórico en CSV",
        data=csv,
        file_name="historico_compresor.csv",
        mime="text/csv"
    )


# =========================================================
# VENTANA: MANTENIMIENTO
# =========================================================

elif menu == "Mantenimiento":

    show_header()

    st.markdown('<div class="diagnostic-title">Mantenimiento recomendado</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="diagnostic-subtitle">Acciones sugeridas según el tipo de diagnóstico.</div>',
        unsafe_allow_html=True
    )

    maintenance_df = pd.DataFrame({
        "Falla detectada": [
            "Rodamientos ruidosos",
            "Bomba de agua ruidosa",
            "Radiador sucio",
            "Válvula de escape sucia",
            "Normal"
        ],
        "Acción recomendada": [
            "Revisar lubricación, vibración y desgaste de rodamientos.",
            "Verificar presión, flujo de agua y estado mecánico de la bomba.",
            "Inspeccionar y limpiar el radiador.",
            "Inspeccionar y limpiar la válvula de escape.",
            "Continuar monitoreo normal."
        ],
        "Prioridad": [
            "Media",
            "Media",
            "Alta",
            "Media",
            "Baja"
        ]
    })

    st.dataframe(maintenance_df, use_container_width=True, hide_index=True)


# =========================================================
# VENTANA: CONFIGURACIÓN
# =========================================================

elif menu == "Configuración":

    show_header()

    st.markdown('<div class="diagnostic-title">Configuración</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="diagnostic-subtitle">Estado de archivos, modelo y variables del sistema.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Archivos")

    st.write(f"Archivo de datos: `{DATA_PATH}`")
    st.write(f"Archivo del modelo: `{MODEL_PATH}`")

    st.markdown("### Estado del modelo")

    if model is not None:
        st.success("Modelo cargado correctamente.")
    else:
        st.error("Modelo no cargado.")
        if model_error:
            st.code(model_error)

    st.markdown("### Variables esperadas por el modelo")

    variables_df = pd.DataFrame({
        "Variable": MODEL_INPUTS,
        "Nombre": [INPUT_LABELS[col] for col in MODEL_INPUTS],
        "Unidad": [INPUT_UNITS[col] for col in MODEL_INPUTS],
        "Existe en data.csv": [col in df.columns for col in MODEL_INPUTS]
    })

    st.dataframe(variables_df, use_container_width=True, hide_index=True)