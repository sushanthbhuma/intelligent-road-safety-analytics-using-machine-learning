
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium

from streamlit_folium import st_folium
from folium.plugins import HeatMap

from ml_pipeline import (
    generate_sample_data,
    train_and_evaluate,
    perform_clustering
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RoadSafe AI | Intelligent Road Safety",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b1120;
    --panel: #111c30;
    --panel2: #16243b;
    --border: #263650;
    --blue: #38bdf8;
    --blue2: #2563eb;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --green: #34d399;
    --red: #fb7185;
    --purple: #a78bfa;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(11,17,32,0.96);
}

[data-testid="stToolbar"] {
    right: 1rem;
}

[data-testid="stSidebar"] {
    background: #0f192b;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.3rem;
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1600px;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #cbd5e1;
}

.small-muted {
    color: var(--muted);
    font-size: 0.86rem;
}

.brand {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.7px;
    color: #f8fafc;
    margin-bottom: 2px;
}

.brand span {
    color: var(--blue);
}

.brand-caption {
    color: #64748b;
    font-size: 0.68rem;
    letter-spacing: 1.4px;
    font-weight: 600;
    margin-bottom: 25px;
}

.section-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin: 20px 0 10px 0;
}

.page-eyebrow {
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.page-title {
    font-size: clamp(1.8rem, 3vw, 2.5rem);
    line-height: 1.2;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -1.2px;
    margin-bottom: 8px;
}

.page-subtitle {
    color: #94a3b8;
    font-size: 0.98rem;
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(145deg, #14223a, #101a2d);
    border: 1px solid #263650;
    border-radius: 15px;
    padding: 20px 21px;
    min-height: 145px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.13);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: #3b82f6;
}

.metric-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.metric-label {
    font-size: 0.83rem;
    color: #a5b4c8;
    font-weight: 500;
}

.metric-icon {
    font-size: 1.15rem;
    background: #20324c;
    border-radius: 9px;
    padding: 8px 10px;
}

.metric-value {
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -1px;
}

.metric-note {
    font-size: 0.76rem;
    color: #64748b;
    margin-top: 9px;
}

.panel {
    background: #111c30;
    border: 1px solid #263650;
    border-radius: 15px;
    padding: 18px 20px;
}

.panel-title {
    color: #f1f5f9;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-bottom: 15px;
}

.status-pill {
    display: inline-block;
    background: #123b32;
    color: #34d399;
    border: 1px solid #16634c;
    border-radius: 30px;
    padding: 5px 11px;
    font-size: 0.73rem;
    font-weight: 700;
}

.info-box {
    background: #142944;
    border: 1px solid #25496e;
    border-left: 4px solid #38bdf8;
    border-radius: 9px;
    padding: 15px 17px;
    color: #cbd5e1;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #2563eb, #0284c7);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 43px;
    transition: 0.2s ease;
}

div[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0369a1);
    border: none;
    color: white;
    transform: translateY(-1px);
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label {
    color: #cbd5e1 !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: #16243b;
    border-color: #334155;
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263650;
    border-radius: 10px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #263650;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
    .metric-card {
        min-height: 125px;
        padding: 15px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATA AND MODEL
# =========================================================

@st.cache_data
def load_data():
    return generate_sample_data(3000)


@st.cache_resource
def train_model(data):
    return train_and_evaluate(data)


@st.cache_data
def get_clustered_data(data):
    return perform_clustering(data)


with st.spinner("Initializing RoadSafe AI..."):
    df = load_data()
    ebm_model, metrics, X_test, y_test = train_model(df)
    clustered_data = get_clustered_data(df)


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def page_heading(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="page-eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value, icon, color, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <div class="metric-label">{label}</div>
                <div class="metric-icon">{icon}</div>
            </div>
            <div class="metric-value" style="color:{color}">
                {value}
            </div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def panel_heading(title, subtitle=""):
    st.markdown(
        f"""
        <div class="panel-title">{title}</div>
        <div class="panel-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def dark_plot(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#cbd5e1",
            size=12
        ),
        title_font=dict(size=15, color="#f1f5f9"),
        margin=dict(l=20, r=20, t=55, b=30),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        ),
        xaxis=dict(
            gridcolor="#263650",
            zerolinecolor="#334155"
        ),
        yaxis=dict(
            gridcolor="#263650",
            zerolinecolor="#334155"
        )
    )
    return fig


def get_auc():
    try:
        return float(metrics["EBM"]["AUC"])
    except (KeyError, TypeError, ValueError):
        return None


def get_map_center(data):
    if "lat" in data.columns and "lng" in data.columns:
        valid = data[["lat", "lng"]].dropna()
        if not valid.empty:
            return [
                float(valid["lat"].mean()),
                float(valid["lng"].mean())
            ]
    return [40.7128, -74.0060]


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">🛡 RoadSafe <span>AI</span></div>
        <div class="brand-caption">INTELLIGENT ROAD SAFETY</div>
        <div class="status-pill">● SYSTEM ONLINE</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-label">WORKSPACE</div>',
                unsafe_allow_html=True)

    pages = [
        "Overview",
        "Accident Map",
        "Risk Prediction",
        "Hotspot Clusters",
        "Model Insights"
    ]

    page_icons = {
        "Overview": "📊",
        "Accident Map": "🗺️",
        "Risk Prediction": "🎯",
        "Hotspot Clusters": "📍",
        "Model Insights": "🧠"
    }

    # Navigation callback: update the radio value before Streamlit reruns.
    # This avoids modifying a widget's session state after it is instantiated.
    def go_to_risk_prediction():
        st.session_state["navigation"] = "Risk Prediction"

    if "navigation" not in st.session_state:
        st.session_state["navigation"] = "Overview"

    page = st.radio(
        "Navigation",
        pages,
        format_func=lambda x: f"{page_icons[x]}   {x}",
        label_visibility="collapsed",
        key="navigation"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">DATASET</div>',
                unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="small-muted">
        <b style="color:#e2e8f0">Records loaded</b><br>
        {len(df):,} records<br><br>
        <b style="color:#e2e8f0">Prediction model</b><br>
        Explainable Boosting Machine<br><br>
        <b style="color:#e2e8f0">Clustering</b><br>
        DBSCAN (check pipeline implementation)
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("RoadSafe AI · Research Prototype")


# =========================================================
# 6. OVERVIEW DASHBOARD
# =========================================================

if page == "Overview":

    page_heading(
        "WORKSPACE / OVERVIEW",
        "Road Safety Dashboard 🚦",
        "Monitor accident records, severity patterns and model performance."
    )

    high_risk = int(df["high_severity"].sum())
    other_records = len(df) - high_risk
    auc = get_auc()

    # Top action
    top_left, top_right = st.columns([4, 1])

    with top_left:
        st.markdown(
            '<span class="status-pill">● Analytics active</span>',
            unsafe_allow_html=True
        )

    with top_right:
        st.button(
            "＋ Risk Prediction",
            use_container_width=True,
            on_click=go_to_risk_prediction
        )

    st.write("")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL RECORDS",
            f"{len(df):,}",
            "📁",
            "#38bdf8",
            "Records in current dataset"
        )

    with c2:
        metric_card(
            "HIGH-SEVERITY RECORDS",
            f"{high_risk:,}",
            "⚠️",
            "#fb7185",
            "Labeled high severity"
        )

    with c3:
        metric_card(
            "OTHER RECORDS",
            f"{other_records:,}",
            "✓",
            "#34d399",
            "Not labeled high severity"
        )

    with c4:
        metric_card(
            "EBM AUC SCORE",
            f"{auc:.3f}" if auc is not None else "N/A",
            "◈",
            "#a78bfa",
            "Model evaluation metric"
        )

    st.write("")
    st.write("")

    # Charts - use native Streamlit containers so the charts render INSIDE the cards.
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        with st.container(border=True):
            panel_heading(
                "Accident Risk by Hour",
                "Average high-severity label across hours"
            )

            if "hour_of_day" in df.columns:
                hour_data = (
                    df.groupby("hour_of_day")["high_severity"]
                    .mean()
                    .reset_index()
                )

                fig = px.area(
                    hour_data,
                    x="hour_of_day",
                    y="high_severity",
                    markers=True
                )

                fig.update_traces(
                    line=dict(color="#38bdf8", width=3),
                    fillcolor="rgba(56,189,248,0.12)",
                    marker=dict(size=5)
                )

                fig.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Average high-severity label",
                    yaxis_tickformat=".0%"
                )

                st.plotly_chart(
                    dark_plot(fig, 340),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True}
                )
            else:
                st.warning("hour_of_day column is missing.")

    with right:
        with st.container(border=True):
            panel_heading(
                "High Severity by Weather",
                "Number of high-severity records per weather category"
            )

            if "weather" in df.columns:
                weather_data = (
                    df.groupby("weather")["high_severity"]
                    .sum()
                    .reset_index()
                    .sort_values("high_severity", ascending=False)
                )

                fig = px.bar(
                    weather_data,
                    x="weather",
                    y="high_severity",
                    text="high_severity"
                )

                fig.update_traces(
                    marker_color="#3b82f6",
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_layout(
                    xaxis_title="Weather",
                    yaxis_title="High-severity records"
                )

                st.plotly_chart(
                    dark_plot(fig, 340),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True}
                )
            else:
                st.warning("weather column is missing.")

    # Model comparison
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading(
        "Model Performance",
        "Evaluation metrics returned by your ML pipeline"
    )

    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(
        metrics_df,
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown(
        """
        <div class="info-box">
        <b>Research focus</b><br>
        This platform combines accident severity classification,
        spatial clustering and interpretable model explanations.
        The dashboard summarizes the supplied dataset and the
        evaluation results produced by the training pipeline.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Note: If generate_sample_data() is used, these are synthetic "
        "records and should not be presented as actual accident statistics."
    )


# =========================================================
# 7. ACCIDENT MAP
# =========================================================

elif page == "Accident Map":

    page_heading(
        "SPATIAL ANALYTICS",
        "Accident Map 🗺️",
        "Explore accident locations and visualize areas with concentrated records."
    )

    if not {"lat", "lng"}.issubset(df.columns):
        st.error("Dataset must contain lat and lng columns.")
        st.stop()

    map_data = df.dropna(subset=["lat", "lng"]).copy()

    if map_data.empty:
        st.warning("No valid geographic records are available.")
        st.stop()

    f1, f2, f3 = st.columns(3)

    with f1:
        map_view = st.selectbox(
            "Map visualization",
            ["Heatmap", "Accident points"]
        )

    with f2:
        severity_filter = st.selectbox(
            "Severity",
            ["All records", "High severity", "Other records"]
        )

    with f3:
        weather_options = ["All"] + sorted(
            map_data["weather"].dropna().astype(str).unique().tolist()
        ) if "weather" in map_data.columns else ["All"]

        weather_filter = st.selectbox(
            "Weather",
            weather_options
        )

    filtered = map_data.copy()

    if severity_filter == "High severity":
        filtered = filtered[filtered["high_severity"] == 1]
    elif severity_filter == "Other records":
        filtered = filtered[filtered["high_severity"] == 0]

    if weather_filter != "All" and "weather" in filtered.columns:
        filtered = filtered[filtered["weather"].astype(str) == weather_filter]

    st.markdown(
        f"""
        <div class="status-pill">
        ● {len(filtered):,} records displayed
        </div>
        """,
        unsafe_allow_html=True
    )

    center = get_map_center(filtered if not filtered.empty else map_data)

    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True
    )

    if not filtered.empty:

        if map_view == "Heatmap":
            heat_points = filtered[
                ["lat", "lng"]
            ].astype(float).values.tolist()

            HeatMap(
                heat_points,
                radius=17,
                blur=22,
                min_opacity=0.35
            ).add_to(m)

        else:
            display_data = filtered.head(3000)

            for _, row in display_data.iterrows():
                is_high = int(row["high_severity"]) == 1

                color = "#fb7185" if is_high else "#38bdf8"

                popup_text = (
                    f"Severity: {'High' if is_high else 'Other'}<br>"
                    f"Weather: {row.get('weather', 'N/A')}<br>"
                    f"Hour: {row.get('hour_of_day', 'N/A')}<br>"
                    f"Road: {row.get('road_type', 'N/A')}"
                )

                folium.CircleMarker(
                    location=[float(row["lat"]), float(row["lng"])],
                    radius=4 if is_high else 3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.75,
                    weight=1,
                    popup=folium.Popup(popup_text, max_width=250)
                ).add_to(m)

    else:
        st.info("No records match the selected filters.")

    st_folium(
        m,
        use_container_width=True,
        height=620,
        key="accident_map"
    )

    st.markdown(
        """
        <div class="small-muted">
        <span style="color:#fb7185">●</span> High severity
        &nbsp;&nbsp;&nbsp;
        <span style="color:#38bdf8">●</span> Other records
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 8. RISK PREDICTION
# =========================================================

elif page == "Risk Prediction":

    page_heading(
        "PREDICTIVE INTELLIGENCE",
        "Risk Prediction 🎯",
        "Enter road and environmental conditions to estimate high-severity probability."
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Prediction target:</b> high_severity<br>
        The model estimates the probability of the high-severity
        class for the input conditions. This is not a guarantee
        that an accident will occur.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    input_col, result_col = st.columns([0.9, 1.35], gap="large")

    with input_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "Input Conditions",
            "Configure the conditions for the model"
        )

        # Set input ranges from data when possible
        lat_min = float(df["lat"].min()) if "lat" in df.columns else 40.6
        lat_max = float(df["lat"].max()) if "lat" in df.columns else 40.8
        lng_min = float(df["lng"].min()) if "lng" in df.columns else -74.1
        lng_max = float(df["lng"].max()) if "lng" in df.columns else -73.9

        if lat_min == lat_max:
            lat_min -= 0.01
            lat_max += 0.01

        if lng_min == lng_max:
            lng_min -= 0.01
            lng_max += 0.01

        default_lat = min(max(40.7128, lat_min), lat_max)
        default_lng = min(max(-74.0060, lng_min), lng_max)

        lat = st.number_input(
            "Latitude",
            min_value=lat_min,
            max_value=lat_max,
            value=default_lat,
            format="%.6f"
        )

        lng = st.number_input(
            "Longitude",
            min_value=lng_min,
            max_value=lng_max,
            value=default_lng,
            format="%.6f"
        )

        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=18
        )

        weather_values = (
            sorted(df["weather"].dropna().astype(str).unique().tolist())
            if "weather" in df.columns else
            ["Clear", "Rain", "Snow", "Fog"]
        )

        road_values = (
            sorted(df["road_type"].dropna().astype(str).unique().tolist())
            if "road_type" in df.columns else
            ["Highway", "Urban", "Rural"]
        )

        weather = st.selectbox("Weather", weather_values)
        road_type = st.selectbox("Road Type", road_values)

        predict_btn = st.button(
            "⚡ Calculate Risk",
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "Prediction Result",
            "Model output and local explanation"
        )

        if predict_btn:

            input_df = pd.DataFrame(
                [[lat, lng, hour, weather, road_type]],
                columns=[
                    "lat",
                    "lng",
                    "hour_of_day",
                    "weather",
                    "road_type"
                ]
            )

            try:
                prob = float(
                    ebm_model.predict_proba(input_df)[0, 1]
                )

                if prob > 0.7:
                    risk_name = "HIGH RISK"
                    risk_color = "#fb7185"
                    risk_message = (
                        "The model assigns a relatively high "
                        "probability to the high-severity class."
                    )
                elif prob > 0.4:
                    risk_name = "MEDIUM RISK"
                    risk_color = "#fbbf24"
                    risk_message = (
                        "The model assigns an intermediate "
                        "probability to the high-severity class."
                    )
                else:
                    risk_name = "LOW RISK"
                    risk_color = "#34d399"
                    risk_message = (
                        "The model assigns a lower probability "
                        "to the high-severity class."
                    )

                st.markdown(
                    f"""
                    <div style="
                        background:#14223a;
                        border:1px solid #334155;
                        border-radius:14px;
                        padding:23px;
                        text-align:center;
                        margin-bottom:20px;">
                        <div style="color:#94a3b8;font-size:.85rem">
                            HIGH-SEVERITY CLASS PROBABILITY
                        </div>
                        <div style="
                            color:{risk_color};
                            font-size:3.5rem;
                            font-weight:800;
                            margin:8px 0;">
                            {prob*100:.1f}%
                        </div>
                        <div style="
                            display:inline-block;
                            color:{risk_color};
                            border:1px solid {risk_color};
                            background:#0b1120;
                            border-radius:30px;
                            padding:7px 18px;
                            font-weight:800;">
                            {risk_name}
                        </div>
                        <div style="
                            color:#94a3b8;
                            font-size:.83rem;
                            margin-top:14px;">
                            {risk_message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Gauge
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={
                            "suffix": "%",
                            "font": {
                                "size": 30,
                                "color": "#f1f5f9"
                            }
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickcolor": "#94a3b8"
                            },
                            "bar": {
                                "color": risk_color,
                                "thickness": 0.25
                            },
                            "bgcolor": "#111c30",
                            "borderwidth": 0,
                            "steps": [
                                {
                                    "range": [0, 40],
                                    "color": "#15382f"
                                },
                                {
                                    "range": [40, 70],
                                    "color": "#4b3a1a"
                                },
                                {
                                    "range": [70, 100],
                                    "color": "#4b2430"
                                }
                            ]
                        },
                        title={
                            "text": "Risk Probability",
                            "font": {
                                "color": "#cbd5e1",
                                "size": 15
                            }
                        }
                    )
                )

                gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#f1f5f9",
                    height=250,
                    margin=dict(l=25, r=25, t=55, b=15)
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

                # Local explanation
                st.markdown("#### Why did the model make this prediction?")

                local_exp = ebm_model.explain_local(
                    input_df,
                    y=None
                )

                exp_data = local_exp.data(0)

                feature_names = exp_data.get("names", [])
                scores = exp_data.get("scores", [])

                if len(feature_names) == len(scores) and len(scores) > 0:

                    explanation_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Contribution": scores
                    })

                    explanation_df = explanation_df.sort_values(
                        "Contribution",
                        key=lambda s: s.abs(),
                        ascending=True
                    )

                    bar_colors = [
                        "#fb7185" if value > 0 else "#34d399"
                        for value in explanation_df["Contribution"]
                    ]

                    fig = go.Figure(
                        go.Bar(
                            x=explanation_df["Contribution"],
                            y=explanation_df["Feature"].astype(str),
                            orientation="h",
                            marker_color=bar_colors,
                            hovertemplate=(
                                "%{y}<br>Contribution: %{x:.3f}"
                                "<extra></extra>"
                            )
                        )
                    )

                    fig.update_layout(
                        xaxis_title="Contribution (model scale)",
                        yaxis_title="Feature",
                        showlegend=False
                    )

                    st.plotly_chart(
                        dark_plot(fig, 350),
                        use_container_width=True
                    )

                    st.caption(
                        "Positive and negative contributions are shown "
                        "on the model's explanation scale. They are not "
                        "causal effects."
                    )
                else:
                    st.info(
                        "The model did not return local feature "
                        "contributions in the expected format."
                    )

            except Exception as e:
                st.error(
                    "Prediction failed. Check that the input feature "
                    "names and data types match the trained model."
                )
                st.code(str(e))

        else:
            st.markdown(
                """
                <div style="
                    border:1px dashed #3b4b63;
                    border-radius:13px;
                    padding:55px 20px;
                    text-align:center;
                    margin-top:20px;">
                    <div style="font-size:2.7rem">🎯</div>
                    <div style="
                        color:#e2e8f0;
                        font-weight:700;
                        font-size:1.1rem;
                        margin:10px 0;">
                        Ready to analyze
                    </div>
                    <div style="color:#94a3b8;font-size:.88rem">
                        Enter the conditions and click
                        Calculate Risk to view the result.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 9. HOTSPOT CLUSTERS
# =========================================================

elif page == "Hotspot Clusters":

    page_heading(
        "SPATIAL INTELLIGENCE",
        "Hotspot Clusters 📍",
        "Explore geographic groups of accident records identified by clustering."
    )

    if not {"lat", "lng"}.issubset(clustered_data.columns):
        st.error("Clustered dataset must contain lat and lng.")
        st.stop()

    if "cluster" not in clustered_data.columns:
        st.error(
            "The clustering function must return a 'cluster' column."
        )
        st.stop()

    cluster_df = clustered_data.dropna(
        subset=["lat", "lng", "cluster"]
    ).copy()

    cluster_df["cluster"] = cluster_df["cluster"].astype(int)

    noise_count = int((cluster_df["cluster"] == -1).sum())
    valid_clusters = sorted(
        x for x in cluster_df["cluster"].unique() if x != -1
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "CLUSTERS FOUND",
            f"{len(valid_clusters):,}",
            "◉",
            "#38bdf8",
            "Non-noise cluster labels"
        )

    with c2:
        metric_card(
            "NOISE RECORDS",
            f"{noise_count:,}",
            "◌",
            "#fbbf24",
            "Labeled -1 by DBSCAN"
        )

    with c3:
        metric_card(
            "RECORDS ANALYZED",
            f"{len(cluster_df):,}",
            "📍",
            "#a78bfa",
            "Valid coordinate records"
        )

    st.write("")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "Geographic Cluster Map",
            "Each color represents a cluster; gray points are noise."
        )

        center = get_map_center(cluster_df)

        cluster_map = folium.Map(
            location=center,
            zoom_start=11,
            tiles="CartoDB dark_matter"
        )

        palette = [
            "#38bdf8", "#fb7185", "#34d399",
            "#a78bfa", "#fbbf24", "#f97316",
            "#2dd4bf", "#e879f9", "#60a5fa",
            "#a3e635", "#f472b6", "#818cf8"
        ]

        for _, row in cluster_df.head(4000).iterrows():
            label = int(row["cluster"])

            if label == -1:
                color = "#64748b"
            else:
                color = palette[label % len(palette)]

            folium.CircleMarker(
                location=[
                    float(row["lat"]),
                    float(row["lng"])
                ],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=1,
                popup=f"Cluster: {label}"
            ).add_to(cluster_map)

        st_folium(
            cluster_map,
            use_container_width=True,
            height=530,
            key="cluster_map"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "Cluster Distribution",
            "Number of records assigned to each cluster"
        )

        counts = (
            cluster_df["cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        counts.columns = ["Cluster", "Records"]

        counts["Cluster"] = counts["Cluster"].astype(str)

        fig = px.bar(
            counts,
            x="Cluster",
            y="Records",
            text="Records"
        )

        fig.update_traces(
            marker_color="#38bdf8",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Cluster ID",
            yaxis_title="Records"
        )

        st.plotly_chart(
            dark_plot(fig, 330),
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "Interpretation",
            "How to read the cluster output"
        )

        st.markdown("""
        - **Cluster IDs:** groups of geographically close records.
        - **Noise (-1):** points not assigned to a cluster.
        - **Large clusters:** areas containing many records.
        - **Small clusters:** smaller spatial groups.
        
        DBSCAN uses spatial density. This page does not establish
        statistical significance or temporal clustering.
        """)

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 10. MODEL INSIGHTS / EBM EXPLAINABILITY
# =========================================================

elif page == "Model Insights":

    page_heading(
        "MODEL INTELLIGENCE",
        "Model Insights 🧠",
        "Inspect the model's performance, global feature effects and interpretability."
    )

    auc = get_auc()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "MODEL",
            "EBM",
            "🧠",
            "#38bdf8",
            "Explainable Boosting Machine"
        )

    with c2:
        metric_card(
            "AUC",
            f"{auc:.3f}" if auc is not None else "N/A",
            "◈",
            "#a78bfa",
            "Area under ROC curve"
        )

    with c3:
        metric_card(
            "FEATURES",
            f"{len(df.columns):,}",
            "▤",
            "#34d399",
            "Dataset columns (including target)"
        )

    st.write("")

    tab1, tab2 = st.tabs([
        "Global Explainability",
        "Performance Metrics"
    ])

    with tab1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "Global Feature Effects",
            "Select a feature to inspect its learned shape function."
        )

        try:
            global_exp = ebm_model.explain_global()
            all_global_data = global_exp.data()

            feature_names = all_global_data.get("names", [])

            if not feature_names:
                st.warning("No global feature names were returned.")
            else:
                selected_feature = st.selectbox(
                    "Select feature",
                    feature_names
                )

                idx = feature_names.index(selected_feature)
                feature_data = global_exp.data(idx)

                scores = feature_data.get("scores", [])
                names = feature_data.get("names", [])

                if len(scores) == 0:
                    st.info(
                        "No shape-function scores are available "
                        "for this feature."
                    )

                elif isinstance(names, list) and len(names) == len(scores):
                    # A feature can be categorical or continuous.
                    # Use the available bins and scores as returned.
                    x_values = [
                        str(x) for x in names
                    ]

                    fig = go.Figure()

                    if feature_data.get("type") == "univariate":
                        # Use a line for ordered continuous features
                        # when their labels can be converted to numbers.
                        numeric_x = pd.to_numeric(
                            pd.Series(names),
                            errors="coerce"
                        )

                        if numeric_x.notna().all():
                            sort_idx = np.argsort(
                                numeric_x.to_numpy()
                            )
                            x_sorted = numeric_x.iloc[
                                sort_idx
                            ].tolist()
                            y_sorted = [
                                scores[i] for i in sort_idx
                            ]

                            fig.add_trace(
                                go.Scatter(
                                    x=x_sorted,
                                    y=y_sorted,
                                    mode="lines+markers",
                                    line=dict(
                                        color="#38bdf8",
                                        width=3
                                    ),
                                    marker=dict(size=6)
                                )
                            )
                        else:
                            fig.add_trace(
                                go.Bar(
                                    x=x_values,
                                    y=scores,
                                    marker_color="#3b82f6"
                                )
                            )

                        fig.update_layout(
                            xaxis_title=selected_feature,
                            yaxis_title="Contribution (model scale)"
                        )

                        st.plotly_chart(
                            dark_plot(fig, 420),
                            use_container_width=True
                        )

                    else:
                        st.info(
                            "This term is not a standard univariate "
                            "shape function. Inspect the feature "
                            "details below."
                        )

                    with st.expander("View returned explanation data"):
                        st.json({
                            "feature": selected_feature,
                            "type": feature_data.get("type"),
                            "scores": scores,
                            "names": x_values
                        })

                else:
                    st.info(
                        "The returned feature data has a different "
                        "structure and cannot be displayed as a "
                        "standard one-dimensional chart."
                    )

        except Exception as e:
            st.error("Unable to load global explanation.")
            st.code(str(e))

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "Evaluation Summary",
            "Metrics calculated by the training pipeline"
        )

        metrics_df = pd.DataFrame(metrics).T

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading(
            "About the EBM model",
            "Interpretability and classification"
        )

        st.markdown("""
        **Explainable Boosting Machine (EBM)** is the classification
        model used by this application.

        The model learns feature effects and can provide local and
        global explanations. The global view describes learned
        patterns across the dataset; the local view shows the
        contributions associated with an individual prediction.

        These explanations describe the model's behavior. They do
        not, by themselves, prove that a feature causes an accident.
        """)

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        display:flex;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:10px;
        color:#64748b;
        font-size:.78rem;
        padding:5px 0 15px 0;">
        <span>🛡 RoadSafe AI · Intelligent Road Safety Analytics</span>
        <span>Research Prototype · EBM + Spatial Clustering</span>
    </div>
    """,
    unsafe_allow_html=True
)