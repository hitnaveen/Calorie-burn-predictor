import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from model import (
    load_data, preprocess, train_all_models, predict_calories, feature_importance, FEATURES
)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# DESIGN TOKENS  (thermal / vitals-monitor identity)
# ----------------------------------------------------------------------------
BG = "#0B0F14"
PANEL = "#121821"
PANEL_ALT = "#161D27"
BORDER = "#232E3B"
TEXT = "#EDEFF2"
TEXT_DIM = "#8B96A5"
BLUE = "#3E6FC4"
AMBER = "#F2A93B"
EMBER = "#E8462A"
GREEN = "#3DDC97"
GRADIENT = f"linear-gradient(90deg, {BLUE} 0%, {AMBER} 55%, {EMBER} 100%)"

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
:root {{
  --bg: {BG}; --panel: {PANEL}; --panel-alt: {PANEL_ALT}; --border: {BORDER};
  --text: {TEXT}; --text-dim: {TEXT_DIM};
  --blue: {BLUE}; --amber: {AMBER}; --ember: {EMBER}; --green: {GREEN};
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
  background: radial-gradient(circle at 15% 0%, #131a24 0%, {BG} 45%) fixed;
  color: var(--text);
}}

/* Hide default streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}

h1, h2, h3, .display {{
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.01em;
}}

.mono {{ font-family: 'JetBrains Mono', monospace; }}

/* ---------- HERO ---------- */
.hero-eyebrow {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--green);
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 14px;
}}
.pulse-dot {{
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 0 rgba(61,220,151,0.6);
  animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 rgba(61,220,151,0.55); }}
  70% {{ box-shadow: 0 0 0 8px rgba(61,220,151,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(61,220,151,0); }}
}}

.hero-title {{
  font-size: 3.1rem;
  font-weight: 700;
  line-height: 1.08;
  margin: 0 0 10px 0;
  background: {GRADIENT};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.hero-sub {{
  color: var(--text-dim);
  font-size: 1.05rem;
  max-width: 640px;
  line-height: 1.6;
  margin-bottom: 26px;
}}

.stat-row {{ display: flex; gap: 34px; flex-wrap: wrap; margin-bottom: 6px; }}
.stat-block {{ border-left: 2px solid var(--border); padding-left: 14px; }}
.stat-num {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.5rem; font-weight: 700; color: var(--text);
}}
.stat-label {{
  font-size: 0.72rem; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.08em;
}}

/* ---------- PANELS / CARDS ---------- */
.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px 24px;
}}
.panel-alt {{
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px 24px;
}}
.panel-title {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.0rem; font-weight: 600;
  color: var(--text); margin-bottom: 4px;
}}
.panel-caption {{
  font-size: 0.8rem; color: var(--text-dim); margin-bottom: 16px;
}}

/* monitor bezel around the gauge */
.bezel {{
  background: linear-gradient(180deg, #0F151D 0%, #0B0F14 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 10px 10px 4px 10px;
  position: relative;
}}
.bezel::before {{
  content: "";
  position: absolute; top: 14px; right: 18px;
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px 2px rgba(61,220,151,0.6);
}}
.bezel-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-dim); padding: 6px 10px 0 10px;
}}

.result-kcal {{
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700; font-size: 2.6rem;
  text-align: center; color: var(--text);
  margin: 0;
}}
.result-caption {{
  text-align: center; color: var(--text-dim); font-size: 0.85rem;
  margin-bottom: 4px;
}}
.equivalence {{
  background: var(--panel-alt);
  border: 1px dashed var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  font-size: 0.88rem;
  color: var(--text-dim);
  margin-top: 10px;
}}
.equivalence b {{ color: var(--amber); }}

/* badges */
.badge {{
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--text-dim);
  margin-right: 6px;
  margin-bottom: 6px;
}}
.badge-best {{ border-color: var(--green); color: var(--green); }}

/* section eyebrow */
.eyebrow {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 6px;
}}

/* Tabs restyle */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px; border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  height: 42px;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  color: var(--text-dim);
  background: transparent;
}}
.stTabs [aria-selected="true"] {{
  color: var(--text) !important;
  border-bottom: 2px solid var(--ember) !important;
}}

/* Buttons */
.stButton>button {{
  background: {GRADIENT};
  color: #0B0F14;
  font-weight: 700;
  border: none;
  border-radius: 10px;
  padding: 0.6rem 1.2rem;
  font-family: 'Space Grotesk', sans-serif;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton>button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(232,70,42,0.25);
  color: #0B0F14;
}}

/* Sliders accent */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
  background-color: var(--ember) !important;
}}

hr {{ border-color: var(--border); }}

::-webkit-scrollbar {{ width: 10px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 6px; }}

.footer-note {{
  text-align: center; color: var(--text-dim); font-size: 0.78rem;
  padding: 30px 0 10px 0; font-family: 'JetBrains Mono', monospace;
}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA + MODEL (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_data():
    raw = load_data("data/calories.csv")
    return preprocess(raw)

@st.cache_resource(show_spinner=False)
def get_trained_bundle(df_hash):
    df = get_clean_data()
    return train_all_models(df)

try:
    df = get_clean_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.spinner("Calibrating models on 15,000 training sessions..."):
    bundle = get_trained_bundle(len(df))

best_model = bundle["best_model"]
best_name = bundle["best_name"]
scaler = bundle["scaler"]
metrics_df = bundle["metrics"]
best_row = metrics_df.iloc[0]

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-eyebrow"><span class="pulse-dot"></span> LIVE MODEL &middot; {len(df):,} TRAINING SESSIONS</div>
<div class="hero-title">Your body already tells you<br>how many calories you burnt.</div>
<div class="hero-sub">
This tool reads the same signals a wearable does — heart rate, body temperature,
duration, and your own biometrics — and turns them into a calorie estimate using
a model trained on {len(df):,} real workout sessions, instead of a generic
population-average formula.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, num, label in zip(
    [c1, c2, c3, c4],
    [f"{best_row['Val R2']*100:.2f}%", f"{best_row['Val MAE']:.2f} kcal", best_name, f"{len(df):,}"],
    ["Validation accuracy (R²)", "Mean absolute error", "Best performing model", "Sessions trained on"]
):
    col.markdown(f"""
    <div class="stat-block">
      <div class="stat-num">{num}</div>
      <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_predict, tab_explore, tab_lab, tab_about = st.tabs(
    ["🔥  Predict", "📊  Data Explorer", "🧪  Model Lab", "ℹ️  About"]
)

# ============================================================================
# TAB 1 — PREDICT
# ============================================================================
with tab_predict:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Session Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Tell us about your workout</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-caption">All seven fields feed the model — nothing is guessed.</div>', unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            age = st.slider("Age (years)", 15, 80, 28)
            height = st.slider("Height (cm)", 130, 220, 170)
            weight = st.slider("Weight (kg)", 35, 140, 68)
        with colB:
            duration = st.slider("Duration (minutes)", 1, 60, 20)
            heart_rate = st.slider("Heart rate (bpm)", 60, 130, 100)
            body_temp = st.slider("Body temperature (°C)", 36.0, 42.0, 40.0, step=0.1)

        st.write("")
        predict_clicked = st.button("⚡  Predict calories burnt", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="bezel">', unsafe_allow_html=True)
        st.markdown('<div class="bezel-label">◎ Calorie Readout</div>', unsafe_allow_html=True)

        if "last_pred" not in st.session_state:
            st.session_state.last_pred = None

        if predict_clicked:
            pred = predict_calories(best_model, scaler, gender, age, height, weight, duration, heart_rate, body_temp)
            st.session_state.last_pred = pred
            st.session_state.last_inputs = dict(
                gender=gender, age=age, height=height, weight=weight,
                duration=duration, heart_rate=heart_rate, body_temp=body_temp
            )

        pred = st.session_state.last_pred

        gauge_max = 350
        gauge_val = pred if pred is not None else 0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            number={"suffix": " kcal", "font": {"family": "JetBrains Mono", "size": 40, "color": TEXT}},
            gauge={
                "axis": {"range": [0, gauge_max], "tickcolor": TEXT_DIM, "tickfont": {"color": TEXT_DIM, "size": 10}},
                "bar": {"color": EMBER, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 100], "color": "#16324D"},
                    {"range": [100, 220], "color": "#4A3418"},
                    {"range": [220, gauge_max], "color": "#3D1712"},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": TEXT},
            height=260,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if pred is not None:
            per_min = pred / max(st.session_state.last_inputs["duration"], 1)
            st.markdown(f"""
            <div class="result-caption">≈ {per_min:.1f} kcal / minute at this intensity</div>
            """, unsafe_allow_html=True)

            pizza_slices = pred / 285
            banana = pred / 105
            walk_min = pred / 4.5
            st.markdown(f"""
            <div class="equivalence">
              That's roughly <b>{pizza_slices:.1f} slices of pizza</b>, <b>{banana:.1f} bananas</b>,
              or the same energy as a <b>{walk_min:.0f}-minute brisk walk</b>.
            </div>
            """, unsafe_allow_html=True)

            report = (
                f"CALORIE BURN PREDICTION REPORT\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"{'-'*40}\n"
                f"Gender: {gender}\nAge: {age}\nHeight: {height} cm\nWeight: {weight} kg\n"
                f"Duration: {duration} min\nHeart Rate: {heart_rate} bpm\nBody Temp: {body_temp} C\n"
                f"{'-'*40}\n"
                f"Predicted Calories Burnt: {pred:.2f} kcal\n"
                f"Model used: {best_name} (Val R2 = {best_row['Val R2']:.4f}, MAE = {best_row['Val MAE']:.2f} kcal)\n"
            )
            st.download_button("⬇  Download this result", report, file_name="calorie_prediction.txt", use_container_width=True)
        else:
            st.markdown('<div class="result-caption">Set your session details and predict to see a reading here.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2 — DATA EXPLORER
# ============================================================================
with tab_explore:
    st.markdown('<div class="eyebrow">Training Data</div>', unsafe_allow_html=True)
    st.markdown('### The dataset behind the model')
    st.caption(f"{len(df):,} logged exercise sessions — age, biometrics, heart rate, body temperature, and measured calorie burn.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sessions", f"{len(df):,}")
    m2.metric("Avg. calories burnt", f"{df['Calories'].mean():.0f} kcal")
    m3.metric("Avg. duration", f"{df['Duration'].mean():.0f} min")
    m4.metric("Avg. heart rate", f"{df['Heart_Rate'].mean():.0f} bpm")

    st.write("")
    gcol, acol = st.columns([1, 2])
    with gcol:
        gender_filter = st.multiselect("Filter by gender", ["Male", "Female"], default=["Male", "Female"])
    with acol:
        age_range = st.slider("Filter by age range", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))

    gender_codes = [0 if g == "Male" else 1 for g in gender_filter] if gender_filter else [0, 1]
    fdf = df[(df.Gender.isin(gender_codes)) & (df.Age.between(*age_range))].copy()
    fdf["Gender_label"] = fdf.Gender.map({0: "Male", 1: "Female"})

    plotly_theme = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Duration vs. Calories</div>', unsafe_allow_html=True)
        fig = px.scatter(
            fdf.sample(min(2000, len(fdf))), x="Duration", y="Calories", color="Gender_label",
            color_discrete_map={"Male": BLUE, "Female": EMBER}, opacity=0.5,
        )
        fig.update_layout(**plotly_theme, height=340)
        fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Heart Rate vs. Calories</div>', unsafe_allow_html=True)
        fig = px.scatter(
            fdf.sample(min(2000, len(fdf))), x="Heart_Rate", y="Calories", color="Gender_label",
            color_discrete_map={"Male": BLUE, "Female": EMBER}, opacity=0.5,
        )
        fig.update_layout(**plotly_theme, height=340)
        fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    left2, right2 = st.columns(2)
    with left2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Calorie burn distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(fdf, x="Calories", nbins=40, color_discrete_sequence=[AMBER])
        fig.update_layout(**plotly_theme, height=320)
        fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Feature correlation</div>', unsafe_allow_html=True)
        corr = fdf[FEATURES + ["Calories"]].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale=[BLUE, PANEL_ALT, EMBER], aspect="auto"
        )
        fig.update_layout(**plotly_theme, height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3 — MODEL LAB
# ============================================================================
with tab_lab:
    st.markdown('<div class="eyebrow">Under the Hood</div>', unsafe_allow_html=True)
    st.markdown('### Five models, one winner')
    st.caption("Each candidate was trained on an identical 80/20 split and scored on held-out validation sessions.")

    plotly_theme = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Validation accuracy (R²) by model</div>', unsafe_allow_html=True)
        sorted_metrics = metrics_df.sort_values("Val R2")
        colors = [GREEN if m == best_name else BLUE for m in sorted_metrics["Model"]]
        fig = go.Figure(go.Bar(
            x=sorted_metrics["Val R2"], y=sorted_metrics["Model"], orientation="h",
            marker_color=colors, text=[f"{v:.4f}" for v in sorted_metrics["Val R2"]],
            textposition="outside", textfont=dict(family="JetBrains Mono", color=TEXT),
        ))
        fig.update_layout(**plotly_theme, height=320, xaxis=dict(range=[0, 1.08], gridcolor=BORDER))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">What drives the prediction</div>', unsafe_allow_html=True)
        fi = feature_importance(best_model)
        if fi is not None:
            fig = go.Figure(go.Bar(
                x=fi["Importance"], y=fi["Feature"], orientation="h",
                marker_color=AMBER,
            ))
            fig.update_layout(**plotly_theme, height=320, xaxis=dict(gridcolor=BORDER, tickformat=".0%"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Full comparison table</div>', unsafe_allow_html=True)
    show_df = metrics_df.copy()
    show_df["Val R2"] = show_df["Val R2"].map(lambda x: f"{x:.4f}")
    show_df["Val MAE"] = show_df["Val MAE"].map(lambda x: f"{x:.2f}")
    show_df["Val RMSE"] = show_df["Val RMSE"].map(lambda x: f"{x:.2f}")
    show_df["Train MAE"] = show_df["Train MAE"].map(lambda x: f"{x:.2f}")
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.markdown(f"""
    <span class="badge badge-best">★ {best_name} — deployed model</span>
    <span class="badge">Scaler: StandardScaler</span>
    <span class="badge">Split: 80 / 20</span>
    <span class="badge">Features: {len(FEATURES)}</span>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 4 — ABOUT
# ============================================================================
with tab_about:
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Academic Project</div>', unsafe_allow_html=True)
        st.markdown('### Calories Burnt Prediction Using Machine Learning')
        st.markdown("""
        <div style="color:#8B96A5; line-height:1.7; font-size:0.94rem;">
        Submitted in partial fulfillment of the Master of Computer Application (MCA) degree,
        Department of Computer Science, University Post Graduate College,
        Osmania University, Secunderabad.
        <br><br>
        <b style="color:#EDEFF2;">Author</b> — Anushka Nizamabad (Roll No. 1010-23-862-005)<br>
        <b style="color:#EDEFF2;">Guide</b> — Dr. G. Vamshi Krishna, Asst. Professor<br>
        <b style="color:#EDEFF2;">Industry Partner</b> — Elle Logitech Service Solutions Pvt Ltd
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        <div style="color:#8B96A5; line-height:1.7; font-size:0.94rem;">
        This app trains five regression models — Linear, Ridge, Lasso, Random Forest,
        and XGBoost — on 15,000 real exercise sessions, each logged with age, gender,
        height, weight, workout duration, heart rate, and body temperature against
        measured calories burnt. The best-performing model on held-out data is used
        for every live prediction on the <b style="color:#EDEFF2;">Predict</b> tab.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-alt">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Tech stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <span class="badge">Python</span>
        <span class="badge">Streamlit</span>
        <span class="badge">scikit-learn</span>
        <span class="badge">XGBoost</span>
        <span class="badge">Plotly</span>
        <span class="badge">Pandas</span>
        <span class="badge">NumPy</span>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="panel-title">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#8B96A5; font-size:0.88rem; line-height:1.6;">
        15,000 rows &middot; 9 columns<br>
        Features: Gender, Age, Height, Weight, Duration, Heart Rate, Body Temperature<br>
        Target: Calories burnt (kcal)
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="footer-note">CALORIE BURN PREDICTOR · BUILT WITH STREAMLIT · MODEL: {best_name.upper()}</div>
""", unsafe_allow_html=True)
