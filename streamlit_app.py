# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from app.preprocessing import preprocess_data
from app.models import run_forecasting_model
from app.evaluation import evaluate_model
import plotly.graph_objs as go
from datetime import timedelta

# --- Color Palette ---
PASTEL_PINK = "#FFDCDC"   # Main highlight
PASTEL_PEACH = "#FFF2EB"  # Secondary highlight
PASTEL_YELLOW = "#FFE8CD" # Accent
PASTEL_ORANGE = "#FFD6BA" # Accent/gradient
BLACK = "#000"
CARD_BG = "#FFF2EB"  # Card background for contrast
SOFT_RED = "#FF8282"

# Set page config with pastel background
st.set_page_config(
    page_title="Data Forever - Smart Forecaster",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for new pastel palette
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {PASTEL_PEACH};
            color: {BLACK};
        }}
        .stSidebar, .css-1d391kg, .css-1cpxqw2, .css-1offfwp, .css-1kyxreq, .css-1dp5vir, .css-1vq4p4l, .css-1lcbmhc, .css-1r6slb0, .css-1v3fvcr, .css-1b7of8t, .css-1q8dd3e, .css-1c7y2kd, .css-1vzeuhh, .css-1v0mbdj, .css-1d3w5wq {{
            background-color: {CARD_BG} !important;
            color: {BLACK} !important;
            border-radius: 10px;
            border: 2px solid {PASTEL_PINK};
        }}
        .stButton>button {{
            background: linear-gradient(90deg, {PASTEL_PINK} 0%, {PASTEL_ORANGE} 100%);
            color: {BLACK};
            border: none;
            padding: 0.5rem 1.2rem;
            border-radius: 5px;
            font-weight: bold;
            font-size: 1.1em;
            box-shadow: 0 2px 8px {PASTEL_YELLOW}44;
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, {PASTEL_YELLOW} 0%, {PASTEL_ORANGE} 100%);
            color: {BLACK};
        }}
        .stRadio>div, .stSelectbox>div, .stExpander, .stDataFrame, .stMarkdown, .stTextInput>div>div>input, .stDataFrame th, .stDataFrame td {{
            color: {BLACK} !important;
        }}
        .stDataFrame {{
            background-color: {CARD_BG};
        }}
        .stSuccess, .stInfo, .stError {{
            background-color: {CARD_BG};
            color: {BLACK};
            border-left: 5px solid {PASTEL_PINK};
        }}
        .stCheckbox>label, .stRadio>label, .stSelectbox>label {{
            color: {BLACK} !important;
        }}
        .stMarkdown code {{
            background-color: {PASTEL_YELLOW};
            color: {BLACK};
        }}
        .stFileUploader, .stFileUploader label, .stFileUploader span {{
            color: {BLACK} !important;
        }}
        .stFileUploader, .stFileUploader>div {{
            background-color: {CARD_BG} !important;
            border-radius: 8px;
            border: 1.5px solid {PASTEL_PINK};
        }}
        .stTextInput>div>div>input {{
            background-color: {CARD_BG};
            color: {BLACK};
        }}
        .stTextInput>div>div>input:focus {{
            border: 1.5px solid {PASTEL_ORANGE};
        }}
        .stExpanderHeader {{
            color: {PASTEL_PINK} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Logo/Header ---
st.markdown(f"""
    <div style='padding:1.5em 0 1em 0;'>
        <span style='font-size:2.5em;font-weight:800;font-family:sans-serif;color:#000;'>Data Forever </span>
        <span style='font-size:2.5em;font-weight:800;font-family:sans-serif;color:{SOFT_RED};'>Suite</span>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar: Upload & Settings ---
with st.sidebar:
    st.markdown(f"<div style='padding:0.7em 0 0.7em 0;'><span style='font-size:1.7em;font-weight:800;font-family:sans-serif;color:#000;'>Upload </span><span style='font-size:1.7em;font-weight:800;font-family:sans-serif;color:{SOFT_RED};'>Data</span></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    demo_mode = st.checkbox("Demo Mode (use dummy data)", value=not uploaded_file)

# --- Step 1: Data Loading ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV Loaded (preview below)")
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        st.stop()
elif demo_mode:
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    values = np.random.randn(120).cumsum() + 100
    df = pd.DataFrame({"date": dates, "value": values})
    st.info("Demo mode: Using dummy time series data.")
else:
    st.info("📁 Please upload a CSV file or enable Demo Mode in the sidebar.")
    st.stop()

# --- Step 2: Data Preview & Column Selection ---
st.markdown(f"<div style='padding:0.7em 0 0.7em 0;'><span style='font-size:2em;font-weight:800;font-family:sans-serif;color:#000;'>Data Preview & </span><span style='font-size:2em;font-weight:800;font-family:sans-serif;color:{SOFT_RED};'>Preprocessing</span></div>", unsafe_allow_html=True)
st.dataframe(pd.concat([df.head(5), df.tail(5)]), use_container_width=True)

# Modern card-style container for preprocessing
st.markdown(f"""
    <div style='background:{CARD_BG}; border-radius:16px; box-shadow:0 2px 16px #0002; padding:2em 2em 1.5em 2em; margin-bottom:2em;'>
        <span style='font-size:1.5em;font-weight:800;font-family:sans-serif;color:#000;'>Preprocessing </span><span style='font-size:1.5em;font-weight:800;font-family:sans-serif;color:{SOFT_RED};'>Setup</span>
        <div style='color:{SOFT_RED};margin-bottom:1em;font-size:1.1em;'>Select your time and value columns, resampling, and missing value handling options below.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    date_column = st.selectbox(
        "<span style='color:#000'>Select Date/Time Column</span>",
        df.columns,
        help="Column containing date/time information.",
        format_func=lambda x: f"🕒 {x}" if 'date' in x.lower() or 'time' in x.lower() else x,
        key="date_col",
        label_visibility="visible"
    )
with col2:
    value_column = st.selectbox(
        "<span style='color:#000'>Select Value Column</span>",
        df.columns,
        help="Column to forecast (numeric time series)",
        format_func=lambda x: f"📈 {x}" if x != date_column else x,
        key="value_col",
        label_visibility="visible"
    )

st.markdown(f"<hr style='border:0.5px solid {PASTEL_YELLOW};margin:1.5em 0 1em 0;'>", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    resample_option = st.radio(
        "<span style='color:#000'>Resample Frequency</span>",
        ["D", "W", "M"],
        format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x],
        horizontal=True,
        key="resample_radio",
        label_visibility="visible"
    )
with col4:
    missing_option = st.radio(
        "<span style='color:#000'>Handle Missing Values</span>",
        ["drop", "ffill", "bfill"],
        format_func=lambda x: {"drop": "Drop", "ffill": "Forward Fill", "bfill": "Backward Fill"}[x],
        horizontal=True,
        key="missing_radio",
        label_visibility="visible"
    )

st.markdown(f"<div style='text-align:right;margin-top:2em;'>", unsafe_allow_html=True)
preprocess_btn = st.button(
    "Preprocess Data",
    type="primary",
    key="preprocess_btn",
    help="Apply preprocessing to your selected columns.",
)
st.markdown("</div></div>", unsafe_allow_html=True)

if preprocess_btn or ("df_preprocessed" in st.session_state):
    if not (preprocess_btn or "df_preprocessed" in st.session_state):
        st.stop()
    if preprocess_btn:
        try:
            df_preprocessed = preprocess_data(df, date_column, value_column, missing_option, resample_option)
            st.session_state.df_preprocessed = df_preprocessed
            st.session_state.value_column = value_column
            st.success("✅ Data Preprocessed")
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            st.stop()
    else:
        df_preprocessed = st.session_state.df_preprocessed
        value_column = st.session_state.value_column
    st.dataframe(df_preprocessed.tail(10), use_container_width=True)
else:
    st.stop()

# --- Step 3: Model Selection & Forecasting ---
st.markdown(f"<div style='padding:0.7em 0 0.7em 0;'><span style='font-size:2em;font-weight:800;font-family:sans-serif;color:#000;'>Model </span><span style='font-size:2em;font-weight:800;font-family:sans-serif;color:{SOFT_RED};'>Forecasting</span></div>", unsafe_allow_html=True)
col1, col2 = st.columns([2,1])
with col1:
    model_choice = st.radio(
        "Select Forecasting Model",
        ["ARIMA", "Prophet", "LSTM"],
        help="ARIMA: Classic statistical model. Prophet: Facebook's model for business time series. LSTM: Deep learning for complex patterns.",
        horizontal=True
    )
with col2:
    horizon_label = st.radio("Forecast Horizon", ["1 Week", "1 Month", "1 Quarter"], horizontal=True)
    horizon_map = {"1 Week": 7, "1 Month": 30, "1 Quarter": 90}
    forecast_periods = horizon_map[horizon_label]

run_forecast = st.button("Run Forecast 🔮", type="primary")

if run_forecast:
    with st.spinner(f"Running {model_choice} forecast..."):
        try:
            forecast_df = run_forecasting_model(
                model_name=model_choice,
                df=df_preprocessed,
                target_column=value_column,
                forecast_periods=forecast_periods
            )
            # Align for evaluation (if possible)
            if len(df_preprocessed) >= forecast_periods:
                true_vals = df_preprocessed[value_column].iloc[-forecast_periods:]
                pred_vals = forecast_df.iloc[:forecast_periods, 0]
                metrics = evaluate_model(true_vals.values, pred_vals.values)
            else:
                metrics = None
            st.success(f"✅ Forecast completed using {model_choice}!")
            # --- Step 4: Results & Visualization ---
            st.markdown(f"<div style='padding:0.7em 0 0.7em 0;'><span style='font-size:1.5em;font-weight:800;font-family:sans-serif;color:#000;'>Forecast </span><span style='font-size:1.5em;font-weight:800;font-family:sans-serif;color:{SOFT_RED};'>Results</span></div>", unsafe_allow_html=True)
            # Forecast Table
            st.markdown("#### Forecast Table")
            styled_df = forecast_df.style.apply(lambda x: [f'background-color: {PASTEL_PINK}; color: {BLACK}' for _ in x], axis=1)
            st.dataframe(styled_df, use_container_width=True)
            # Download Button
            csv = forecast_df.to_csv().encode('utf-8')
            st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv", help="Download the forecasted values as CSV.")
            # Visualization
            st.markdown("#### Visualization")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_preprocessed.index, y=df_preprocessed[value_column], 
                                   mode='lines+markers', name='Actual', 
                                   line=dict(color=PASTEL_PINK)))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df.iloc[:,0], 
                                   mode='lines+markers', name='Forecast', 
                                   line=dict(color=PASTEL_ORANGE)))
            fig.add_vrect(x0=forecast_df.index[0], x1=forecast_df.index[-1], 
                         fillcolor=PASTEL_YELLOW, opacity=0.1, line_width=0)
            fig.update_layout(
                template="plotly_white",
                plot_bgcolor=PASTEL_PEACH,
                paper_bgcolor=PASTEL_PEACH,
                font=dict(color=BLACK),
                legend=dict(orientation="h"),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            # Evaluation Metrics
            if metrics:
                st.markdown("#### Evaluation Metrics")
                metrics_df = pd.DataFrame([metrics]).T
                metrics_df.columns = ['Value']
                st.dataframe(metrics_df.style.format({'Value': '{:.3f}'}))
        except Exception as e:
            st.error(f"<span style='color:{PASTEL_PINK}'>Error in forecasting: {str(e)}</span>", unsafe_allow_html=True)
else:
    st.info("Configure options and click 'Run Forecast' to see results.")

# --- End-to-End Test Section (for CI/CD or local testing) ---
def run_end_to_end_test():
    st.markdown("""
    <div style='background-color:#222;padding:1em;border-radius:8px;margin-top:2em;'>
        <h3 style='color:#FFA500;'>End-to-End Dummy Data Test</h3>
    </div>
    """, unsafe_allow_html=True)
    # Dummy data
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    values = np.random.randn(60).cumsum() + 100
    dummy_df = pd.DataFrame({"date": dates, "value": values})
    st.write("Dummy Data Preview:", dummy_df.head())
    # Preprocess
    try:
        processed = preprocess_data(dummy_df, "date", "value", missing_option="drop", freq_option="D")
        st.success("Preprocessing successful!")
        st.dataframe(processed.head())
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return
    # Test all models
    for model in ["ARIMA", "Prophet", "LSTM"]:
        st.markdown(f"#### Testing {model}")
        try:
            forecast = run_forecasting_model(model, processed, "value", 14)
            st.dataframe(forecast.head())
            st.success(f"{model} forecast successful!")
        except Exception as e:
            st.error(f"{model} forecast failed: {e}")

# Only run test if in test mode (for CI/CD or local check)
if st.sidebar.checkbox("Run End-to-End Dummy Test (for CI/CD)"):
    run_end_to_end_test()

# --- Footer ---
st.markdown(f"""
    <hr style='border:1.5px solid {PASTEL_YELLOW};margin-top:2em;'>
    <div style='text-align:center; color:{PASTEL_ORANGE}; margin-top:1em;'>
        <b>Tip:</b> Try running with demo data and tweak the flow for your needs.<br>
        <span style='color:{PASTEL_PINK}'>Data Forever</span> &copy; 2024
    </div>
""", unsafe_allow_html=True)
