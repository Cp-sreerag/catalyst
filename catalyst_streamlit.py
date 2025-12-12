# catalyst_streamlit.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import timedelta
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# -------------------------
# Config & Styling
# -------------------------
st.set_page_config(page_title="Catalyst — Dark Forecast", layout="wide", initial_sidebar_state="collapsed")

DARK_BG = "#0e1117"
CARD_BG = "#0f1724"
TEXT = "#E6EEF3"
MUTED = "#9AA7B2"
ACCENT = "#00E676"  # green
ACCENT_DOWN = "#FF5252"  # red

st.markdown(
    f"""
    <style>
      .stApp {{ background: {DARK_BG}; color: {TEXT}; }}
      .card {{ background: {CARD_BG}; padding: 16px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6);}}
      .metric {{ font-size: 20px; color: {TEXT}; }}
      .muted {{ color: {MUTED}; }}
      .small {{ font-size:12px; color:{MUTED}; }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helper functions
# -------------------------
def fetch_data(symbol, start="2010-01-01", end=None):
    t = yf.Ticker(symbol)
    df = t.history(start=start, end=end)
    if df is None:
        return pd.DataFrame()
    df = df.reset_index()
    return df

def compute_indicators(df):
    df = df.copy()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA100"] = df["Close"].rolling(window=100).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    # RSI (14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    return df

def make_plotly_chart(df, preds_aligned, pred_future_price, symbol):
    # Main price + MAs + predicted overlay
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Close",
        line=dict(color="#81D4FA"), hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>"
    ))
    # MAs
    for ma, color, name in [("MA50", "#90CAF9", "MA50"), ("MA100", "#A5D6A7", "MA100"), ("MA200", "#F48FB1", "MA200")]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], mode="lines", line=dict(color=color, dash="dash"), name=name, hoverinfo="skip"))

    # Predicted historical (aligned, may have nulls)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=preds_aligned, mode="lines", name="Model Pred (hist)",
        line=dict(color="#FFD54F", dash="dot"), opacity=0.9,
        hovertemplate="%{x|%Y-%m-%d}<br>Pred: %{y:.2f}<extra></extra>"
    ))

    # Forecast next day marker/line
    if pred_future_price is not None:
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        next_date = last_date + BDay(1)
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[df["Close"].iloc[-1], pred_future_price],
            mode="lines+markers", name="Next Day Forecast",
            line=dict(color="#00E676", width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=[next_date], y=[pred_future_price], mode="markers",
            marker=dict(size=12, color="#00E676" if pred_future_price >= df["Close"].iloc[-1] else "#FF5252"),
            name="Predicted Close"
        ))

    # Volume as bars (secondary y)
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"], name="Volume", marker=dict(color="#263238"), opacity=0.25, yaxis="y2", hoverinfo="skip"
    ))

    # Layout - dark theme
    fig.update_layout(
        template=None,
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor"),
        yaxis=dict(showgrid=True, gridcolor="#1f2a33", zeroline=False),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume")
    )

    # Range selector like TradingView
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor="#0b1116", activecolor="#1a73e8", font=dict(color=TEXT)
        )
    )

    return fig

def create_lstm_windows(scaled, time_step=60):
    X = []
    for i in range(len(scaled) - time_step):
        X.append(scaled[i:i+time_step, 0])
    X = np.array(X)
    if X.size == 0:
        return np.empty((0, time_step, 1))
    return X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------
# UI
# -------------------------
st.title("Catalyst — Dark Forecast (TradingView style)")

col1, col2 = st.columns([3,1])
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:600; font-size:14px'>Settings</div>", unsafe_allow_html=True)
    ticker = st.text_input("Ticker", value="AAPL").upper()
    start = st.date_input("Start date", value=pd.to_datetime("2010-01-01"))
    # end intentionally left None for live updates
    st.markdown("</div>", unsafe_allow_html=True)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:700; font-size:20px'>{ticker} — Price & Forecast</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Fetch and validate
with st.spinner("Fetching data..."):
    df = fetch_data(ticker, start=start)
if df.empty:
    st.error("No data returned for ticker. Check symbol or try another.")
    st.stop()

# indicators
df = compute_indicators(df)

# Prepare data for model
data = df[["Close"]].copy()
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# create windows for historical rolling predictions (for plotting)
time_step = 60
X_windows = create_lstm_windows(scaled_data, time_step=time_step)

# Load model
model_path = "stock_models.keras"
model = None
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        st.warning("Model file 'stock_models.keras' not found in current folder. Historical predictions will be disabled.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None

preds_aligned = [None] * len(df)
pred_future_price = None

if model is not None and X_windows.size:
    # predict historical windows
    preds_scaled = model.predict(X_windows, verbose=0)
    # preds_scaled shape should be (n,1)
    try:
        preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    except Exception:
        preds = scaler.inverse_transform(preds_scaled).flatten()

    # align predictions: first prediction corresponds to index time_step
    for i, p in enumerate(preds):
        idx = i + time_step
        if idx < len(preds_aligned):
            preds_aligned[idx] = p

    # next-day prediction using last 60 values
    last_60 = scaled_data[-time_step:]
    if last_60.shape[0] == time_step:
        x_input = last_60.reshape(1, time_step, 1)
        next_scaled = model.predict(x_input, verbose=0)
        try:
            next_price = scaler.inverse_transform(next_scaled.reshape(-1,1)).flatten()[0]
        except Exception:
            next_price = scaler.inverse_transform(next_scaled).flatten()[0]
        pred_future_price = float(next_price)

# -------------------------
# Metrics / Trend cards
# -------------------------
yesterday = float(df["Close"].iloc[-2])
today = float(df["Close"].iloc[-1])
today_change = today - yesterday
today_pct = (today_change / yesterday) * 100.0

forecast_price = pred_future_price
forecast_change = None
forecast_pct = None
forecast_trend = "—"
if forecast_price is not None:
    forecast_change = forecast_price - today
    forecast_pct = (forecast_change / today) * 100.0
    forecast_trend = "UP" if forecast_change > 0 else ("DOWN" if forecast_change < 0 else "FLAT")

# layout metrics
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    trend_color = ACCENT if today_change >= 0 else ACCENT_DOWN
    st.metric(label="Yesterday Close", value=f"${yesterday:,.2f}", delta=None)
with mcol2:
    sign = "+" if today_change >= 0 else ""
    st.metric(label="Today Close", value=f"${today:,.2f}", delta=f"{sign}{today_change:,.2f} ({sign}{today_pct:.2f}%)")
with mcol3:
    if forecast_price is not None:
        sign2 = "+" if forecast_change >= 0 else ""
        st.metric(label="Predicted Next Close", value=f"${forecast_price:,.2f}", delta=f"{sign2}{forecast_change:.2f} ({sign2}{forecast_pct:.2f}%)")
    else:
        st.metric(label="Predicted Next Close", value="Model not loaded", delta=None)
with mcol4:
    st.markdown(f"<div class='card'><div style='font-weight:600'>Forecast Trend</div><div style='font-size:18px; margin-top:6px; color:{ACCENT if forecast_change and forecast_change>=0 else ACCENT_DOWN}'>{forecast_trend}</div></div>", unsafe_allow_html=True)

# -------------------------
# Plotly Chart
# -------------------------
fig = make_plotly_chart(df, preds_aligned, pred_future_price, ticker)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# RSI mini-chart + footer
# -------------------------
rcol1, rcol2 = st.columns([3,1])
with rcol1:
    # RSI small plot
    rfig = go.Figure()
    rfig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI (14)", line=dict(color="#FFD180")))
    rfig.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font=dict(color=TEXT), margin=dict(l=10,r=10,t=20,b=10), height=220)
    rfig.update_yaxes(range=[0,100])
    st.plotly_chart(rfig, use_container_width=True)

with rcol2:
    st.markdown("<div class='card'><div style='font-weight:600'>Model Info</div>", unsafe_allow_html=True)
    if model is not None:
        st.markdown(f"<div class='small'>Model loaded: <code>{os.path.basename(model_path)}</code></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>Input window: {time_step} days</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small'>Model unavailable — historical preds disabled</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Notes & explanation
# -------------------------
st.markdown("---")
st.markdown("<div class='card'><div style='font-weight:600'>Notes</div><div class='small' style='margin-top:8px'>This dashboard uses a simple LSTM-style next-day predictor trained on past 60 days. It is suitable for trend estimation and demos; do not treat predictions as financial advice. Use the Plotly controls to zoom, pan, and inspect values. Model predictions shown as dotted line; next-day forecast shown as highlighted marker.</div></div>", unsafe_allow_html=True)
