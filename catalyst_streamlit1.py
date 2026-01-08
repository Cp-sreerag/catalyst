# catalyst_streamlit.py
import os
import json
import requests
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
st.set_page_config(page_title="Catalyst — Dark Forecast (INR)", layout="wide", initial_sidebar_state="collapsed")

DARK_BG = "#0e1117"
CARD_BG = "#0f1724"
TEXT = "#E6EEF3"
MUTED = "#9AA7B2"
ACCENT = "#00E676"      # green
ACCENT_DOWN = "#FF5252" # red

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
# Helpers
# -------------------------
def fetch_data_try_nse(symbol_raw: str, start="2010-01-01", end=None):
    """
    Try symbol.NS first (Indian NSE). If no data, fall back to symbol as-is.
    Returns DataFrame and the used symbol string.
    """
    s = symbol_raw.strip().upper()
    # If user explicitly included exchange suffix (contains '.'), use as-is
    if "." in s:
        used = s
        df = yf.Ticker(used).history(start=start, end=end).reset_index()
        return df, used

    # Try NSE suffix first
    candidate = s + ".NS"
    df = yf.Ticker(candidate).history(start=start, end=end)
    if df is not None and not df.empty:
        return df.reset_index(), candidate

    # fallback: try raw symbol
    df2 = yf.Ticker(s).history(start=start, end=end)
    if df2 is not None and not df2.empty:
        return df2.reset_index(), s

    # nothing found
    return pd.DataFrame(), s

def get_ticker_currency(symbol):
    try:
        info = yf.Ticker(symbol).info
        currency = info.get("currency", None)
        return currency
    except Exception:
        return None

def fetch_usd_to_inr_rate():
    """
    Use exchangerate.host free API to fetch USD->INR spot rate.
    Returns float or None on failure.
    """
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR", timeout=6)
        if r.status_code == 200:
            data = r.json()
            rate = data.get("rates", {}).get("INR", None)
            if rate:
                return float(rate)
    except Exception:
        pass
    return None

def compute_indicators(df):
    df = df.copy()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA100"] = df["Close"].rolling(window=100).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    # RSI (14 simple method)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    return df

def make_plotly_chart(df, preds_aligned, pred_future_price, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Close",
        line=dict(color="#81D4FA"), hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>"
    ))
    # MAs
    for ma, color, name in [("MA50", "#90CAF9", "MA50"), ("MA100", "#A5D6A7", "MA100"), ("MA200", "#F48FB1", "MA200")]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], mode="lines", line=dict(color=color, dash="dash"), name=name, hoverinfo="skip"))

    # Predicted historical (aligned, None where missing)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=preds_aligned, mode="lines", name="Model Pred (hist)",
        line=dict(color="#FFD54F", dash="dot"), opacity=0.95,
        hovertemplate="%{x|%Y-%m-%d}<br>Pred: %{y:.2f}<extra></extra>"
    ))

    # Next day line + marker
    if pred_future_price is not None:
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        next_date = last_date + BDay(1)
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[df["Close"].iloc[-1], pred_future_price],
            mode="lines+markers", name="Next Day Forecast",
            line=dict(color=ACCENT, width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=[next_date], y=[pred_future_price], mode="markers",
            marker=dict(size=12, color=(ACCENT if pred_future_price >= df["Close"].iloc[-1] else ACCENT_DOWN)),
            name="Predicted Close"
        ))

    # Volume bars on secondary y-axis
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"], name="Volume", marker=dict(color="#263238"), opacity=0.25, yaxis="y2", hoverinfo="skip"
    ))

    # Layout dark theme
    fig.update_layout(
    template=None,
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
    font=dict(color=TEXT),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=80, r=40, t=40, b=40),  # 👈 MORE SPACE ON LEFT
    xaxis=dict(showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor"),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1f2a33",
        zeroline=False,
        automargin=True,        # 👈 IMPORTANT
        tickformat=",.2f"       # 👈 Proper numeric formatting
    ),
    yaxis2=dict(
        overlaying="y",
        side="right",
        showgrid=False,
        title="Volume",
        automargin=True
    )
)

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
st.title("Catalyst — Dark Forecast (TradingView style, INR)")

col1, col2 = st.columns([3,1])
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:600; font-size:14px'>Settings</div>", unsafe_allow_html=True)
    user_input = st.text_input("Enter Ticker (Indian default)", value="RELIANCE")
    start = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
    st.markdown("<div class='small' style='margin-top:6px'>By default the app tries <code>TICKER.NS</code> (NSE). If not found it falls back to raw ticker (US/others).</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:700; font-size:20px'>Ticker: {user_input.upper()}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Fetch data (try NSE first) - ROBUST VERSION
# -------------------------
with st.spinner("Fetching data..."):
    try:
        df_raw, used_symbol = fetch_data_try_nse(user_input, start=start)
    except Exception as e:
        st.error("Error fetching data. Please check your internet connection.")
        st.stop()

# If no data returned -> friendly message and stop
if df_raw is None or df_raw.empty:
    st.error(
        f"No data found for '{user_input}'.\n\n"
        "Possible causes:\n"
        "- Spelling mistake or wrong ticker symbol\n"
        "- The ticker is not listed on NSE/NYSE\n\n"
        "Try: RELIANCE, TCS, INFY, or an explicit symbol like AAPL (US)."
    )
    st.stop()

# Determine currency and try to convert to INR if needed
currency = get_ticker_currency(used_symbol)
fx_rate = None
price_multiplier_note = ""

# Best-effort: if currency unknown assume INR for NSE, USD otherwise
if currency is None:
    currency = "INR" if used_symbol.endswith(".NS") else "USD"

# Start with df set to raw copy — ensures df is always defined
df = df_raw.copy()

if currency != "INR":
    st.info(f"Detected currency for {used_symbol}: {currency} → attempting conversion to INR.")
    try:
        fx_rate = fetch_usd_to_inr_rate()
    except Exception:
        fx_rate = None

    if not fx_rate:
        # conversion failed — gracefully continue with original currency and inform user
        st.warning("Failed to fetch live USD→INR rate. Showing prices in the ticker's native currency.")
        price_multiplier_note = f"Prices shown in {currency} (conversion failed)."
    else:
        # If currency is USD we already have fx_rate; if different, try direct conversion
        if currency != "USD":
            try:
                q = requests.get(f"https://api.exchangerate.host/latest?base={currency}&symbols=INR", timeout=6).json()
                r2 = q.get("rates", {}).get("INR", None)
                if r2:
                    fx_rate = float(r2)
            except Exception:
                # keep previously fetched fx_rate (USD->INR) as fallback
                pass

        # perform conversion if we have a valid fx_rate
        if fx_rate:
            for col in ["Open", "High", "Low", "Close"]:
                # guard in case some columns missing or non-numeric
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce") * fx_rate
            df["Value_INR"] = df["Close"] * df.get("Volume", 0)
            price_multiplier_note = f"Converted from {currency} to INR at rate {fx_rate:.4f}"
        else:
            price_multiplier_note = f"Prices shown in {currency} (no conversion rate available)."
else:
    price_multiplier_note = "Prices in INR (native ticker)."

# Compute technical indicators
df = compute_indicators(df)

# Prepare data for model
data = df[["Close"]].copy()
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data.values)

time_step = 60
X_windows = create_lstm_windows(scaled_data, time_step=time_step)

# Load model if present
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
    preds_scaled = model.predict(X_windows, verbose=0)
    try:
        preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    except Exception:
        preds = scaler.inverse_transform(preds_scaled).flatten()
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
# Safely fetch last two closes (handle tiny datasets)
if len(df) < 2:
    st.error("Not enough historical data to compute trends.")
    st.stop()

yesterday = float(df["Close"].iloc[-2])
today = float(df["Close"].iloc[-1])
today_change = today - yesterday
today_pct = (today_change / yesterday) * 100.0 if yesterday != 0 else 0.0

forecast_price = pred_future_price
forecast_change = None
forecast_pct = None
forecast_trend = "—"
if forecast_price is not None:
    forecast_change = forecast_price - today
    forecast_pct = (forecast_change / today) * 100.0 if today != 0 else 0.0
    forecast_trend = "UP" if forecast_change > 0 else ("DOWN" if forecast_change < 0 else "FLAT")

# layout metrics
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric(label="Yesterday Close (INR)", value=f"₹{yesterday:,.2f}")
with mcol2:
    sign = "+" if today_change >= 0 else ""
    st.metric(label="Today Close (INR)", value=f"₹{today:,.2f}", delta=f"{sign}{today_change:,.2f} ({sign}{today_pct:.2f}%)")
with mcol3:
    if forecast_price is not None:
        sign2 = "+" if forecast_change >= 0 else ""
        st.metric(label="Predicted Next Close (INR)", value=f"₹{forecast_price:,.2f}", delta=f"{sign2}{forecast_change:.2f} ({sign2}{forecast_pct:.2f}%)")
    else:
        st.metric(label="Predicted Next Close (INR)", value="Model missing", delta=None)
with mcol4:
    st.markdown(f"<div class='card'><div style='font-weight:600'>Forecast Trend</div><div style='font-size:18px; margin-top:6px; color:{ACCENT if forecast_change and forecast_change>=0 else ACCENT_DOWN}'>{forecast_trend}</div></div>", unsafe_allow_html=True)

# -------------------------
# Plotly Chart
# -------------------------
fig = make_plotly_chart(df, preds_aligned, pred_future_price, used_symbol)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# RSI mini-chart with heading + Model info
# -------------------------
rcol1, rcol2 = st.columns([3,1])
with rcol1:
    rfig = go.Figure()
    rfig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI (14)", line=dict(color="#FFD180")))
    rfig.update_layout(
        title=dict(text="RSI (14) — Relative Strength Index", x=0.01, font=dict(color=TEXT, size=14)),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        height=220
    )
    rfig.update_yaxes(range=[0,100])
    st.plotly_chart(rfig, use_container_width=True)

with rcol2:
    st.markdown("<div class='card'><div style='font-weight:600'>Model Info</div>", unsafe_allow_html=True)
    if model is not None:
        st.markdown(f"<div class='small'>Model loaded: <code>{os.path.basename(model_path)}</code></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>Input window: {time_step} days</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small'>Model unavailable — historical preds disabled</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small' style='margin-top:6px'>{price_multiplier_note}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Notes & explanation
# -------------------------
st.markdown("---")
st.markdown("<div class='card'><div style='font-weight:600'>Notes</div><div class='small' style='margin-top:8px'>This dashboard uses a simple LSTM-style next-day predictor trained on past 60 days. It is suitable for trend estimation and demos; do not treat predictions as financial advice. Prices are shown in INR; conversion uses a live FX rate where applicable. The app tries <code>TICKER.NS</code> first to prioritise NSE tickers. Use Plotly controls to zoom, pan, and inspect values.</div></div>", unsafe_allow_html=True)
