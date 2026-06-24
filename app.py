import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide"
)

st.title("📈 Stock Forecast Dashboard")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="AAPL"
)

years = st.sidebar.slider(
    "Historical Data (Years)",
    min_value=1,
    max_value=10,
    value=5
)

run = st.sidebar.button("Run Analysis")

if run:

    with st.spinner("Downloading stock data..."):

        df = yf.download(
            ticker,
            period=f"{years}y",
            auto_adjust=True,
            progress=False
        )

    if df.empty:
        st.error("No data found.")
        st.stop()

    # Fix yfinance MultiIndex issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ------------------------
    # Technical Indicators
    # ------------------------

    df["SMA50"] = df["Close"].rolling(50).mean()

    df["SMA200"] = df["Close"].rolling(200).mean()

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    # RSI

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # ------------------------
    # Metrics
    # ------------------------

    latest_price = float(df["Close"].dropna().iloc[-1])

    sma50 = float(df["SMA50"].dropna().iloc[-1])

    sma200 = float(df["SMA200"].dropna().iloc[-1])

    rsi = float(df["RSI"].dropna().iloc[-1])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Price",
        f"${latest_price:.2f}"
    )

    col2.metric(
        "SMA 50",
        f"${sma50:.2f}"
    )

    col3.metric(
        "SMA 200",
        f"${sma200:.2f}"
    )

    col4.metric(
        "RSI",
        f"{rsi:.2f}"
    )

    # ------------------------
    # Price Chart
    # ------------------------

    st.subheader("Interactive Price Chart")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA50"],
            name="SMA 50"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA200"],
            name="SMA 200"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA20"],
            name="EMA 20"
        )
    )

    fig.update_layout(
        height=600,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # RSI Chart
    # ------------------------

    st.subheader("RSI")

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI"
        )
    )

    rsi_fig.add_hline(y=70)

    rsi_fig.add_hline(y=30)

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )

    # ------------------------
    # MACD Chart
    # ------------------------

    st.subheader("MACD")

    macd_fig = go.Figure()

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD"
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Signal"],
            name="Signal"
        )
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True
    )

    # ------------------------
    # Prophet Forecast
    # ------------------------

    st.subheader("Forecast")

    prophet_df = df.reset_index()[["Date", "Close"]]

    prophet_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )

    model.fit(prophet_df)

    target_date = pd.Timestamp("2027-06-30")

    last_date = prophet_df["ds"].max()

    forecast_days = (target_date - last_date).days

    if forecast_days < 30:
        forecast_days = 365

    future = model.make_future_dataframe(
        periods=forecast_days,
        freq="B"
    )

    forecast = model.predict(future)

    forecast_fig = go.Figure()

    forecast_fig.add_trace(
        go.Scatter(
            x=prophet_df["ds"],
            y=prophet_df["y"],
            name="Historical"
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Forecast"
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            name="Upper CI"
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            name="Lower CI"
        )
    )

    forecast_fig.update_layout(
        height=600
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

    # ------------------------
    # June 2027 Prediction
    # ------------------------

    closest_idx = (
        forecast["ds"] - target_date
    ).abs().idxmin()

    predicted_price = float(
        forecast.loc[closest_idx, "yhat"]
    )

    lower_ci = float(
        forecast.loc[closest_idx, "yhat_lower"]
    )

    upper_ci = float(
        forecast.loc[closest_idx, "yhat_upper"]
    )

    st.success(
        f"""
        Forecasted Price near June 2027: ${predicted_price:.2f}

        95% Confidence Interval:
        ${lower_ci:.2f} to ${upper_ci:.2f}
        """
    )

    # ------------------------
    # Download CSV
    # ------------------------

    st.download_button(
        "Download Forecast CSV",
        forecast.to_csv(index=False),
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv"
    )
