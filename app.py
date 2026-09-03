import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Gold Price Forecaster",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 Gold Price Forecaster")
st.write(
    "Predict next-day gold prices using historical market trends "
    "and machine learning."
)


# Load dataset
df = pd.read_csv("gold_price.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.reset_index(drop=True)


# Create features
df["Previous_Close"] = df["Close"].shift(1)
df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_14"] = df["Close"].rolling(14).mean()
df["MA_30"] = df["Close"].rolling(30).mean()

df["Daily_Return"] = df["Close"].pct_change()
df["Volatility_7"] = df["Daily_Return"].rolling(7).std()

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year


# Historical chart
st.header("📈 Historical Gold Price")

st.line_chart(
    df.set_index("Date")["Close"]
)


# Load trained model
try:
    model = joblib.load("gold_model.pkl")
except FileNotFoundError:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()


# Latest market information
latest = df.iloc[-1]

st.header("💰 Latest Market Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Close",
    f"{latest['Close']:.2f}"
)

col2.metric(
    "Latest High",
    f"{latest['High']:.2f}"
)

col3.metric(
    "Latest Low",
    f"{latest['Low']:.2f}"
)

col4.metric(
    "Latest Open",
    f"{latest['Open']:.2f}"
)


# Next-day prediction
st.header("🤖 Next-Day Gold Price Prediction")


previous_close = df["Close"].iloc[-1]

ma_7 = df["Close"].tail(7).mean()
ma_14 = df["Close"].tail(14).mean()
ma_30 = df["Close"].tail(30).mean()

daily_return = (
    df["Close"].iloc[-1] /
    df["Close"].iloc[-2]
) - 1

volatility_7 = (
    df["Close"].pct_change().tail(7).std()
)


# Model input
input_data = pd.DataFrame({
    "Open": [latest["Open"]],
    "High": [latest["High"]],
    "Low": [latest["Low"]],
    "Close": [latest["Close"]],
    "Previous_Close": [previous_close],
    "MA_7": [ma_7],
    "MA_14": [ma_14],
    "MA_30": [ma_30],
    "Daily_Return": [daily_return],
    "Volatility_7": [volatility_7],
    "Day": [latest["Date"].day],
    "Month": [latest["Date"].month],
    "Year": [latest["Date"].year]
})


# Predict next-day return
predicted_return = model.predict(input_data)[0]


# Convert return into predicted price
predicted_price = (
    latest["Close"] *
    (1 + predicted_return)
)


st.success(
    f"Predicted next-day gold price: {predicted_price:.2f}"
)


# Show expected movement
price_change = predicted_price - latest["Close"]

percentage_change = (
    price_change / latest["Close"]
) * 100


st.info(
    f"Expected change: {price_change:.2f} "
    f"({percentage_change:.2f}%)"
)


st.warning(
    "This project is an educational machine-learning model. "
    "It should not be used as financial advice or as a guarantee "
    "of future gold prices."
)