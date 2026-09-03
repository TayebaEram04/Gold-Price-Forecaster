import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv("gold_price.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.reset_index(drop=True)


# 2. Create features
df["Previous_Close"] = df["Close"].shift(1)

df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_14"] = df["Close"].rolling(14).mean()
df["MA_30"] = df["Close"].rolling(30).mean()

df["Daily_Return"] = df["Close"].pct_change()
df["Volatility_7"] = df["Daily_Return"].rolling(7).std()

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year


# 3. Target = next-day percentage change
df["Next_Return"] = (
    df["Close"].shift(-1) / df["Close"]
) - 1

df = df.dropna()


# 4. Select features
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Previous_Close",
    "MA_7",
    "MA_14",
    "MA_30",
    "Daily_Return",
    "Volatility_7",
    "Day",
    "Month",
    "Year"
]

X = df[features]
y = df["Next_Return"]


# 5. Chronological train/test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training data:", len(X_train))
print("Testing data:", len(X_test))


# 6. Create model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# 7. Train model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")


# 8. Predict next-day returns
predicted_returns = model.predict(X_test)


# 9. Convert predicted returns to prices
current_prices = df["Close"].iloc[split_index:].values

predicted_prices = current_prices * (1 + predicted_returns)

actual_prices = df["Close"].shift(-1).iloc[split_index:].values


# Remove final missing value if present
valid = ~np.isnan(actual_prices)

actual_prices = actual_prices[valid]
predicted_prices = predicted_prices[valid]


# 10. Evaluate price predictions
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)


print("\n------------------------------")
print("MODEL PERFORMANCE")
print("------------------------------")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# 11. Plot results
test_dates = df["Date"].iloc[split_index:][valid]

plt.figure(figsize=(12, 6))

plt.plot(
    test_dates,
    actual_prices,
    label="Actual Gold Price"
)

plt.plot(
    test_dates,
    predicted_prices,
    label="Predicted Gold Price"
)

plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.title("Actual vs Predicted Next-Day Gold Price")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# 12. Save model
joblib.dump(model, "gold_model.pkl")

print("\nModel saved as gold_model.pkl")