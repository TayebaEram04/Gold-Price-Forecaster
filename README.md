# 🪙 Gold Price Forecaster

## 📌 Project Overview

Gold Price Forecaster is a machine learning project that predicts the **next-day gold price** using historical gold market data.

The project analyzes historical price trends and creates useful features such as moving averages, daily returns, and volatility. A **Random Forest Regression** model is then trained to forecast the next-day gold price.

The project also includes an interactive **Streamlit web application** for viewing historical prices and predictions.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze historical gold price data
- Identify useful market trends and patterns
- Perform feature engineering
- Train a machine learning model
- Predict the next-day gold price
- Evaluate model performance
- Build an interactive web application using Streamlit

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – Data processing
- **NumPy** – Numerical calculations
- **Scikit-learn** – Machine learning
- **Matplotlib** – Data visualization
- **Joblib** – Model saving and loading
- **Streamlit** – Interactive web application
- **Git & GitHub** – Version control

---

## 📊 Dataset

The project uses historical daily gold-price data containing:

- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume

The dataset contains historical **XAU/USD gold-price data**.

---

## 🧠 Machine Learning Model

The project uses a:

**Random Forest Regressor**

Instead of directly predicting the next day's price, the model predicts the **next-day percentage change (return)**.

The predicted return is then converted into the estimated next-day gold price.

### Features Used

The model uses the following features:

- Open Price
- High Price
- Low Price
- Close Price
- Previous Close
- 7-Day Moving Average
- 14-Day Moving Average
- 30-Day Moving Average
- Daily Return
- 7-Day Volatility
- Day
- Month
- Year

---

## 🔄 Project Workflow

```text
Historical Gold Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Next-Day Return Calculation
        ↓
Train/Test Split
        ↓
Random Forest Model
        ↓
Model Evaluation
        ↓
Next-Day Price Prediction
        ↓
Streamlit Dashboard---

## 📈 Model Performance

The current model achieved approximately:

| Metric | Result |
|---|---:|
| MAE | 46.95 |
| RMSE | 67.85 |
| R² Score | 0.9729 |

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Measures how well the model explains the variation in the test data.

**Note:** R² Score should not be interpreted as prediction accuracy.

---

## 💻 Streamlit Application

The project includes an interactive web application built using **Streamlit**.

The application displays:

- 📈 Historical Gold Price Chart
- 💰 Latest Market Information
- 🤖 Next-Day Gold Price Prediction
- 📊 Expected Price Change

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/TayebaEram04/Gold-Price-Forecaster.git### 2. Open the Project Folder

```bash
cd Gold-Price-Forecaster### 3. Create a Virtual Environment

```bash
python -m venv venv### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activatepip install 
pip install -r requirements.txt
python train_model.py
This will train the Random Forest model and create:

```text
gold_model.pkl
streamlit run app.py
Gold-Price-Forecaster/
│
├── app.py
├── train_model.py
├── gold_price.csv
├── XAU_1d_data.jsonl
├── requirements.txt
├── .gitignore
├── README.md
└── gold_model.pkl
> **Note:** `gold_model.pkl` is generated locally by running `train_model.py` and is not included in the GitHub repository because it is larger than GitHub's regular 100 MB file-size limit.

---

## 🔮 Prediction Process

The prediction process works as follows:

1. Historical gold-price data is loaded.
2. The data is cleaned and sorted by date.
3. Technical and time-based features are created.
4. The Random Forest model predicts the next-day percentage return.
5. The predicted return is converted into an estimated next-day gold price.
6. The prediction is displayed through the Streamlit dashboard.

---

## ⚠️ Disclaimer

This project is created for **educational and demonstration purposes only**.

The predictions are generated using historical market data and a machine learning model. They should **not be considered financial advice or a guarantee of future gold prices**.

---

## 👩‍💻 Conclusion

This project demonstrates how **Python, data analysis, feature engineering, machine learning, and Streamlit** can be combined to create an interactive gold-price forecasting application.

It provides a practical example of applying machine learning techniques to historical financial market data.