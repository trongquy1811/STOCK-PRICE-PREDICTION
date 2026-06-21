# MSFT Stock Price Prediction

This project uses Machine Learning (ML) and Deep Learning (DL) techniques to predict the trend and returns of Microsoft (MSFT) stock. It implements a complete pipeline starting from automated data fetching, preprocessing, Exploratory Data Analysis (EDA), to training, evaluating/comparing multiple models, and Hyperparameter Tuning.

## 🚀 Key Features
- **Automated Data Fetching**: Retrieves historical stock prices and currency exchange rates directly from Yahoo Finance (`yfinance`).
- **Exploratory Data Analysis (EDA)**:
  - Visualizes feature relationships using a **Correlation Matrix** heatmap.
  - Explores distributions and pairwise relationships via a **Scatter Matrix**.
  - Analyzes trends and seasonality using **Time Series Seasonal Decomposition**.
- **Multi-Model Evaluation (Regression & Ensemble)**:
  - Baseline linear models: Linear Regression (LR), Lasso, ElasticNet (EN).
  - Non-linear models: K-Nearest Neighbors (KNN), Decision Tree (CART), Support Vector Regression (SVR), Multi-layer Perceptron (MLP).
  - Ensemble methods: AdaBoost (ABR), Gradient Boosting (GBR), Random Forest (RFR), Extra Trees (ETR).
- **Time Series & Deep Learning Models**:
  - Exogenous ARIMA model implementation.
  - Long Short-Term Memory (LSTM) recurrent neural network.
- **Model Optimization**: Automated Grid Search/hyperparameter tuning for the ARIMA model.
- **Evaluation & Visualization**:
  - Robust evaluation using 10-fold cross-validation.
  - Visual comparison of Train vs. Test Mean Squared Error (MSE) across all models.
  - Actual vs. Predicted plots for the tuned ARIMA model.

---

## 📂 Project Structure
```text
STOCK-PRICE-PREDICTION/
├── main.py                    # Main pipeline script (data download, training, and evaluation)
├── Algorithm Comparison.png   # Model performance comparison chart
├── Correlation Matrix.png     # Feature correlation heatmap
├── MSFT.png                   # Actual vs. predicted comparison plot for the tuned ARIMA model
├── README.md                  # Project documentation (this file)
└── .gitignore                 # Git ignore file
```

---

## 🛠️ Requirements & Installation

### 1. Prerequisites
- Python 3.8+
- Essential data science & machine learning libraries:
  - `numpy`
  - `pandas`
  - `yfinance`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `statsmodels`
  - `tensorflow` / `keras`

### 2. Installation
Install all required packages via `pip`:
```bash
pip install numpy pandas yfinance matplotlib seaborn scikit-learn statsmodels tensorflow
```

---

## 📈 Methodology & Data Pipeline

### 1. Feature Engineering
To improve predictive accuracy, the model incorporates several market indicators as features:
- **Target Variable**: 5-day forward log returns of `MSFT`.
- **Correlated Tech Stocks**: `GOOGL`, `IBM`.
- **Exchange Rates**: `JPY=X` (USD/JPY), `GBP=X` (GBP/USD).
- **Market Indexes**: `SPY` (S&P 500), `DIA` (Dow Jones), `^VIX` (CBOE Volatility Index).
- **Lag Features**: Past log returns of MSFT at various intervals (5-day, 15-day, 30-day, 60-day lags).

### 2. Correlation Analysis
The correlation matrix helps identify relationships between predictors and the target variable:

![Correlation Matrix](Correlation%20Matrix.png)

---

## 📊 Evaluation & Results

Models are evaluated using the Mean Squared Error (MSE) metric. Below are the training and testing results:

### 1. Machine Learning Algorithms Comparison
Cross-validation performance and Train/Test error distributions across models:

![Algorithm Comparison](Algorithm%20Comparison.png)

### 2. Tuned ARIMA Forecasting
After performing a grid search for the optimal ARIMA `(p, d, q)` parameters with exogenous variables, the following chart compares actual cumulative returns of MSFT against the model's predictions:

![MSFT Predictions](MSFT.png)

---

## 🏃 Running the Project

To run the complete data fetching, training, and evaluation pipeline, execute:
```bash
python main.py
```
*Note: Running the script may take a few minutes as it trains the LSTM network and performs a grid search for ARIMA.*

---

## 📝 License
This project is developed for educational and research purposes in Quantitative Finance and Machine Learning.