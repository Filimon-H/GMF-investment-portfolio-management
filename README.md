# 📈 GMF Investment Portfolio Forecasting & Optimization

## 📄 Project Overview
Guide Me in Finance (GMF) Investments is a forward-thinking financial advisory firm specializing in personalized portfolio management.  
This project applies advanced **time series forecasting** and **Modern Portfolio Theory (MPT)** to optimize portfolio allocation across Tesla (TSLA), SPY, and BND, aiming to enhance returns while managing risk.

We used:
- **Historical market data** from Yahoo Finance (`yfinance`)
- **ARIMA** and **LSTM** forecasting models
- **Efficient Frontier optimization**
- **Backtesting** to validate the strategy

---

## 📚 Table of Contents
1. [Features](#-features)
2. [Tasks & Methodology](#-tasks--methodology)
3. [Project Structure](#-project-structure)
4. [Setup Instructions](#-setup-instructions)
5. [Usage](#-usage)
6. [Results Summary](#-results-summary)
7. [Conclusion](#-conclusion)
8. [Reproducibility](#-reproducibility)
9. [Troubleshooting](#-troubleshooting)
10. [License](#-license)

---

## ✨ Features
- **Data Collection** from Yahoo Finance (`yfinance`)
- **Data Cleaning & Processing** with reproducible scripts
- **Exploratory Data Analysis** for trends, seasonality, and volatility
- **ARIMA & LSTM Forecasting** for TSLA
- **Efficient Frontier Optimization** (MPT) for TSLA, SPY, BND
- **Portfolio Backtesting** vs. 60/40 SPY/BND benchmark
- **Results & Visualizations** stored in `results/`

---

## 📊 Tasks & Methodology

### **Task 1 – Data Preprocessing & EDA**
- Collected daily OHLCV data (2015–2025) for **TSLA**, **SPY**, and **BND**.
- Cleaned missing values (interpolation) and normalized closing prices for comparison.
- Analyzed volatility using rolling statistics.
- Performed **stationarity tests (ADF)**:
  - Prices → Non-stationary
  - Returns → Stationary
- **Risk metrics**:
  - **VaR (95%)**: TSLA 5.47%, SPY 1.72%, BND 0.49%
  - **Sharpe Ratios**: TSLA 0.76, SPY 0.74, BND 0.17

---

### **Task 2 – Forecasting Models**
- Implemented **ARIMA** and **LSTM** models for TSLA.
- **ARIMA**:
  - MAE: 62.97, RMSE: 77.96, MAPE: 24.09%
  - Produced flat forecasts, failed to capture volatility/trends.
- **LSTM**:
  - MAE: 10.83, RMSE: 15.11, MAPE: 4.07%
  - Closely tracked actual prices, captured both trends and volatility.
- **Decision**: LSTM selected for future forecasting due to significantly better performance.

---

### **Task 3 – Future Trend Forecasting**
- Generated **12-month forecast** for TSLA using LSTM.
- Prediction: gradual decline, stabilizing near **$240–$250**.
- Added ±5% **confidence band** to show uncertainty.
- Lower expected volatility compared to recent history.
- This forecast used as TSLA’s **expected return** in portfolio optimization.

---

### **Task 4 – Portfolio Optimization (MPT)**
- Used:
  - TSLA forecast return (from Task 3)
  - Historical returns for SPY and BND
  - Covariance matrix from historical daily returns
- Simulated portfolios to generate the **Efficient Frontier**.
- Identified:
  - ⭐ **Max Sharpe Ratio Portfolio**:
    - Return: 9.10%, Volatility: 0.69%, Sharpe: 11.77
    - Weights: TSLA 0.01%, SPY 57.03%, BND 42.96%
  - 🛡 **Min Volatility Portfolio**:
    - Return: 2.74%, Volatility: 0.34%, Sharpe: 5.09
    - Weights: TSLA 0.30%, SPY 6.88%, BND 92.82%

---

### **Task 5 – Backtesting**
- Compared the **Max Sharpe portfolio** to a **60/40 SPY/BND benchmark** over the last year.
- Metrics:
  - Cumulative return of optimized portfolio exceeded benchmark.
  - Higher Sharpe Ratio for optimized portfolio.
- Conclusion: Optimized portfolio offered better risk-adjusted returns in the test period.

---

## 📂 Project Structure
```
├── data/
│   ├── raw/             # Unprocessed CSVs from yfinance
│   ├── processed/       # Cleaned data
├── models/              # Saved model artifacts (.pkl, .keras)
├── notebooks/           # Jupyter notebooks (EDA, modeling, optimization, backtesting)
├── results/
│   ├── forecasts/       # Model forecasts
│   ├── plots/           # Visualization outputs
│   ├── optimization/    # Efficient frontier results
├── scripts/             # Standalone Python scripts for reproducibility
├── src/                 # Modular Python code
├── requirements.txt     # Dependencies (pinned)
├── README.md
└── LICENSE
```

---

## ⚙️ Setup Instructions
```bash
git clone https://github.com/yourusername/gmf-portfolio-forecasting.git
cd gmf-portfolio-forecasting
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage
1. **Fetch & Preprocess Data**
```bash
python scripts/fetch_data.py
python scripts/preprocess.py
```
2. **Run notebooks in order**:
```
1_data_preprocessing.ipynb
2_EDA.ipynb
3_diagnostics.ipynb
4_forecasting.ipynb
5_forecast_future.ipynb
6_portfolio_optimization.ipynb
7_backtesting.ipynb
```

---

## 📊 Results Summary
| Portfolio | Return | Volatility | Sharpe | TSLA | SPY | BND |
|-----------|--------|------------|--------|------|-----|-----|
| Max Sharpe | 9.10% | 0.69% | 11.77 | 0.01% | 57.03% | 42.96% |
| Min Volatility | 2.74% | 0.34% | 5.09 | 0.30% | 6.88% | 92.82% |

---

## 🏁 Conclusion
- **Task 1–3** built a strong forecasting foundation, showing LSTM’s superiority for TSLA.
- **Task 4** transformed forecasts into actionable portfolio allocations using MPT.
- **Task 5** confirmed through backtesting that the optimized portfolio outperformed a standard 60/40 benchmark in risk-adjusted returns.
- **Investor Takeaway**:
  - **Aggressive** → Max Sharpe Portfolio for higher returns with controlled risk.
  - **Conservative** → Min Volatility Portfolio for maximum stability.
- The process demonstrated how **data-driven forecasts + optimization** can guide portfolio strategy.

---

## 🔁 Reproducibility
- Fixed random seeds:  
  - NumPy: `np.random.seed(42)`  
  - TensorFlow: `tf.keras.utils.set_random_seed(42)`
- Pinned dependencies in `requirements.txt`
- All results saved in `results/`

---

## 🆘 Troubleshooting
- **ModuleNotFoundError: `src`** → Add:
```python
import sys, os
sys.path.append(os.path.abspath(".."))
```
- **`squared` argument error** →  
```python
rmse = mean_squared_error(y_true, y_pred) ** 0.5
```
- TensorFlow protobuf warnings are safe to ignore.

---

## 📜 License
MIT License – see [LICENSE](LICENSE) for details.
