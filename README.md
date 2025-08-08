# 📈 GMF Investment Portfolio Forecasting & Optimization

## 📄 Project Overview
Guide Me in Finance (GMF) Investments is a forward-thinking financial advisory firm specializing in personalized portfolio management.  
This project applies advanced **time series forecasting** and **Modern Portfolio Theory (MPT)** to optimize portfolio allocation across Tesla (TSLA), SPY, and BND, aiming to enhance returns while managing risk.

We use:
- **Historical market data** from Yahoo Finance (`yfinance`)
- **ARIMA** and **LSTM** forecasting models
- **Efficient Frontier optimization**
- **Backtesting** to validate the strategy

---

## 📚 Table of Contents
1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Setup Instructions](#-setup-instructions)
4. [Usage](#-usage)
5. [Reproducibility](#-reproducibility)
6. [Troubleshooting](#-troubleshooting)
7. [License](#-license)

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

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gmf-portfolio-forecasting.git
cd gmf-portfolio-forecasting
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Fetch and Preprocess Data
```bash
python scripts/fetch_data.py
python scripts/preprocess.py
```

### 2. Run the Notebooks in Order
```plaintext
notebooks/
  1_data_preprocessing.ipynb
  2_EDA.ipynb
  3,Diagnostics.ipynb
  4_forecasting.ipynb
  5_forecast_future.ipynb
  6_portfolio_optimization.ipynb
  7_backtesting.ipynb
```

### 3. Artifacts
- **Forecasts** → `results/forecasts/`
- **Plots** → `results/plots/`
- **Optimization results** → `results/optimization/`
- **Saved models** → `models/`

---

## 🔁 Reproducibility
- Random seeds fixed where applicable:
  - NumPy: `np.random.seed(42)`
  - TensorFlow/Keras: `tf.keras.utils.set_random_seed(42)`
- Dependencies pinned in `requirements.txt`
- Outputs stored deterministically in `results/` and `models/`

---

## 🆘 Troubleshooting
- **ModuleNotFoundError: `src`** → Add at the top of your notebook:
  ```python
  import sys, os
  sys.path.append(os.path.abspath(".."))
  ```
- **`squared` argument error in sklearn** → Compute RMSE as:
  ```python
  rmse = mean_squared_error(y_true, y_pred) ** 0.5
  ```
- **TensorFlow protobuf warnings** → Safe to ignore, training proceeds normally.

---

## 📜 License
MIT License – see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Statsmodels](https://www.statsmodels.org/)
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)

---
