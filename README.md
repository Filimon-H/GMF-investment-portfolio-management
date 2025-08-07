
# 📈 Portfolio Forecasting and Model Comparison for GMF Investments

A time series forecasting project designed for **Guide Me in Finance (GMF) Investments**, a forward-thinking financial advisory firm that leverages cutting-edge technology to optimize personalized portfolios. This project focuses on forecasting future market trends using ARIMA and LSTM models and supports portfolio optimization through enhanced insights into asset behaviors.

---

## 📚 Table of Contents

- [Business Context](#business-context)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing Guidelines](#contributing-guidelines)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact Information](#contact-information)

---

## 💼 Business Context

**Guide Me in Finance (GMF) Investments** specializes in tailored portfolio management powered by real-time financial data and predictive analytics. The company uses advanced time series forecasting models to:
- Predict trends in high-growth stocks (e.g., TSLA), diversified ETFs (e.g., SPY), and stable assets (e.g., BND)
- Identify volatility and momentum patterns
- Optimize asset allocation based on market movement forecasts
- Make data-driven investment decisions that align with clients' risk and return profiles

This project simulates the work of a GMF Financial Analyst using historical data and forecasting models to enhance portfolio strategy.

---

## ✨ Features

- Download historical price data using YFinance (TSLA, SPY, BND)
- Clean and preprocess time series data
- Perform exploratory data analysis (EDA) on price trends and volatility
- Analyze risk with stationarity testing, Value at Risk (VaR), and Sharpe Ratio
- Build and compare ARIMA and LSTM models on financial assets
- Forecast future market movements (6–12 months)
- Save forecasts and models for reproducibility
- Plot trends and pseudo-confidence intervals

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/gmf-portfolio-forecasting.git
cd gmf-portfolio-forecasting
pip install -r requirements.txt
```

---

## 🚀 Usage

Run step-by-step forecasting and portfolio optimization notebooks:

```bash
jupyter notebook notebooks/task1_data_preprocessing.ipynb
```

Notebooks:
- `data_preprocessing.ipynb` – Clean and structure raw YFinance data
- `EDA.ipynb` – Visual and statistical exploration of asset behavior
- `forecasting.ipynb` – Build ARIMA and LSTM models
- `forecast_future.ipynb` – Long-term forecasts with interpretation

Outputs:
- `data/processed/`: Clean datasets
- `models/saved/`: Trained ARIMA and LSTM models
- `results/forecasts/`: Prediction results
- `results/plots/`: Trend and volatility visualizations

---

## 🤝 Contributing Guidelines

We welcome contributions:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'feat: add something'`
4. Push: `git push origin feature/my-feature`
5. Submit a Pull Request

Refer to `CONTRIBUTING.md` if available.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

## 🙌 Acknowledgments

- 10 Academy – for framing the business challenge
- Yahoo Finance (YFinance) – for real-time financial data
- TensorFlow, Pmdarima, Statsmodels – for time series modeling tools

---

## 📬 Contact Information

Created by **Filimon Hailemariam**  
Challenge: **10 Academy – Week 11**  
Domain: **Time Series Forecasting & Portfolio Optimization for GMF Investments**
