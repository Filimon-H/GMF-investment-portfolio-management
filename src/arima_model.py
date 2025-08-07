import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(train_series: pd.Series, seasonal: bool = False):
    """
    Automatically find ARIMA(p,d,q) parameters and train model.

    Returns:
        fitted model
    """
    # Auto ARIMA to choose best p, d, q
    model_auto = auto_arima(train_series,
                            seasonal=seasonal,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore')

    print("Best ARIMA order:", model_auto.order)

    # Train final ARIMA model with best params
    model = ARIMA(train_series, order=model_auto.order)
    model_fit = model.fit()
    return model_fit


def forecast_arima(model_fit, steps: int) -> pd.Series:
    """
    Forecast the next `steps` values using a trained ARIMA model.
    
    Returns:
        forecasted values as a pandas Series
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast
