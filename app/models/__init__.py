# app/models/__init__.py

from .arima_model import run_arima_model
from .prophet_model import run_prophet_model
from .lstm_model import run_lstm_model

def run_forecasting_model(model_name, df, target_column, forecast_periods):
    if model_name == "ARIMA":
        return run_arima_model(df, target_column, forecast_periods)
    elif model_name == "Prophet":
        return run_prophet_model(df, target_column, forecast_periods)
    elif model_name == "LSTM":
        return run_lstm_model(df, target_column, forecast_periods)
    else:
        raise ValueError("Unsupported model")
