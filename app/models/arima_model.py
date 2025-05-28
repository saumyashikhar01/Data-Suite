# app/models/arima_model.py

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def run_arima_model(df, target_column, forecast_periods):
    y = df[target_column].dropna()
    if len(y) < 20:
        raise ValueError("Time series too short for ARIMA. Need at least 20 data points.")
    model = ARIMA(y, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_periods+1, freq='D')[1:]
    return pd.DataFrame({f"{target_column}_forecast": forecast}, index=forecast_index)
