# app/models/prophet_model.py

from prophet import Prophet
import pandas as pd

def run_prophet_model(df, target_column, forecast_periods):
    df_prophet = df.reset_index().rename(columns={df.index.name or 'index': 'ds', target_column: 'y'})
    if 'ds' not in df_prophet or 'y' not in df_prophet:
        raise ValueError('Dataframe must have columns "ds" and "y" with the dates and values respectively.')
    if len(df_prophet) < 20:
        raise ValueError("Time series too short for Prophet. Need at least 20 data points.")
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].set_index("ds").rename(columns={"yhat": f"{target_column}_forecast"})
