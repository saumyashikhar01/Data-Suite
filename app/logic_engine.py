from app.models import run_forecasting_model

def forecasting_logic(model_name, data, forecast_column, periods):
    # Call the right model function
    forecast = run_forecasting_model(model_name, data, forecast_column, periods)
    return forecast
