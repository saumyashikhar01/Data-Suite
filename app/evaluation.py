from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
