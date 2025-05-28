# app/models/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def run_lstm_model(df, target_column, forecast_periods):
    look_back = 30
    y = df[target_column].dropna()
    if len(y) <= look_back:
        raise ValueError(f"Time series too short for LSTM. Need more than {look_back} data points.")
    data = df[[target_column]].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    last_sequence = data_scaled[-look_back:]
    predictions = []
    for _ in range(forecast_periods):
        pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)[0][0]
        predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_periods+1, freq='D')[1:]
    return pd.DataFrame({f"{target_column}_forecast": predictions.flatten()}, index=forecast_index)
