import pandas as pd
import numpy as np

def preprocess_data(df, date_column, forecast_column, missing_option="drop", freq_option="D"):
    try:
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # 1. Convert date column to datetime with better error handling
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error converting date column: {str(e)}")

        # 2. Drop rows with invalid dates
        invalid_dates = df[date_column].isna()
        if invalid_dates.any():
            print(f"Warning: Dropping {invalid_dates.sum()} rows with invalid dates")
            df = df.dropna(subset=[date_column])

        # 3. Ensure forecast column is numeric
        try:
            df[forecast_column] = pd.to_numeric(df[forecast_column], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error converting forecast column to numeric: {str(e)}")

        # 4. Set the datetime column as index
        df = df.set_index(date_column)

        # 5. Sort index to ensure chronological order
        df = df.sort_index()

        # 6. Keep only the forecast column (and drop others)
        df = df[[forecast_column]]

        # 7. Handle missing values
        if missing_option == "drop":
            df = df.dropna()
        elif missing_option == "ffill":
            df = df.fillna(method="ffill")
        elif missing_option == "bfill":
            df = df.fillna(method="bfill")
        else:
            raise ValueError(f"Invalid missing_option: {missing_option}")

        # 8. Resample to desired frequency (Daily, Weekly, Monthly)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex for resampling")
        
        # Map frequency options to pandas frequency strings
        freq_map = {
            "D": "D",  # Daily
            "W": "W",  # Weekly
            "M": "M"   # Monthly
        }
        
        if freq_option not in freq_map:
            raise ValueError(f"Invalid frequency option: {freq_option}")
            
        df = df.resample(freq_map[freq_option]).mean()

        # 9. Final cleanup (just in case resampling added NaNs)
        df = df.dropna()

        return df

    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")
