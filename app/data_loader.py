import pandas as pd

def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Placeholder for future API loader
def load_api_data(api_url):
    # Implement API call and return dataframe
    raise NotImplementedError("API data loading not implemented yet")
