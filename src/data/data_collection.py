import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
def load_data(filepath: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: from {filepath}:{e}")
    
def load_params(filepath: str) -> float:
    try:
        with open(filepath,'r') as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Failed to load parameters: from {filepath}:{e}")
    
def split_data(data: pd.DataFrame, test_size: float):
    try:
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        return train_data, test_data
    except ValueError as e:
        raise ValueError(f"Failed to split data: {e}")
    
def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Failed to save data: {filepath}:{e}") 

def main():
    data_filepath = "water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join('data', 'raw')
    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)
        os.makedirs(raw_data_path, exist_ok=True)
        save_data(train_data, os.path.join(raw_data_path, 'train_data.csv'))
        save_data(test_data, os.path.join(raw_data_path, 'test_data.csv'))
    except Exception as e:
        raise Exception(f"Failed to execute main: {e}")

if __name__ == '__main__':
    main()