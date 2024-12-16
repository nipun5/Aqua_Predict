import pandas as pd
import numpy as np
import os

def load_data(directory: str, filename: str) -> pd.DataFrame:
    filepath = os.path.join(directory, filename)
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: from {filepath}:{e}")

def f_m_w_m(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                m_v = df[column].median()
                df[column].fillna(m_v, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Failed to fill missing values: {e}")

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Failed to save data: {e}")

def main():
    raw_data_path = './data/raw'
    train_data = load_data(raw_data_path, 'train_data.csv')
    test_data = load_data(raw_data_path, 'test_data.csv')
    try:
        train_p_d = f_m_w_m(train_data)
        test_p_d = f_m_w_m(test_data)

        data_preprocessed_path = os.path.join('data', 'preprocessed')
        os.makedirs(data_preprocessed_path, exist_ok=True)

        save_data(train_p_d, os.path.join(data_preprocessed_path, 'train_data_preprocessed.csv'))
        save_data(test_p_d, os.path.join(data_preprocessed_path, 'test_data_preprocessed.csv'))
    except Exception as e:
        raise Exception(f"Failed to execute main: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise Exception(f"Failed to execute main: {e}")