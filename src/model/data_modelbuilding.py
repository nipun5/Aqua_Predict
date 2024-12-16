import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
            return params["data_modelbuilding"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Failed to load parameters {params_path}:{e}")
    
def load_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def prepare_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to prepare data: {e}")
    
def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Failed to save model {filepath}: {e}")

def main():
    try:
        params_path = 'params.yaml'
        data_preprocessed_path = './data/preprocessed/train_data_preprocessed.csv'
        model_name = 'models/model.pkl'
        n_estimators = load_params(params_path)
        train_data = load_data(data_preprocessed_path)
        X_train, y_train = prepare_data(train_data)
        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f"Failed to execute main: {e}")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise Exception(f"Failed to execute main: {e}")
