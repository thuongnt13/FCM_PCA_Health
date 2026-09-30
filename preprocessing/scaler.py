from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class ClinicalScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = None

    def fit_transform(self, df: pd.DataFrame, feature_columns: list) -> np.ndarray:
        self.features = feature_columns
        return self.scaler.fit_transform(df[feature_columns].values)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[self.features].values)