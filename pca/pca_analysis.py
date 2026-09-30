import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class PCAAnalysis:
    def __init__(self, n_components=None):
        self.pca = PCA(n_components=n_components)
        self.feature_names = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, X_scaled: np.ndarray, feature_names: list) -> np.ndarray:
        self.feature_names = feature_names
        X_pca = self.pca.fit_transform(X_scaled)
        self.components_ = self.pca.components_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return X_pca

    def get_loadings_table(self, top_k=5) -> pd.DataFrame:
        """
        Tính bảng Squared Cosines (Factor Loadings) tương đương Bảng 2 trong bài báo.
        """
        loadings = self.components_[:top_k, :].T ** 2
        col_names = [f"F{i+1}" for i in range(top_k)]
        df_loadings = pd.DataFrame(loadings, index=self.feature_names, columns=col_names)
        df_loadings['Max_Factor'] = df_loadings.max(axis=1)
        return df_loadings.sort_values(by='Max_Factor', ascending=False).round(3)

    def get_variance_summary(self) -> pd.DataFrame:
        var_exp = self.explained_variance_ratio_ * 100
        cum_var = np.cumsum(var_exp)
        return pd.DataFrame({
            'Thành phần': [f"PC{i+1}" for i in range(len(var_exp))],
            'Phương sai riêng lẻ (%)': np.round(var_exp, 2),
            'Phương sai tích lũy (%)': np.round(cum_var, 2)
        })