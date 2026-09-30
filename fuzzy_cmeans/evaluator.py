import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd

class FCMEvaluator:
    @staticmethod
    def evaluate_model(fcm_model, X: np.ndarray) -> dict:
        """
        Tính 4 chỉ số đo lường hiệu năng của cụm mờ:
        - BCV: Phương sai giữa các cụm
        - PC: Hệ số phân hoạch (Partition Coefficient, -> 1 là tốt)
        - PE: Độ hỗn loạn phân hoạch (Partition Entropy, càng nhỏ càng tốt)
        - SS: Silhouette Score (độ tách biệt)
        """
        u = fcm_model.u
        centroids = fcm_model.centroids
        n, c = u.shape
        
        # 1. Partition Coefficient
        pc = np.sum(u ** 2) / n
        
        # 2. Partition Entropy
        pe = -np.sum(u * np.log(np.fmax(u, 1e-10))) / n
        
        # 3. Silhouette Score
        labels = np.argmax(u, axis=1)
        silhouette = silhouette_score(X, labels)
        
        # 4. Between-Cluster Variation (BCV)
        grand_mean = np.mean(X, axis=0)
        bcv = 0.0
        for i in range(c):
            cluster_size = np.sum(labels == i)
            bcv += cluster_size * np.linalg.norm(centroids[i] - grand_mean) ** 2
            
        return {
            'BCV': round(bcv, 2),
            'PC': round(pc, 3),
            'PE': round(-pe, 3),  # Biểu diễn dưới dạng giá trị âm như trong bài báo
            'Silhouette': round(silhouette, 3)
        }

    @staticmethod
    def compare_k_configurations(X: np.ndarray, k_list=[3, 4, 5], m=2.0) -> pd.DataFrame:
        records = []
        for k in k_list:
            from fuzzy_cmeans.fcm_model import FuzzyCMeans
            model = FuzzyCMeans(n_clusters=k, m=m, random_state=42).fit(X)
            metrics = FCMEvaluator.evaluate_model(model, X)
            records.append({
                'Cấu hình': f"Mô hình K={k}",
                'BCV (Phương sai giữa cụm)': metrics['BCV'],
                'PC (Hệ số phân hoạch)': metrics['PC'],
                'PE (Độ hỗn loạn)': metrics['PE'],
                'Silhouette Score': metrics['Silhouette']
            })
        return pd.DataFrame(records)