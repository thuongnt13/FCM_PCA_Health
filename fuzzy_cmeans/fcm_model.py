import numpy as np

class FuzzyCMeans:
    """
    Thuật toán phân cụm mờ Fuzzy C-Means (Bezdek, 1984).
    """
    def __init__(self, n_clusters=4, m=2.0, max_iter=200, tol=1e-5, random_state=42):
        self.c = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.u = None
        self.centroids = None

    def fit(self, X: np.ndarray):
        np.random.seed(self.random_state)
        n_samples, _ = X.shape
        
        # Khởi tạo ma trận độ thuộc ngẫu nhiên Dirichlet (tổng dòng = 1)
        u = np.random.dirichlet(np.ones(self.c), size=n_samples)
        
        for _ in range(self.max_iter):
            u_old = u.copy()
            
            # 1. Cập nhật tâm cụm v_i
            um = u ** self.m
            centroids = (um.T @ X) / (um.sum(axis=0)[:, None] + 1e-10)
            
            # 2. Tính ma trận khoảng cách Euclid d_ij
            dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            dist = np.fmax(dist, 1e-10)
            
            # 3. Cập nhật ma trận độ thuộc u_ij
            power = 2.0 / (self.m - 1.0)
            inv_dist = 1.0 / dist
            u = (inv_dist ** power) / (np.sum(inv_dist ** power, axis=1, keepdims=True))
            
            # Điều kiện dừng
            if np.max(np.abs(u - u_old)) < self.tol:
                break
                
        self.u = u
        self.centroids = centroids
        return self

    def get_hard_labels(self) -> np.ndarray:
        return np.argmax(self.u, axis=1)