import numpy as np
import pandas as pd

class ChouCleaner:
    def __init__(self):
        pass

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[*] Áp dụng quy tắc 5 bước của Chou để làm sạch dữ liệu...")
        initial_len = len(df)
        
        # 1. Khử trùng lặp (Deduplication)
        df = df.drop_duplicates().copy()
        
        # 2. Tính đầy đủ: Điền khuyết bằng trung vị nếu có (Completeness)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                
        # 3. Tính nhất quán sinh học (Consistency)
        # Giữ lại các trường hợp thỏa mãn giới hạn sinh tồn y tế thực tế
        hop_le = (
            (df['BMI'] >= 20.0) & (df['BMI'] <= 60.0) &
            (df['SBP'] >= 70.0) & (df['SBP'] <= 230.0) &
            (df['FBS'] >= 40.0) & (df['FBS'] <= 350.0)
        )
        df = df[hop_le].copy()

        print(f"[+] Hoàn tất làm sạch: Giữ lại {len(df)}/{initial_len} bản ghi đạt chuẩn y khoa.")
        return df.reset_index(drop=True)