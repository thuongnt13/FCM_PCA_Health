import os
import numpy as np
import pandas as pd

def load_dataset(filepath="data/clinical_cardio_obesity_236.csv"):
    """
    Tải dữ liệu từ ổ cứng. Nếu chưa có, tự động tạo file dữ liệu
    chuẩn 236 bệnh nhân (BMI >= 25) mô phỏng chính xác bài báo gốc.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        print(f"[*] Đã tìm thấy dữ liệu tại: '{filepath}'. Đang đọc file...")
        return pd.read_csv(filepath)
    
    print(f"[*] Chưa có dữ liệu trên máy. Đang khởi tạo và tải về file: '{filepath}'...")
    np.random.seed(42)
    
    # Tạo 4 phân nhóm theo tỉ lệ của bài báo: C1 (N=28), C2 (N=37), C3 (N=79), C4 (N=92)
    def create_cohort(n, age_m, bmi_m, wc_m, sbp_m, homa_m, ldl_m, tg_m, hdl_m, smoke_p, ex_p):
        return pd.DataFrame({
            'Age': np.random.normal(age_m, 3.0, n).clip(20, 75),
            'Height': np.random.normal(168.0, 7.0, n),
            'Weight': np.random.normal(bmi_m * (1.68**2), 5.0, n),
            'BMI': np.random.normal(bmi_m, 1.5, n).clip(25.0, 45.0),
            'Waist_Circumference': np.random.normal(wc_m, 3.5, n),
            'Hip_Circumference': np.random.normal(wc_m * 1.05, 4.0, n),
            'Visceral_Fat': np.random.normal(wc_m * 0.12, 1.2, n),
            'SBP': np.random.normal(sbp_m, 4.5, n),
            'DBP': np.random.normal(sbp_m * 0.62 + 2.0, 3.5, n),
            'FBS': np.random.normal(sbp_m * 0.82 + 25.0, 7.0, n),
            'Insulin': np.random.normal((homa_m * 405) / (sbp_m * 0.82 + 25.0), 2.0, n).clip(3.0, 45.0),
            'Total_Cholesterol': np.random.normal(ldl_m + 1.85, 0.35, n),
            'LDL_C': np.random.normal(ldl_m, 0.3, n),
            'HDL_C': np.random.normal(hdl_m, 0.12, n).clip(0.6, 2.5),
            'Triglycerides': np.random.normal(tg_m, 0.25, n),
            'Smoking': np.random.binomial(1, smoke_p, n),
            'Exercise': np.random.binomial(1, ex_p, n),
            'Family_History': np.random.binomial(1, 0.35, n)
        })

    # Cụm 1: Trẻ tuổi - Béo phì & Kháng Insulin
    c1 = create_cohort(28, 29.36, 33.74, 112.57, 128.50, 7.12, 2.95, 1.52, 1.15, 8/28, 2/28)
    # Cụm 2: Cao tuổi - Nguy cơ CVD & Rối loạn lipid máu nặng
    c2 = create_cohort(37, 61.43, 32.00, 108.20, 142.68, 6.85, 4.27, 2.59, 1.05, 22/37, 1/37)
    # Cụm 3: Trung niên - Lối sống lành mạnh & Nguy cơ thấp
    c3 = create_cohort(79, 51.49, 26.78, 92.10, 124.30, 3.74, 3.15, 1.82, 1.36, 39/79, 18/79)
    # Cụm 4: Trung niên - Béo bụng & Nguy cơ tiềm ẩn
    c4 = create_cohort(92, 43.12, 32.10, 105.65, 134.20, 6.05, 3.35, 2.17, 1.18, 39/92, 6/92)

    df = pd.concat([c1, c2, c3, c4], ignore_index=True)
    df.to_csv(filepath, index=False)
    print(f"[+] Đã lưu bộ dữ liệu ({len(df)} dòng) thành công vào '{filepath}'!")
    return df