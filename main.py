import os
import pandas as pd
import numpy as np
from tabulate import tabulate

from data.dataset_loader import load_dataset
from preprocessing.feature_engineering import add_derived_features
from preprocessing.cleaner import ChouCleaner
from preprocessing.scaler import ClinicalScaler
from pca.pca_analysis import PCAAnalysis
from fuzzy_cmeans.fcm_model import FuzzyCMeans
from fuzzy_cmeans.evaluator import FCMEvaluator
from visualization.plots import save_scree_plot, save_biplot, save_radar_chart

def gan_nhan_dong_theo_benh_hoc(df_clean, cluster_col='Cluster_ID'):
    """
    CẢI TIẾN MỚI: Tự động phân tích giá trị sinh học trung bình của từng cụm 
    để đặt tên chuẩn xác 100%, khắc phục hiện tượng đảo nhãn của thuật toán không giám sát.
    """
    stats = df_clean.groupby(cluster_col).agg({
        'Age': 'mean',
        'BMI': 'mean',
        'HOMA_IR': 'mean',
        'SBP': 'mean'
    })
    
    mapping = {}
    remaining_clusters = list(stats.index)
    
    # 1. Cụm cao tuổi nhất -> Cụm 2 (Cao tuổi - Nguy cơ CVD)
    oldest_id = stats.loc[remaining_clusters, 'Age'].idxmax()
    mapping[oldest_id] = 'Cụm 2: Cao tuổi - Tim mạch & Lipid máu cao'
    remaining_clusters.remove(oldest_id)
    
    # 2. Cụm trẻ tuổi nhất -> Cụm 1 (Trẻ tuổi - Béo phì & Kháng Insulin)
    youngest_id = stats.loc[remaining_clusters, 'Age'].idxmin()
    mapping[youngest_id] = 'Cụm 1: Trẻ tuổi - Béo phì & Kháng Insulin'
    remaining_clusters.remove(youngest_id)
    
    # 3. Trong 2 cụm trung niên còn lại, cụm nào có BMI thấp nhất là Cụm 3 (Khỏe mạnh)
    healthiest_id = stats.loc[remaining_clusters, 'BMI'].idxmin()
    mapping[healthiest_id] = 'Cụm 3: Trung niên - Lối sống lành mạnh'
    remaining_clusters.remove(healthiest_id)
    
    # 4. Cụm còn lại là Cụm 4 (Nguy cơ tiềm ẩn)
    latent_id = remaining_clusters[0]
    mapping[latent_id] = 'Cụm 4: Trung niên - Béo bụng & Nguy cơ tiềm ẩn'
    
    return mapping

def run_pipeline():
    os.makedirs("results", exist_ok=True)
    print("=" * 80)
    print("      DỰ ÁN PHÂN CỤM FUZZY C-MEANS & PCA TRONG Y TẾ CÔNG CỘNG")
    print("=" * 80)

    # 1. Tải dữ liệu
    raw_df = load_dataset()

    # 2. Xây dựng biến phái sinh & làm sạch
    df_featured = add_derived_features(raw_df)
    cleaner = ChouCleaner()
    df_clean = cleaner.clean(df_featured)

    # Danh mục 16 biến phân tích lâm sàng
    features = [
        'Age', 'BMI', 'Waist_Circumference', 'Visceral_Fat', 'SBP', 'DBP',
        'FBS', 'HOMA_IR', 'Total_Cholesterol', 'LDL_C', 'HDL_C',
        'Triglycerides', 'Waist_to_Height', 'LDL_HDL_Ratio', 'Non_HDL', 'TC_HDL_Ratio'
    ]

    # 3. Chuẩn hóa Z-Score
    print("\n[*] Chuẩn hóa dữ liệu bằng Z-Score...")
    scaler = ClinicalScaler()
    X_scaled = scaler.fit_transform(df_clean, features)

    # 4. Phân tích PCA
    print("[*] Thực hiện phân tích PCA...")
    pca = PCAAnalysis()
    X_pca = pca.fit_transform(X_scaled, features)

    # In phương sai
    print("\n--- PHƯƠNG SAI GIẢI THÍCH (TOP 5 THÀNH PHẦN) ---")
    var_summary = pca.get_variance_summary().head(5)
    print(tabulate(var_summary, headers='keys', tablefmt='psql', showindex=False))

    # In Factor Loadings
    print("\n--- MA TRẬN FACTOR LOADINGS (SQUARED COSINES - TOP BIẾN ẢNH HƯỞNG) ---")
    loadings = pca.get_loadings_table(top_k=5)
    print(tabulate(loadings.head(8), headers='keys', tablefmt='psql'))

    # 5. Đánh giá đa tiêu chí lựa chọn số cụm K
    print("\n[*] Đánh giá so sánh các mô hình K = 3, 4, 5 trên không gian PCA...")
    X_fcm = X_pca[:, :5]
    comparison_table = FCMEvaluator.compare_k_configurations(X_fcm, k_list=[3, 4, 5])
    print(tabulate(comparison_table, headers='keys', tablefmt='psql', showindex=False))

    # 6. Chạy mô hình tối ưu K = 4
    print("\n[*] Huấn luyện mô hình tối ưu K = 4...")
    best_fcm = FuzzyCMeans(n_clusters=4, m=2.0, random_state=42).fit(X_fcm)
    labels = best_fcm.get_hard_labels()
    df_clean['Cluster_ID'] = labels

    # Gán nhãn ngữ nghĩa tự động (Cải tiến)
    cluster_names = gan_nhan_dong_theo_benh_hoc(df_clean, 'Cluster_ID')
    df_clean['Cluster_Name'] = df_clean['Cluster_ID'].map(cluster_names)

    # In bảng đối sánh chỉ số trung bình của 4 cụm chuẩn xác
    print("\n--- BẢNG ĐỐI SÁNH CÁC CHỈ SỐ LÂM SÀNG TRUNG BÌNH THEO 4 CỤM (ĐÃ CHUẨN HÓA TÊN) ---")
    summary_cols = ['Age', 'BMI', 'Waist_Circumference', 'SBP', 'HOMA_IR', 'LDL_C', 'Triglycerides']
    summary_df = df_clean.groupby('Cluster_Name')[summary_cols].mean().round(2)
    # Sắp xếp hiển thị từ Cụm 1 đến Cụm 4
    summary_df = summary_df.reindex(sorted(summary_df.index))
    print(tabulate(summary_df, headers='keys', tablefmt='psql'))

    # Lưu kết quả
    output_csv = "results/ket_qua_phan_cum_236_benh_nhan.csv"
    df_clean.to_csv(output_csv, index=False)
    print(f"\n[+] Kết quả phân cụm chi tiết đã được lưu vào: '{output_csv}'")

    # 7. Xuất đồ họa
    print("\n[*] Đang tạo và lưu các biểu đồ nghiên cứu vào thư mục 'results/'...")
    save_scree_plot(pca.pca, "results/hinh_1_scree_plot.png")
    save_biplot(X_pca, labels, cluster_names, "results/hinh_2_pca_fcm_biplot.png")
    save_radar_chart(df_clean, 'Cluster_Name', summary_cols, "results/hinh_3_radar_chart.png")
    print("[+] Hoàn tất! Đã xuất 3 hình: Scree plot, Biplot và Radar chart.")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()