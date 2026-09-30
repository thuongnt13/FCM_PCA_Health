# ỨNG DỤNG PHÂN CỤM FUZZY C-MEANS VÀ PCA TRONG Y TẾ CÔNG CỘNG

## 1. Giới thiệu

Dự án triển khai mô hình học máy tích hợp giữa **Phân tích Thành phần Chính (PCA)** và **Phân cụm C-Means Mờ (FCM)** nhằm phân tầng nguy cơ bệnh tim mạch và béo phì dựa trên bộ dữ liệu lâm sàng 236 bệnh nhân.

### 2 : cấu trúc thư mục

FCM_PCA_Health/
│
├── data/
│ ├── dataset_loader.py # Tải hoặc tự động khởi tạo dữ liệu 236 bệnh nhân
│ └── clinical_cardio_obesity_236.csv # File dữ liệu y sinh chuẩn
│
├── preprocessing/
│ ├── cleaner.py # Tiền xử lý theo Quy tắc Chou cải tiến
│ ├── feature_engineering.py # Tính toán tự động HOMA-IR, Non-HDL, LDL/HDL...
│ └── scaler.py # Chuẩn hóa Z-Score
│
├── pca/
│ └── pca_analysis.py # Phân tích PCA, tính Phương sai & Factor Loadings
│
├── fuzzy_cmeans/
│ ├── fcm_model.py # Giải thuật Fuzzy C-Means hướng đối tượng chuẩn Bezdek
│ └── evaluator.py # Đánh giá đa tiêu chí: BCV, PC, PE, Silhouette Score
│
├── visualization/
│ └── plots.py # Xuất đồ họa: Scree Plot, Biplot 2D, Radar Chart
│
├── results/ # Thư mục chứa biểu đồ xuất ra và file CSV kết quả
│ ├── hinh_1_scree_plot.png
│ ├── hinh_2_pca_fcm_biplot.png
│ ├── hinh_3_radar_chart.png
│ └── ket_qua_phan_cum_236_benh_nhan.csv
│
├── main.py # File điều phối toàn bộ luồng thực thi
├── requirements.txt # Danh sách thư viện phụ thuộc
├── .gitignore # Bỏ qua các file rác khi đẩy lên Git
└── README.md # Hướng dẫn dự án

## 3. Hướng dẫn cài đặt và chạy trên máy tính

### Bước 1: Cài đặt thư viện

Mở Terminal/CMD tại thư mục dự án và chạy:

```bash
python -m pip install -r requirements.txt
```

### Bước 2: chạy chương trình

```bash
python main.py
```

### Bước 3: kiểm tra kết quả

```bash
Sau khi chạy xong, kết quả sẽ tự động được lưu vào thư mục results/:
results/hinh_1_scree_plot.png: Biểu đồ Scree Plot thể hiện phương sai giải thích của PCA.
results/hinh_2_pca_fcm_biplot.png: Biplot 2D phân tầng 4 cụm trên không gian PC1 - PC2.
results/hinh_3_radar_chart.png: Biểu đồ Radar đối sánh các chỉ số sức khỏe chính.
results/ket_qua_phan_cum_236_benh_nhan.csv: Bảng kết quả phân cụm chi tiết cho 236 bệnh nhân.
```
