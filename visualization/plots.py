import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_scree_plot(pca_obj, output_path="results/scree_plot.png"):
    var_exp = pca_obj.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)
    
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    
    pcs = np.arange(1, len(var_exp) + 1)
    ax1.bar(pcs[:10], var_exp[:10], color='skyblue', alpha=0.8, label='Phương sai riêng (%)')
    ax2.plot(pcs[:10], cum_var[:10], color='firebrick', marker='o', linewidth=2, label='Tích lũy (%)')
    
    ax1.set_xlabel('Số lượng Thành phần chính (PCs)')
    ax1.set_ylabel('Phương sai riêng lẻ (%)', color='steelblue')
    ax2.set_ylabel('Phương sai tích lũy (%)', color='firebrick')
    plt.title('Scree Plot: Xác định điểm uốn (Elbow) và phương sai giải thích', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_biplot(X_pca, labels, class_names, output_path="results/pca_fcm_biplot.png"):
    plt.figure(figsize=(9, 6.5))
    palette = ['#e74c3c', '#8e44ad', '#27ae60', '#f39c12']
    
    df_plot = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': [class_names[i] for i in labels]})
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette=palette, s=70, alpha=0.9, edgecolor='k')
    
    plt.title('Biplot 2D: Phân tầng 4 Cụm nguy cơ trên không gian PCA', fontweight='bold')
    plt.xlabel('Thành phần chính 1 (PC1)')
    plt.ylabel('Thành phần chính 2 (PC2)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_radar_chart(df, label_col, features, output_path="results/radar_chart.png"):
    radar_raw = df.groupby(label_col)[features].mean()
    radar_norm = (radar_raw - radar_raw.min()) / (radar_raw.max() - radar_raw.min() + 1e-5)
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    # Tăng kích thước canvas và chỉnh vị trí layout
    fig, ax = plt.subplots(figsize=(8.5, 7.5), subplot_kw=dict(polar=True))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (c_name, row) in enumerate(radar_norm.iterrows()):
        vals = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, vals, linewidth=2.2, label=c_name, color=colors[idx % len(colors)])
        ax.fill(angles, vals, alpha=0.12, color=colors[idx % len(colors)])
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), features, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.05)
    
    # Đặt tiêu đề cao hơn một chút
    plt.title('Biểu đồ Radar chuẩn hóa các chỉ số sức khỏe chính', fontweight='bold', fontsize=13, y=1.12)
    
    # Đặt Legend nằm hẳn ra góc ngoài bên dưới, không đè lên hình tròn
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Thêm vào file: visualization/plots.py

def save_correlation_matrix(df, features, output_path="results/hinh_phu_correlation_matrix.png"):
    """Vẽ Ma trận tương quan giữa các biến lâm sàng (Tương ứng Hình 3 trong bài báo gốc)"""
    plt.figure(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=False, linewidths=0.5)
    plt.title('Ma trận hệ số tương quan giữa các biến số lâm sàng', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_cluster_bar_comparisons(df, output_path="results/hinh_phu_cluster_bars.png"):
    """Vẽ biểu đồ cột so sánh từng chỉ số theo 4 cụm (Tương ứng Hình 7 trong bài báo gốc)"""
    vars_to_plot = ['Age', 'BMI', 'FBS', 'SBP', 'LDL_C', 'HOMA_IR', 'Triglycerides']
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
    
    palette = ['#e74c3c', '#8e44ad', '#27ae60', '#f39c12']
    for idx, var in enumerate(vars_to_plot):
        sns.barplot(data=df, x='Cluster_Name', y=var, ax=axes[idx], palette=palette, ci=None)
        axes[idx].set_title(f'So sánh chỉ số: {var}', fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].tick_params(axis='x', rotation=15)
        
    axes[-1].axis('off') # Ẩn ô cuối cùng
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()