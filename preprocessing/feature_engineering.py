import pandas as pd

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các chỉ số phái sinh chuẩn y sinh:
    - HOMA-IR: Chỉ số kháng insulin
    - Non-HDL: Cholesterol sinh xơ vữa
    - VLDL: Lipoprotein tỷ trọng rất thấp
    - LDL/HDL & TC/HDL: Tỷ số rủi ro tim mạch
    - WHtR & WHR: Tỷ lệ mỡ bụng
    """
    df = df.copy()
    
    # 1. Kháng Insulin
    df['HOMA_IR'] = (df['FBS'] * df['Insulin']) / 405.0
    
    # 2. Bộ mỡ máu phái sinh
    df['Non_HDL'] = df['Total_Cholesterol'] - df['HDL_C']
    df['VLDL'] = df['Triglycerides'] / 2.2
    df['LDL_HDL_Ratio'] = df['LDL_C'] / df['HDL_C']
    df['TC_HDL_Ratio'] = df['Total_Cholesterol'] / df['HDL_C']
    
    # 3. Chỉ số nhân trắc học
    df['Waist_to_Height'] = df['Waist_Circumference'] / df['Height']
    df['Waist_to_Hip'] = df['Waist_Circumference'] / df['Hip_Circumference']
    
    return df