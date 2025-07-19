import pandas as pd
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_shape(df, label='DataFrame'):
    logging.info(f"{label}: {df.shape[0]} rows x {df.shape[1]} columns")

def engineer_features(df: pd.DataFrame):
    if 'industry' in df.columns:
        df['is_tech'] = df['industry'].apply(lambda x: 1 if 'tech' in str(x).lower() else 0)
    if 'company_size' in df.columns:
        df['is_enterprise'] = df['company_size'].apply(lambda x: 1 if x >= 1000 else 0)
    if 'created_at' in df.columns:
        df['lead_month'] = pd.to_datetime(df['created_at'], errors='coerce').dt.month
    return df

def auc_score(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)

def remove_outliers(df, column, threshold=3):
    from scipy.stats import zscore
    z = np.abs(zscore(df[column].dropna()))
    return df[z < threshold]

if __name__ == '__main__':
    setup_logging()
    df = pd.DataFrame({
        'industry': ['Tech', 'Finance', 'Healthcare'],
        'company_size': [1200, 450, 800],
        'created_at': ['2024-01-15', '2024-02-01', '2024-01-19']
    })
    df = engineer_features(df)
    log_shape(df)
    print(df)
