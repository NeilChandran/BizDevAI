import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

def load_customer_data(file_path: str) -> pd.DataFrame:
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df

def preprocess(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    print("Preprocessing customer features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    return X

def optimal_k(X, max_k=10) -> int:
    wcss = []
    print("Finding optimal number of clusters...")
    for k in range(1, max_k+1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        wcss.append(model.inertia_)
    # Simple elbow method - in production, more robust methods recommended
    deltas = [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
    optimal = deltas.index(max(deltas)) + 2  # +2 due to diff and 0-index
    print(f"Optimal k (elbow): {optimal}")
    return optimal

def segment_customers(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, KMeans]:
    X = preprocess(df, features)
    k = optimal_k(X)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['segment'] = clusters
    print(f"Assigned {k} clusters to customers.")
    return df, kmeans

def summarize_segments(df: pd.DataFrame):
    print("\nSegments summary:")
    seg_summary = df.groupby('segment').agg({
        'deal_size': 'mean',
        'industry': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
        'company_size': 'mean'
    })
    print(seg_summary)

def main():
    file_path = 'data/all_leads_cleaned.csv'
    features = ['company_size', 'deal_size']
    df = load_customer_data(file_path)
    segmented_df, model = segment_customers(df, features)
    segmented_df.to_csv('data/customer_segments.csv', index=False)
    summarize_segments(segmented_df)
    print("\nSegmentation complete.")

if __name__ == '__main__':
    main()
