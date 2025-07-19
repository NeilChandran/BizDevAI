import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from typing import Tuple

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Fill missing values and encode categorical data
    print("Preprocessing data for model training...")
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    y = df['deal_won'].astype(int)
    X = df.drop(['deal_won', 'email', 'created_at'], axis=1)
    X = pd.get_dummies(X)
    return X, y

def train_lead_scorer(X, y):
    print("Training the lead scoring model...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:,1]
    print(classification_report(y_val, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.3f}")
    return model

def score_leads(df: pd.DataFrame, model):
    print("Scoring leads...")
    X, _ = preprocess_data(df)
    return model.predict_proba(X)[:,1]

def main():
    data = pd.read_csv('data/all_leads_cleaned.csv')
    X, y = preprocess_data(data)
    model = train_lead_scorer(X, y)
    # Save model with joblib, pickle, etc. For brevity, not implemented here.

    # Score new leads
    test_data = pd.read_csv('data/new_leads.csv')
    scores = score_leads(test_data, model)
    test_data['lead_score'] = scores
    test_data.sort_values('lead_score', ascending=False, inplace=True)
    test_data.to_csv('data/scored_leads.csv', index=False)
    print("Lead scoring complete.")

if __name__ == '__main__':
    main()
