import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_model():
    print("[+] Running XGBoost Model Evaluation (Temporal Validation)...")
    df = pd.read_csv("ML/preprocessed_data.csv")
    
    # 1. Feature Engineering
    features = ["price", "discounted_price", "time_of_day", "current_page", "device", "traffic"]
    data = df[features + ["action_flag", "is_drifted"]]
    data = pd.get_dummies(data, columns=["time_of_day", "current_page", "device", "traffic"])
    
    # 2. Sequential Splitting
    # Baseline (Clean) Data
    baseline_data = data[data["is_drifted"] == 0]
    drifted_data = data[data["is_drifted"] == 1]
    
    # Split Baseline into Train and Val (Temporally)
    split_idx = int(len(baseline_data) * 0.8)
    train_df = baseline_data.iloc[:split_idx]
    val_df = baseline_data.iloc[split_idx:]
    
    X_train = train_df.drop(columns=["action_flag", "is_drifted"])
    y_train = train_df["action_flag"]
    
    X_val = val_df.drop(columns=["action_flag", "is_drifted"])
    y_val = val_df["action_flag"]
    
    X_test_drift = drifted_data.drop(columns=["action_flag", "is_drifted"])
    y_test_drift = drifted_data["action_flag"]
    
    # 3. Training
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Rigorous Evaluation
    # Baseline Validation Accuracy (Realistic Performance)
    val_pred = model.predict(X_val)
    acc_val = accuracy_score(y_val, val_pred)
    
    # Drift Accuracy
    drift_pred = model.predict(X_test_drift)
    acc_drift = accuracy_score(y_test_drift, drift_pred)
    
    degradation = acc_val - acc_drift
    
    print(f"\n[XGBoost Performance Report]")
    print(f"  Validation (Baseline) Accuracy: {acc_val:.4f}")
    print(f"  Drifted Accuracy:              {acc_drift:.4f}")
    print(f"  Scientific Model Degradation:   {degradation:.4f}")
    
    with open("ML/model_degradation_report.txt", "w") as f:
        f.write("Model Evaluation Report (Temporal Validation)\n")
        f.write(f"Validation (Clean) Accuracy: {acc_val:.4f}\n")
        f.write(f"Drifted Accuracy:            {acc_drift:.4f}\n")
        f.write(f"Accuracy Collapse:           {degradation:.4f}\n")

if __name__ == "__main__":
    evaluate_model()
