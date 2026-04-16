import pandas as pd
import numpy as np

def run_preprocessing():
    print("[+] Running Preprocessing...")
    
    # 1. Load Dataset
    df = pd.read_csv("ecommerce_behavior_dataset.csv")
    gt = pd.read_csv("row_level_ground_truth.csv")
    
    # Merge for easier label-aware processing (Internal use only)
    full = pd.concat([df, gt[["is_drifted"]]], axis=1)
    
    # 2. Sort by sequence
    full = full.sort_values(["session_id", "step_index"])
    
    # 3. Create Behavioral Signals
    full["action_flag"] = (full["action"] == "add_to_cart").astype(int)
    
    # Rolling Conversion Rate (Smoothing for ADWIN)
    full["rolling_cr"] = full["action_flag"].rolling(window=100, min_periods=100).mean()
    
    # Feature Smoothing & Normalization (Drift Clarity)
    full["price_smooth"] = full["price"].rolling(window=100, min_periods=100).mean()
    full["price_norm"] = (full["price"] - full["price"].mean()) / full["price"].std()
    
    # 4. Create Transition Pairs (for Markov)
    full["next_page"] = full.groupby("session_id")["current_page"].shift(-1)
    
    # 5. Handle Categoricals for XGBoost
    # We will do this later in model_evaluation for flexibility, or save a preprocessed version
    
    full.to_csv("ML/preprocessed_data.csv", index=False)
    print(f"[+] Saved ML/preprocessed_data.csv ({len(full)} rows)")

if __name__ == "__main__":
    run_preprocessing()
