from river.drift import ADWIN
import pandas as pd
import json

def detect_drift():
    print("[+] Running ADWIN Drift Detection (Final Refinement)...")
    df = pd.read_csv("ML/preprocessed_data.csv")
    
    # 1. Initialize Detectors
    adwin_price = ADWIN()
    adwin_behavior = ADWIN()
    
    price_drift_points = []
    behavior_drift_points = []
    
    last_detected_price = -1000
    last_detected_behavior = -1000
    
    # 2. Iterate Stream
    for i, row in df.iterrows():
        # FINAL SAFETY: Skip first 1500 rows to ensure complete stability
        if i < 1500:
            continue
            
        # Stream Price (SMOOTHED)
        adwin_price.update(row["price_smooth"])
        if adwin_price.drift_detected:
            # Refractory logic: Only detect if 200 items have passed since last
            if (i - last_detected_price > 200):
                price_drift_points.append(int(i))
                last_detected_price = i
            
        # Stream Behavior (Smoothed CR)
        if pd.notnull(row["rolling_cr"]):
            adwin_behavior.update(row["rolling_cr"])
            if adwin_behavior.drift_detected:
                if (i - last_detected_behavior > 200):
                    behavior_drift_points.append(int(i))
                    last_detected_behavior = i

    results = {
        "price_drift_indices": price_drift_points,
        "behavior_drift_indices": behavior_drift_points,
        "combined_first_detection": min(price_drift_points + behavior_drift_points) if (price_drift_points + behavior_drift_points) else None
    }
    
    with open("ML/drift_detection_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"[+] Detection Complete. First detection: {results['combined_first_detection']}")
    print(f"    Feature Drift (Price) points: {price_drift_points}")
    print(f"    Behavioral Drift points: {behavior_drift_points}")

if __name__ == "__main__":
    detect_drift()
