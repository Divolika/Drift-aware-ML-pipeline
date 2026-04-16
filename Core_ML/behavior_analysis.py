import pandas as pd
import numpy as np

def analyze_behavior():
    print("[+] Running Markov Behavioral Analysis...")
    df = pd.read_csv("ML/preprocessed_data.csv")
    df = df.dropna(subset=["next_page"])
    
    # 1. Split Baseline vs Drifted
    baseline = df[df["is_drifted"] == 0]
    drifted = df[df["is_drifted"] == 1]
    
    def get_matrix(data):
        m = data.groupby(["current_page", "next_page"]).size().unstack(fill_value=0)
        return m.div(m.sum(axis=1), axis=0)

    m_base = get_matrix(baseline)
    m_drift = get_matrix(drifted)
    
    # 2. Calculate Delta P
    # Focused on Product -> Cart
    it_base = m_base.loc["product_details", "cart"] if "product_details" in m_base.index and "cart" in m_base.columns else 0
    it_drift = m_drift.loc["product_details", "cart"] if "product_details" in m_drift.index and "cart" in m_drift.columns else 0
    
    delta_p = abs(it_base - it_drift)
    
    print(f"\n[Markov Result: Product -> Cart]")
    print(f"  P(Baseline): {it_base:.4f}")
    print(f"  P(Drifted):  {it_drift:.4f}")
    print(f"  Delta P:     {delta_p:.4f}")
    
    with open("ML/behavior_deltas.txt", "w") as f:
        f.write(f"Markov Analysis Results\n")
        f.write(f"Product -> Cart Step\n")
        f.write(f"Baseline: {it_base:.4f}\n")
        f.write(f"Drifted:  {it_drift:.4f}\n")
        f.write(f"Delta P:  {delta_p:.4f}\n")

if __name__ == "__main__":
    analyze_behavior()
