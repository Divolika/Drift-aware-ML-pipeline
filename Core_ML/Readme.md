This folder containes a **complete machine learning pipeline** to simulate, detect, and evaluate **concept drift** in e-commerce user behavior.

The system is designed to answer:

 *What happens to ML models when user behavior changes over time?*

It includes:

* Data preprocessing
* Drift detection (ADWIN)
* Behavioral modeling (Markov Chains)
* Model performance evaluation (XGBoost)

---

##  Pipeline Flow

```text
Raw Dataset
   ↓
Preprocessing
   ↓
Drift Detection (ADWIN)
   ↓
Behavior Analysis (Markov)
   ↓
Model Evaluation (XGBoost)
```

---

#  Detailed Explanation of Each File

---

##  1. preprocessing.py

###  Purpose:

Prepares raw dataset for analysis and modeling.

###  What it does:

* Loads dataset
* Sorts data by `session_id` and `step_index`
* Creates behavioral signal:

  * `action_flag` (1 if user adds to cart, else 0)
* Generates **rolling conversion rate** (smoothed behavior signal)
* Applies **price smoothing**
* Creates `next_page` column for transition analysis

###  Output:

* `preprocessed_data.csv`

---

##  2. drift_detection.py (ADWIN)

###  Purpose:

Detects **concept drift in streaming data**

###  What it does:

* Uses **ADWIN (Adaptive Windowing)**
* Monitors two signals:

  * `price_smooth` → feature drift
  * `rolling_cr` → behavioral drift
* Processes data row-by-row (stream simulation)
* Applies:

  * warm-up skip (first 1500 rows)
  * refractory logic (avoid repeated detections)

###  Output:

* `drift_detection_results.json`

---

##  3. behavior_analysis.py (Markov Chains)

### Purpose:

Analyzes **user behavior changes**

###  What it does:

* Splits dataset into:

  * Baseline (no drift)
  * Drifted data
* Builds **transition probability matrix**
* Focuses on:

  * `Product → Cart` transition

### Key Formula:

ΔP = |P_baseline − P_drifted|

###  Output:

* `behavior_deltas.txt`

---

##  4. model_evaluation.py (XGBoost)

###  Purpose:

Measures **ML model degradation under drift**

### What it does:

* Performs feature engineering (encoding categorical variables)
* Uses **temporal split**:

  * Train → baseline data
  * Validation → clean unseen data
  * Test → drifted data
* Trains XGBoost model
* Evaluates performance before and after drift

### Output:

* `model_degradation_report.txt`

---

## 5. run_pipeline.py

### Purpose:

Runs the entire system automatically

### What it does:

* Executes all scripts in order:

  1. preprocessing
  2. drift detection
  3. behavior analysis
  4. model evaluation
* Handles execution flow

---

#  Key Results

| Metric                | Value  |
| --------------------- | ------ |
| Behavioral Drift (ΔP) | 0.6326 |
| Validation Accuracy   | 91.11% |
| Drifted Accuracy      | 72.13% |
| Accuracy Drop         | ~19%   |

---

# How to Run

## 1. Install dependencies

```bash
pip install pandas numpy xgboost river scikit-learn
```

## 2. Run pipeline

```bash
python run_pipeline.py
```


