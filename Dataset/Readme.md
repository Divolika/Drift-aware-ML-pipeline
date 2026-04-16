#  E-commerce Behavior Dataset (Drift-Aware)

##  Overview

This folder contains the **synthetic dataset and ground truth labels** used in the Drift-Aware ML Pipeline.

The dataset is designed to simulate **real-world e-commerce user behavior** with **controlled concept drift**, enabling:

* Drift detection experiments
* Behavioral analysis
* Machine learning robustness evaluation

---

##  Files Description

```bash
.
├── ecommerce_behavior_dataset.csv
├── preprocessed_data.csv
├── row_level_ground_truth.csv
```

---

#  1. ecommerce_behavior_dataset.csv

###  Purpose:

This is the **raw synthetic dataset** representing user interactions.

---

##  What it contains:

Each row represents **one step in a user session**.

###  Key Features:

| Category   | Features                                   |
| ---------- | ------------------------------------------ |
| User       | session_id, device, region, traffic        |
| Time       | timestamp, time_of_day                     |
| Behavior   | current_page, action                       |
| Product    | price, discounted_price, color, model_type |
| Navigation | step_index                                 |

---

##  Behavior Flow

Typical session follows:

```
Home → Category → Product → Cart → Checkout
```

---

##  Important Notes:

* Data is **sequential** (session-based)
* Contains **realistic navigation patterns**
* Used as input for preprocessing and ML pipeline

---

#  2. row_level_ground_truth.csv

###  Purpose:

Contains **hidden labels for drift evaluation**

---

## 📊 What it contains:

| Column     | Description               |
| ---------- | ------------------------- |
| is_drifted | 0 = baseline, 1 = drifted |

---

##  Why this is important:

* Used to **validate drift detection accuracy**
* NOT used directly in model training (to avoid leakage)

---

##  Usage:

* Helps compare:

  * actual drift vs detected drift
* Enables **scientific evaluation**

---

#  3. preprocessed_data.csv

###  Purpose:

Processed dataset used by ML pipeline

---

##  What preprocessing adds:

| Feature      | Description              |
| ------------ | ------------------------ |
| action_flag  | 1 if user adds to cart   |
| rolling_cr   | Smoothed conversion rate |
| price_smooth | Smoothed price signal    |
| next_page    | Next step in navigation  |
| is_drifted   | Ground truth label       |

---

##  Why this is needed:

* Converts raw data into **model-ready format**
* Reduces noise for drift detection
* Enables:

  * ADWIN
  * Markov modeling
  * XGBoost training

---

# 📊 Dataset Characteristics

| Property      | Value                          |
| ------------- | ------------------------------ |
| Type          | Synthetic                      |
| Structure     | Session-based                  |
| Rows          | 10,000+                        |
| Drift Types   | Feature + Behavioral + Coupled |
| Drift Pattern | Controlled & reproducible      |

---

#  Drift Types in Dataset

---

## 1. Feature Drift

* Price distribution shifts:

```
Low-price → High-price
```

---

## 2. Behavioral Drift

* Transition probability changes:

```
Product → Cart decreases sharply
```

---

## 3. Coupled Drift

* High price reduces conversion probability

---

## 4. Temporal Drift

* Behavior changes based on time of day

---

#  How This Dataset Is Used

Pipeline:

```
Raw Dataset
   ↓
Preprocessing
   ↓
Drift Detection
   ↓
Behavior Analysis
   ↓
Model Evaluation
```

---

#  Important Guidelines

* Do NOT use `is_drifted` for training models
* Always preserve session order
* Use rolling signals for stability

---

#  Key Insights Enabled

This dataset allows you to:

* Detect concept drift
* Analyze user behavior changes
* Measure ML model degradation
* Study real-world dynamic systems

---

# 🚀 Future Improvements

* Add real-world data integration
* Increase dataset size
* Add more user behavior types
* Introduce adversarial drift


