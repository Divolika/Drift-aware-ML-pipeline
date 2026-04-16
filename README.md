# Drift-Aware E-commerce Behavior Simulation & ML Evaluation Framework

## Overview

This project presents a **research-grade machine learning pipeline** designed to simulate, detect, and evaluate **concept drift** in dynamic e-commerce environments.

It combines:

* Synthetic behavioral data generation
* Drift injection (feature + behavioral + coupled)
* Real-time drift detection (ADWIN)
* Behavioral modeling (Markov Chains)
* ML robustness evaluation (XGBoost)

---

## Problem Statement

Machine learning models assume static data distributions.

In real-world systems:

* User behavior changes
* Prices fluctuate
* Conversion patterns evolve

This leads to **concept drift**, causing model failure.

---

## Objective

To build an **end-to-end framework** that:

* Simulates realistic drift scenarios
* Detects drift using streaming algorithms
* Quantifies behavioral changes
* Measures ML performance degradation

---

## System Architecture

Pipeline:

1. Dataset Generator
2. Drift Injection Engine
3. Preprocessing
4. Drift Detection (ADWIN)
5. Behavioral Analysis (Markov Chains)
6. ML Evaluation (XGBoost)

---

## 📊 Key Results

| Metric                | Value               |
| --------------------- | ------------------- |
| Behavioral Drift (ΔP) | 0.6326              |
| Validation Accuracy   | 91.11%              |
| Drifted Accuracy      | 72.13%              |
| Accuracy Drop         | ~19%                |
| Drift Detection       | ~±200 rows accuracy |

---

## Methodology

### 1. Dataset Generation

* Session-based user simulation
* Realistic navigation:
  `Home → Category → Product → Cart → Checkout`

### 2. Drift Injection

* Feature Drift (price shift)
* Behavioral Drift (transition change)
* Coupled Drift (price affects conversion)
* Temporal Drift (time-based behavior)

### 3. Drift Detection

* ADWIN algorithm
* Dual-stream monitoring:

  * Feature (price)
  * Behavior (conversion rate)

### 4. Behavioral Modeling

* Markov Chains
* Transition probability comparison

### 5. ML Evaluation

* XGBoost classifier
* Temporal validation (no leakage)
* Performance degradation analysis

---

## Project Structure

```
ML/                → ML pipeline
app/               → data generator
data/              → dataset
results/           → outputs
report/            → IEEE paper
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Pipeline

```bash
python ML/run_pipeline.py
```

---

## Outputs

* Drift detection results
* Behavioral transition changes
* Model performance degradation

---

## Technologies Used

* Python
* XGBoost
* River (ADWIN)
* Pandas / NumPy
* Flask (for dataset generation)

---

## Research Contribution

* Controlled drift simulation framework
* Multi-signal drift detection
* Behavioral drift quantification
* ML robustness evaluation

---

## Future Work

* Real-time streaming integration
* Adaptive model retraining
* Deep learning sequence models

---

## Author

**Divolika Bajpai**

---

## If you find this useful

Give a ⭐ to support the project!
