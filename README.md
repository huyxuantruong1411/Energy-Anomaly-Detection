# Global XGBoost Framework for Large-Scale Energy Load Forecasting & Anomaly Detection

> **BÁO CÁO KỸ THUẬT / TECHNICAL REPORT (Markdown Version)**
> This README.md is a **full, faithful, and non-reductive Markdown transformation** of the original technical report.
> **No content, assumptions, equations, or experimental details from the source document have been removed.**
> Additional explanations, clarifications, and structural enrichments are provided **only to improve readability, reproducibility, and GitHub documentation quality**.

---

##  Project Overview

This repository presents a **global, end-to-end machine learning framework** for:

* 📈 **Electricity load forecasting**
* 🚨 **Unsupervised anomaly detection**

at **large scale** (1,600+ buildings), using **XGBoost accelerated by GPU**, robust statistics, and memory-efficient data pipelines.

The framework was designed and validated on the **Building Data Genome Project 2** dataset and demonstrates that **a single global model can reliably replace thousands of local models** while maintaining high predictive accuracy and anomaly-detection robustness.

---

##  Author & Metadata

* **Author**: Trương Xuân Huy
* **Date**: 29/01/2026
* **Referencer**: TS. Dương Thị Kim Chi
---

## Abstract

This study proposes a comprehensive solution for large-scale electricity consumption forecasting and anomaly detection across a network of more than **1,600 buildings** (Building Data Genome Project 2).

Facing challenges related to **load-scale heterogeneity** and **limited computational resources**, we develop a **single Global Model** based on **Extreme Gradient Boosting (XGBoost)** with **GPU acceleration**.

The system integrates:

* Chunk-based batch processing
* Dynamic memory compression
* Automatic hyperparameter optimization via **Optuna**

For anomaly detection in an **unlabeled environment**, we introduce a **Contextual Adaptive Thresholding mechanism** grounded in **robust statistics (Median & IQR)**.

### Key Results (200-building simulation):

* **Forecasting RMSE**: ~14.7
* **Anomaly Detection F1-score**: **0.957**

These results demonstrate strong generalization, scalability, and operational robustness.

---

## 1️⃣ Introduction

### 1.1 Background & Challenges

With the rise of **smart grids**, data from **Advanced Metering Infrastructure (AMI)** enables fine-grained energy optimization. However, large-scale deployment introduces two fundamental challenges:

#### 🔹 Scalability

Traditional approaches train a **separate local model per building**.

* For **N = 1,600 buildings**, this implies managing **1,600 independent models**
* High computational cost
* Complex deployment and maintenance

> This approach is impractical for real-world, city-scale energy systems.

#### 🔹 Heterogeneity

Buildings vary widely in:

* Load magnitude (kW → MW)
* Consumption patterns
* Temporal behavior

This heterogeneity severely limits naive global training approaches.

---

### 1.2 Research Contributions

This work proposes a **shift from local to global modeling**, where **a single model learns shared consumption dynamics across all buildings**.

**Core contributions**:

1. Memory-optimized **big-data processing pipeline**
2. GPU-accelerated **XGBoost regression** (`reg:squarederror`)
3. **Residual-based anomaly detection** with adaptive IQR thresholds
4. Robust evaluation via **synthetic fault injection**

---

## 2️⃣ Methodology

### 2.1 Data Pipeline Optimization & Memory Management

#### Dataset Characteristics

* **1,636 buildings**
* **Hourly resolution**
* Long-term time-series

Running this workload on **Google Colab (RAM-limited)** required careful memory engineering.

---

### a) Chunk Processing

Instead of loading the full dataset in **wide format**, data is processed in **chunks of buildings**:

* Example: 50 buildings per chunk
* Each chunk is:

  * Converted to **long format**
  * Feature-engineered
  * Concatenated incrementally

This avoids catastrophic memory spikes.

---

### b) Dynamic Downcasting

A custom `reduce_mem_usage` routine inspects each numeric column:

| Original Type | Optimized Type |
| ------------- | -------------- |
| float64       | float32        |
| int64         | int16 / int8   |

➡️ Achieves **60–70% memory reduction** with **no information loss**.

**Formal rule**:

* Use smallest numeric type satisfying:

```
min(x) ≥ lower_bound AND max(x) ≤ upper_bound
```

---

## 2.2 Feature Engineering

XGBoost does **not** inherently model temporal order → time awareness must be injected manually.

---

### a) Cyclical Time Encoding

To preserve temporal periodicity:

```
x_sin = sin(2πt / T)
x_cos = cos(2πt / T)
```

Where:

* `T = 24` (hour of day)
* `T = 7` (day of week)

This ensures **23:00 is close to 00:00** in feature space.

---

### b) Lag Features

For each building `b`:

```
L_k(t, b) = y(t−k, b),  k ∈ {1, 24, 168}
```

Captures:

* Short-term inertia
* Daily seasonality
* Weekly seasonality

---

### c) Velocity / Derivative Feature

To detect abrupt load changes:

```
Δy_t = y_t − y_{t−1}
```

This is critical for anomaly sensitivity.

---

### d) Log Transformation

Because load scales vary dramatically:

```
y' = ln(1 + y)
```

This stabilizes variance and converts the task into **relative-error space**, enabling effective global learning.

---

## 2.3 Forecasting Model: XGBoost

XGBoost (Gradient Boosted Decision Trees) is selected for:

* High performance on tabular data
* Robust non-linear modeling
* GPU acceleration support

### Objective Function

```
L^(t) = Σ l(y_i, ŷ_i^(t−1) + f_t(x_i)) + Ω(f_t)
```

Where:

* `l`: squared error loss
* `Ω`: regularization (L1 + L2)

### Hardware Configuration

```
tree_method = 'hist'
device = 'cuda'
```

➡️ Training speedup: **10–20× vs CPU** (Tesla T4)

---

## 2.4 Hyperparameter Optimization (Optuna)

Optuna with **TPE sampler** replaces brute-force grid search.

### Search Space

| Parameter        | Range            |
| ---------------- | ---------------- |
| learning_rate    | 0.01 – 0.2 (log) |
| max_depth        | 6 – 12           |
| subsample        | 0.6 – 0.9        |
| colsample_bytree | 0.6 – 0.9        |
| n_estimators     | 500 – 1500       |

Early stopping is applied to prevent overfitting.

---

## 2.5 Anomaly Detection Framework

### Step 1: Residual Computation

```
e(t, b) = |y_real(t, b) − y_pred(t, b)|
```

---

### Step 2: Grouped Robust Statistics

For each building `b`:

* Median(e)
* IQR(e) = Q3 − Q1

Median/IQR are preferred over mean/std due to **outlier resistance**.

---

### Step 3: Contextual Threshold

```
Threshold_b = Median_b(e) + K × IQR_b(e)
```

With:

```
K = 3.0
```

---

### Step 4: Floor Threshold

To prevent trivial zero-error alarms:

```
T_final = max(Threshold_b, 0.1 × μ_b)
```

Where `μ_b` is average load of building `b`.

---

## 3️⃣ Experimental Setup

### 3.1 Dataset & Split Strategy

* **Source**: Building Data Genome Project 2
* **Evaluation subset**: 200 buildings
* **Split**:

  * 80% train (chronological)
  * 20% test

Time-aware splitting avoids **data leakage**.

---

### 3.2 Fault Injection Protocol

Because real anomalies are unlabeled, synthetic faults are injected:

* Type: Point anomaly (sudden spike)
* Intensity:

```
y_injected = y_real × 1.5
```

* Injection rate: 5% of test points
* Scope: All buildings

---

## 4️⃣ Results & Discussion

### 4.1 Forecasting Performance

| Metric | Value   | Interpretation                |
| ------ | ------- | ----------------------------- |
| RMSE   | 14.7247 | Low global error              |
| MAE    | 1.9217  | Very small absolute deviation |

Confirms effectiveness of:

* Log normalization
* Lag-based features

---

### 4.2 Anomaly Detection Performance

Injected anomalies: **34,701 points**

| Metric    | Score      | Practical Meaning       |
| --------- | ---------- | ----------------------- |
| Precision | 0.9242     | Few false alarms        |
| Recall    | 0.9923     | Almost no missed faults |
| F1-score  | **0.9570** | Excellent balance       |

---

### 4.3 Visual Analysis

Figures (see notebooks) show:

* Forecast curve tightly following baseline load
* Green safe zone absorbing noise
* Red X anomalies precisely localized

---

## 5️⃣ Conclusion

This work delivers a **production-ready, scalable energy analytics framework**.

### Key Takeaways

1. **Efficiency**: One global model replaces thousands of local models
2. **Accuracy**: XGBoost + Optuna ensures strong predictive power
3. **Robustness**: IQR-based adaptive thresholds generalize across buildings

Achieved **F1 ≈ 0.96** in unlabeled anomaly detection.

---

## Future Work

* Explore drift and contextual anomalies
* Deploy as real-time inference API

---

## References

1. Miller, C., et al. *Building Data Genome Project 2*. Nature Scientific Data (2020)
2. Chen, T., & Guestrin, C. *XGBoost: A Scalable Tree Boosting System*. KDD (2016)
3. Stjelja, D., et al. *Energy & Buildings* (2024)
4. Ambat, A., & Sahoo, J. *ETRI Journal* (2024)
5. Alba, E. L., et al. *Forecasting* (2024)
