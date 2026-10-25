# PMAD: Predictive Maintenance & Anomaly Detection
## Complete & Concise Project Defense Guide (B.Tech AI & Data Science)

> **Author**: Bhavesh Bargat  
> **Target Domain**: Heavy Manufacturing & Defense (e.g., **Solar Industries India Limited**)  
> **Key Goal**: Explain the complete project with zero fluff, crystal-clear analogies, and 100% of the model details.

---

## 1. The 30-Second Elevator Pitch

> "I built **PMAD**, an industrial AI system that prevents catastrophic machine breakdowns in continuous heavy manufacturing, like the chemical slurry mixers and CNC shell units at **Solar Industries India Limited**.  
> 
> Instead of feeding raw sensor numbers into a black box, I engineered physics-based features: **Mechanical Power ($P = \tau \cdot \omega$)**, **Coolant Heat Gradients ($\Delta T$)**, and **Tool Fatigue**.  
> 
> I deployed a **dual AI architecture**: a **Balanced Random Forest** achieving **98.75% accuracy and 0.9758 ROC-AUC** for known failure prediction, paired with an **Unsupervised Isolation Forest (0.8691 Alignment ROC-AUC)** to catch novel zero-day anomalies. Finally, the system maps predictions directly into factory **Standard Operating Procedures (SOPs)**."

---

## 2. The Core Problem (Solar Industries Context)

| Maintenance Type | What It Does | Why It Fails in Explosives / Defense Plants |
| :--- | :--- | :--- |
| **Reactive** *(Run-to-failure)* | Fix machine *after* it breaks | Friction spike or motor stall in an energetic slurry mixer causes **overheating, fire, or explosion**. Unacceptable safety risk. |
| **Preventive** *(Calendar-based)* | Replace tool every Monday | Throws away expensive carbide inserts with 40% life left, or tool snaps on Sunday before inspection. |
| **PMAD (Predictive & Prescriptive)** | Continuous AI sensor telemetry | **Predicts failure hours in advance**, explains root cause, and prescribes the exact SOP to prevent downtime. |

---

## 3. Telemetry & Target Leakage (The #1 Student Mistake)

### Raw Sensors
- **Air Temp ($T_{\text{air}}$)**: Room ambient temperature (295–304 K).
- **Process Temp ($T_{\text{process}}$)**: Tool-workpiece interface temperature (305–314 K).
- **Rotational Speed**: Spindle velocity (1,168–2,886 RPM).
- **Torque**: Twisting force applied by drive motor (3.8–76.6 Nm).
- **Tool Wear**: Cumulative cutting time (0–253 minutes).
- **Type**: Machine variant: **L** (Light 50%), **M** (Medium 30%), **H** (Heavy-Duty 20%).

### ⚠️ The Target Leakage Trap
- **The Trap**: The dataset contains 5 diagnostic tags: `TWF` (Tool Wear), `HDF` (Heat Dissipation), `PWF` (Power), `OSF` (Overstrain), `RNF` (Random). Leaving these in gives **99.9% fake accuracy**!
- **The Analogy**: *Checking the autopsy report to predict if a patient will die.* In a factory, those flags only exist **after** the machine has already broken down.
- **Our Strict Rule**: Dropped `UDI`, `Product ID`, and all 5 post-failure flags. Models train **strictly** on pre-failure sensor physics.

---

## 4. Physics-Informed Feature Engineering

We transform raw sensor numbers into mechanical engineering laws:

| Feature | Equation | Physical Principle & Failure Threshold |
| :--- | :--- | :--- |
| **$\Delta T$ (Thermal Gradient)** | $T_{\text{process}} - T_{\text{air}}$ | Heat dissipation: If $\Delta T < 8.6\text{ K}$ at low RPM (< 1,380), coolant circulation failed $\to$ **Heat Dissipation Failure (HDF)**. |
| **Mechanical Power ($P$)** | $\text{Torque} \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$ | Motor envelope: Safe operating zone is **3,500 W to 9,000 W**. Demand $> 9,000\text{ W}$ stalls the motor $\to$ **Power Failure (PWF)**. |
| **Overstrain Fatigue Index** | $\text{Torque} \times \text{Tool Wear}$ | Tool loading: When torque $\times$ wear exceeds structural limit (**11k for L, 12k for M, 13k for H**), shank snaps $\to$ **Overstrain Failure (OSF)**. |
| **Speed-Torque Ratio** | $\frac{\text{RPM}}{\text{Torque} + 10^{-5}}$ | Operating regime: High speed/low torque = finishing; Low speed/high torque = roughing cut. |

---

## 5. The Machine Learning Models: Complete Technical Breakdown

```
                             [ Raw Sensor Telemetry ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      [ Balanced Random Forest ]                     [ Isolation Forest ]
      (Supervised Classifier)                     (Unsupervised Anomaly)
      • 150 Decision Trees                        • 150 Isolation Trees
      • Handles 3.39% Imbalance                   • Zero-Day Anomaly Detection
      • Predicts Known Breakdowns                 • Detects Distribution Shifts
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                        [ Prescriptive SOP Engine ]
                        • Deterministic Validation
                        • Automated Plant SOP Dispatch
```

---

### Model 1: Balanced Random Forest (Supervised Production)

* **Architecture & Hyperparameters**:
  * `n_estimators = 150` (150 de-correlated decision trees).
  * `max_depth = 14` (prevents memorization while capturing non-linear interactions).
  * `min_samples_split = 4`, `min_samples_leaf = 2`.
  * `class_weight = 'balanced'` (auto-adjusts tree split weights inversely proportional to class frequencies).
* **How It Works**:
  1. **Bagging**: Each tree trains on a random bootstrap sample with replacement.
  2. **Feature Subspacing**: At each split, only a random subset ($\sqrt{d}$) of features is considered.
  3. **Majority Ensemble Vote**: Aggregates 150 predictions into a calibrated probability $P(\text{Failure})$.
* **Why `class_weight='balanced'` is Mandatory**:
  * Dataset has **9,661 healthy (96.61%)** vs **339 failures (3.39%)**.
  * A naive model predicting "Healthy" every time gets **96.6% accuracy** while missing 100% of breakdowns!
  * `class_weight='balanced'` penalizes false negatives by $\approx 28.5\times$, forcing trees to prioritize the rare failure class.
* **Exact Empirical Test Performance (Held-out Test Set)**:
  * **Accuracy**: **98.75%**
  * **ROC-AUC**: **0.9758** (exceptional separation across all thresholds)
  * **PR-AUC**: **0.8651** (true gold standard on imbalanced data)
  * **Recall**: **86.76%** (catches nearly 9 out of 10 impending failures)
  * **Precision**: **78.67%**
  * **F1-Score**: **0.8252** | **F2-Score**: **0.8501** (favors recall)
  * **Confusion Matrix**: $\text{TN} = 1,916$, $\text{FP} = 16$, $\text{FN} = 9$, $\text{TP} = 59$.
* **Feature Importance: Gini vs Permutation**:
  * **Gini Impurity (Split Frequency)**: Favors continuous ratios $\to$ `Speed_Torque_Ratio` (0.1691) and `Power_W` (0.1589).
  * **Permutation Importance (Generalization Drop on Test Data)**: Proves true physical drivers $\to$ `Tool wear` (drop: 0.0260) and `Temp_Difference_K` (drop: 0.0230) rank #1 and #2.

---

### Model 2: Isolation Forest (Unsupervised Production)

* **Why Unsupervised?**:
  * Supervised models are blind to **novel, zero-day failures** never seen in historical training data (e.g., contaminated lubricant batch, loose mounting bolt, cracked bearing).
* **Architecture & Hyperparameters**:
  * `n_estimators = 150`, `contamination = 0.04` (prior estimated breakdown rate), `random_state = 42`.
* **Mathematical Mechanism**:
  * Partitions space recursively using random orthogonal hyperplanes.
  * **Path Length $h(x)$**: Normal points cluster tightly and require deep trees to isolate. Anomalies exist in sparse periphery and get isolated in very few splits ($h(x) \ll c(n)$).
  * **Anomaly Score**: $s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$, normalized to $0.0 - 1.0$ ($0\% - 100\%$).
* **Post-Hoc Alignment Validation (Evaluated Against True Failures Without Seeing Labels)**:
  * **Alignment ROC-AUC**: **0.8691**
  * **Sensitivity by Failure Mode**:
    * **Power Failures (PWF)**: **83.2% detected** (violent electrical/power outliers are easily isolated).
    * **Overstrain (OSF)**: **50.0% detected**.
    * **Heat Dissipation (HDF)**: **15.7% detected**.
    * **Tool Wear (TWF)**: **10.9% detected** (gradual mechanical wear blends into normal clusters, proving why supervised learning is also needed!).

---

### Model 3: Baseline Comparison (Why Logistic Regression Failed)

| Metric | Logistic Regression (Baseline) | Balanced Random Forest (PMAD) | Why Random Forest Won |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 84.95% | **98.75%** | +13.80% overall accuracy |
| **Precision** | **17.00%** | **78.67%** | **4.6x higher precision** |
| **Recall** | 88.24% | **86.76%** | Comparable high recall |
| **PR-AUC** | 0.4486 | **0.8651** | Nearly double PR curve area |
| **False Alarms (FP)**| **293 false alarms** | **16 false alarms** | **18x reduction in false alarms!** |

* **The Core Reason**: Logistic Regression fits a linear plane ($z = w^T x + b$). It cannot model non-linear physics—like the hyperbolic boundary of Mechanical Power ($P = \tau \cdot \omega$) or the sudden cliff when Overstrain exceeds 11,000—without triggering 293 false alarms.

---

## 6. Business Math: Why We Tuned Decision Threshold to 0.35

* **The Standard Default**: Scikit-Learn defaults to cutoff $T = 0.50$.
* **The Cost Asymmetry in Heavy Industry**:
  $$\text{Total Cost} = (C_{\text{FN}} \times \text{FN}) + (C_{\text{FP}} \times \text{FP})$$
  * **Cost of False Negative ($C_{\text{FN}}$)**: $\approx \text{₹5,00,000}$ (catastrophic tool smash, machine downtime, safety hazard).
  * **Cost of False Positive ($C_{\text{FP}}$)**: $\approx \text{₹5,000}$ (10-minute visual inspection by a technician).
  * A missed breakdown is **$100\times$ more expensive** than a false alarm!
* **The Optimization**:
  * Lowering the threshold to **$0.35$** aggressively minimizes total plant financial loss, boosting failure recall to $>90\%$ while keeping false alarms strictly controlled.

---

## 7. Deterministic Sensor Validation (Engineering Integrity)

Before telemetry enters the ML models, it must pass physical boundary checks:
1. **Thermodynamic Invariant**:
   $$T_{\text{process}} \ge T_{\text{air}} - 2.0\text{ K}$$
   In an air-cooled machine without a refrigerator, process heat cannot be cooler than room air. If violated $\to$ Flag calibration error / sensor disconnect!
2. **Operational Bounds**:
   * Air Temp: $280 - 335\text{ K}$ | Process Temp: $285 - 345\text{ K}$
   * RPM: $500 - 4,000\text{ RPM}$ | Torque: $0 - 120\text{ Nm}$ | Tool Wear: $0 - 350\text{ min}$

---

## 8. Prescriptive Maintenance SOPs & Tool RUL

PMAD translates AI probabilities into factory actions:

$$\text{Tool RUL [min]} = \max\left(0, \frac{\text{Limit}_{\text{variant}} - \text{Overstrain Index}}{\text{Torque}}\right)$$

| Condition | Status | Prescriptive Factory SOP |
| :--- | :--- | :--- |
| **Tool Wear $> 200\text{ min}$ or RUL $< 15\text{ min}$** | `CRITICAL` | **SOP-TL-01**: Tool Wear Exhaustion. Pause feed axis at cycle end. Replace carbide insert. |
| **$\Delta T < 8.6\text{ K}$ at low RPM** | `CRITICAL` | **SOP-TH-02**: Thermal Dissipation Failure. Inspect coolant level, pump pressure, and mesh strainer. |
| **Mechanical Power $> 9,000\text{ W}$** | `CRITICAL` | **SOP-DR-03**: Motor Overload Hazard. E-stop feed axis. Check mechanical jam and 3-phase current balance. |
| **Overstrain $> 85\%$ of variant limit** | `WARNING` | **SOP-OS-03**: Excessive Mechanical Load. Reduce feed rate 15%, schedule tool swap at shift end. |
| **Anomaly Score $> 70\%$ but RF Prob $< 35\%$** | `WARNING` | **SOP-AN-05**: Zero-Day Drift Alert. Check sensor wiring calibration and spindle bearing vibration. |
| **All channels nominal** | `NOMINAL` | System healthy. Log telemetry and continue normal production cycle. |

---

## 9. Dashboard Architecture (9 Streamlit Tabs)

1. **Executive Overview**: Fleet KPIs, failure rate breakdown, one-click PDF/MD Plant Audit Report download.
2. **Data Explorer**: Interactive multi-sensor range filters, raw data inspection, and CSV export.
3. **Sensor Analytics**: Correlation matrix, thermal distributions, and Speed vs Torque power envelopes.
4. **Anomaly Detection**: 2D/3D Isolation Forest cluster projections and failure mode sensitivity breakdowns.
5. **Failure Prediction**: Real-time single machine diagnostic gauge, Z-score sensor profiles, and batch CSV inferencing.
6. **Maintenance SOPs**: Actionable alert feed with step-by-step standard operating procedures.
7. **Model Evaluation**: 12-metric side-by-side benchmark, ROC/PR curves, financial cost simulator, and permutation importance.
8. **Simulated Live Monitoring**: Demonstration edge streaming console with live fault injection buttons (power trip, tool wear, thermal lockup).
9. **System Architecture**: Complete engineering pipeline documentation and defense guide.

---

## 10. Top 7 Viva & Interview Quick-Fire Answers

**Q1: What did you personally build in this project?**  
> *"I designed and built the full pipeline: zero-leakage data sanitization, 4 physics-grounded features, a dual ML architecture (Balanced Random Forest + Isolation Forest), deterministic sensor validation, prescriptive SOP mapping, a 28-test Pytest suite, and a 9-tab Streamlit dashboard."*

**Q2: Why Random Forest instead of XGBoost or Deep Learning?**  
> *"Tabular industrial sensor data has sharp physical boundaries (e.g., motor trip at 9 kW). Random Forest with `class_weight='balanced'` handles extreme 3.39% imbalance cleanly without synthetic SMOTE artifacts, provides sub-millisecond edge latency, and avoids deep learning overfitting on 10,000 records."*

**Q3: Why do Gini and Permutation importances give different rankings?**  
> *"Gini measures training split frequency, favoring continuous combinations like `Speed_Torque_Ratio`. Permutation importance measures true generalization drop on unseen test data when features are shuffled—proving that cumulative `Tool wear` and `Temp_Difference_K` are the true physical drivers of failure."*

**Q4: How can you evaluate Isolation Forest if it is unsupervised?**  
> *"It was trained strictly unsupervised without ground-truth labels. Post-training, we benchmarked its anomaly scores against true events: it achieved an alignment ROC-AUC of 0.8691, detecting 83.2% of abrupt power trips and 50.0% of overstrain events purely through spatial density isolation."*

**Q5: Why did you lower the decision threshold to 0.35?**  
> *"Industrial costs are asymmetric: an unplanned breakdown ($C_{\text{FN}}$) costs ₹5,00,000, while a 10-minute check ($C_{\text{FP}}$) costs ₹5,000. Lowering the threshold to 0.35 minimizes total financial risk and maximizes recall on catastrophic events."*

**Q6: What is Target Leakage and how did you prevent it?**  
> *"Leaving failure mode flags (`TWF`, `HDF`, `PWF`, etc.) in the features is target leakage—it's like reading the post-mortem report before the patient dies. I removed all 5 failure tags and IDs, training strictly on telemetry that exists while the machine is running."*

**Q7: How does this apply to Solar Industries India Limited?**  
> *"Solar Industries manufactures defense propellants, explosives, and warheads. Unmonitored torque spikes in slurry mixers can trigger catastrophic thermal runaway. PMAD's real-time $\Delta T$ and power monitoring prevents thermal lockups, tracks CNC tool RUL for munition casings, and automatically triggers prescriptive plant SOPs."*

---
*Ready to run: `pytest` for tests | `python train_all.py` to retrain | `streamlit run app.py` for dashboard.*
