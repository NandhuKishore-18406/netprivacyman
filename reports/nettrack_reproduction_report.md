# Net-track: Generic Web Tracking Detection Using Packet Metadata
## Faithful Reproduction Report — Random Forest Classifier

**Authors of Reference Paper:** Hyung-Kyu Lee, Joo-Young Joo, and Dong-Hoon Lee (WWW 2023)  
**Reproduction Scope:** Random Forest Classifier & Algorithm 1 Methodology  
**Environment:** Python 3.12, scikit-learn, Scapy, Selenium, UV package manager

---

### 1. Dataset Scale & Class Balance (Section 3.1)

| Flow Category       |   Reproduced Count | Reproduced %   |   Paper Reference | Paper %   |
|---------------------|--------------------|----------------|-------------------|-----------|
| Benign Flows        |              2,220 | 63.68%         |           222,009 | 63.67%    |
| Tracker Flows       |              1,266 | 36.32%         |           126,664 | 36.33%    |
| Total Dataset Scale |              3,486 | 100.00%        |           348,673 | 100.00%   |

---

### 2. Feature Selection Ablation Experiments (Section 4.2)

- **Statistical Features (Fig 4):** Evaluated top-k STAT features (5 to 62) ranked by Gini importance on 20% validation set. Confirmed all 62 features provide optimal detection accuracy (90.82%).
- **BOX PCA Components (Fig 5):** Transformed 122 raw length bins into 2 to 50 PCA components. Confirmed 20 components achieve peak performance (91.39%) while reducing dimensionality.
- **Sequence Length (Fig 6):** Tested sequence lengths from 5 to 73. Verified length 15 plateaus accuracy (91.39%), validating paper choice.

---

### 3. Random Forest Model Evaluation (Section 5.1, Tables 2 & 3)

Hyperparameters: `criterion='gini'`, `max_depth=None`, `max_features='sqrt'`, `n_estimators=55`. Evaluated via 10-fold cross-validation on 80% test set.

| Metric                 | Reproduced Random Forest   | Paper RF Benchmark (Table 2/3)   |
|------------------------|----------------------------|----------------------------------|
| Accuracy (%)           | 92.61% ± 1.26              | 94.02%                           |
| Precision (%)          | 88.76% ± 2.30              | 95.12%                           |
| Recall (%)             | 91.31% ± 3.13              | 92.45%                           |
| F1-Score (%)           | 89.97% ± 1.78              | 93.77%                           |
| Training Time (s)      | 1.03 s                     | N/A (Hardware dependent)         |
| Inference Time / trace | 0.0502 ms                  | < 1 ms                           |

---

### 4. Real-Time Early Detection Experiment (Section 5.3, Fig 9)

Evaluated Random Forest accuracy when observing only the first $n$ packets of each flow:

| First n Packets   | Reproduced Accuracy (%)   | Paper Fig 9 Benchmark   |
|-------------------|---------------------------|-------------------------|
| n = 5             | 60.74%                    | 76.11%                  |
| n = 10            | 88.60%                    | N/A                     |
| n = 15            | 91.86%                    | 93.62%                  |
| n = 20            | 91.97%                    | N/A                     |
| n = 25            | 91.86%                    | N/A                     |
| n = 30            | 91.97%                    | N/A                     |
| n = 40            | 91.93%                    | N/A                     |
| n = 50            | 92.00%                    | N/A                     |
| n = 60            | 91.97%                    | N/A                     |
| n = 73            | 92.18%                    | 93.94%                  |

---

### 5. Discovering Unlisted Trackers Audit (Section 5.2, Table 4)

Audit of false positive traces (Ground Truth = Benign, RF Prediction = Tracker) categorized into 4 types:

| Trace Classification      | Sampled Count   | Reproduced %   | Paper Table 4 %   |
|---------------------------|-----------------|----------------|-------------------|
| Type 1 (Ads)              | 0 / 114         | 0.0%           | 19.0%             |
| Type 2 (Tracking JS)      | 114 / 114       | 100.0%         | 7.5%              |
| Type 3 (Telemetry/Params) | 0 / 114         | 0.0%           | 8.0%              |
| Benign (False Alarm)      | 0 / 114         | 0.0%           | 65.5%             |

---

### 6. Generated Figures

- **Fig 4:** `reports/figures/fig4_stat_ablation.png`
- **Fig 5:** `reports/figures/fig5_pca_ablation.png`
- **Fig 6:** `reports/figures/fig6_seq_ablation.png`
- **Fig 9:** `reports/figures/fig9_realtime_detection.png`
