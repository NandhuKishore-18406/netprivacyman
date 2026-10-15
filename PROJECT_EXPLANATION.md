# Net-track: Technical Codebase Architecture & Explanation
## Faithful Reproduction of Net-track Random Forest Methodology (Lee, Joo & Lee, WWW 2023)

This document provides a comprehensive technical breakdown of every file in `src/netrack/`, explaining its role, key components, algorithms, and execution details.

---

## 1. Package Entry Point — `src/netrack/__init__.py`
- **File Path:** [`src/netrack/__init__.py`](file:///home/nandhu/projects/mlproject%28netrack%29/src/netrack/__init__.py)
- **Role:** Package metadata definition.
- **What it does:** Defines package version (`__version__ = "0.1.0"`).
- **How it works:** When Python imports `netrack`, this file initializes `netrack` as a top-level module namespace.

---

## 2. Dataset Engine — `src/netrack/dataset.py`
- **File Path:** [`src/netrack/dataset.py`](file:///home/nandhu/projects/mlproject%28netrack%29/src/netrack/dataset.py)
- **Role:** Web traffic crawling, synthetic flow generation, PCAP stream splitting, EasyList/EasyPrivacy filter list ground-truth labeling, and dataset split management.
- **Key Components:**
  1. **`WebTrafficCollector`**:
     - **What it does:** Collects network traffic from real-world website homepages.
     - **How it works:** Spawns a background `tshark` process capturing raw `.pcap` packet files on the target network interface for 120 seconds (`-a duration:120`). Simultaneously launches headless Chrome via Selenium to navigate to the target homepage and trigger all asynchronous ads/analytics scripts.
  2. **`BenchmarkFlowGenerator`**:
     - **What it does:** Generates realistic flow trace datasets with empirical distributions matching the paper target dataset scale (222,009 benign, 126,664 tracker flows).
     - **How it works:** Samples packet count distributions and simulates packet lengths with normal/exponential distributions:
       - **Tracker flows:** Higher uplink frequency (~35% client-to-server pings), smaller downlink payloads (1x1 pixel beacons / HTTP 204 responses).
       - **Benign flows:** Downlink heavy (~82% server-to-client), large MSS packet bursts (1400–1500 bytes for media/JS bundles).
       - Includes ~6% statistical noise to simulate real-world label ambiguities.
  3. **`FlowSplitter`**:
     - **What it does:** Extracts distinct client-server connection flows from multi-connection PCAP traces.
     - **How it works:** Reads raw packets using Scapy, inspects IP/IPv6 and TCP/UDP headers, and groups packets by their canonical 4-tuple key: `(src_ip, src_port, dst_ip, dst_port)`. It assigns direction $+1$ (uplink: client $\rightarrow$ server) or $-1$ (downlink: server $\rightarrow$ client) and records packet payload byte lengths.
  4. **`FilterListLabeler`**:
     - **What it does:** Implements the paper's ground-truth labeling rule (Section 3.1).
     - **How it works:** Downloads and compiles **EasyList** and **EasyPrivacy** filter rules using `adblockparser`. It inspects request URLs in a flow: if **any** URL matches an ad or tracker rule, the trace is labeled **`1` (`tracker`)**; otherwise **`0` (`benign`)**.
  5. **`DatasetManager`**:
     - **What it does:** Performs the **20% validation / 80% test split** (Section 4.2) and generates dataset scale/class balance reports.
     - **How it works:** Uses stratified sampling (`sklearn.model_selection.train_test_split`) to preserve exact class balance ratios (~63.7% Benign / ~36.3% Tracker) across both validation and test sets.

---

## 3. Algorithm 1 Feature Extractor — `src/netrack/features.py`
- **File Path:** [`src/netrack/features.py`](file:///home/nandhu/projects/mlproject%28netrack%29/src/netrack/features.py)
- **Role:** Algorithm 1 Feature Extractor (constructs the 97-dimensional feature vector $V$ per flow trace).
- **How it works:**
  1. **Packet Classification:** From raw packet sequence $P = [(dir_1, len_1), \dots]$, it constructs 4 arrays:
     - $A$: All packet lengths
     - $U$: Uplink packet lengths ($dir > 0$)
     - $D$: Downlink packet lengths ($dir < 0$)
     - $S$: Signed length sequence ($s_i = dir_i \cdot len_i$)
  2. **STAT Sub-vector (62 dimensions):**
     - For each of $U, D, A$ (3 lists), computes **18 summary statistics**: *min, max, mean, Median Absolute Deviation (MAD: $\text{median}(\|x - \text{median}(x)\|)$), std, variance, skewness, kurtosis, 10th to 90th percentiles in steps of 10, and count*. ($3 \times 18 = 54$ features).
     - Computes **8 flow-level features**: total byte sums for $U, D, A$; Shannon information entropy ($H = -\sum p(x) \log_2 p(x)$) for $U, D, A$; and uplink-to-downlink length & count ratios.
  3. **BOX Sub-vector (20 dimensions post-PCA):**
     - Bins packet lengths into 25-byte wide bins from 0 to 1500 bytes (bin 0: [0, 25), bin 1: [25, 50), ..., bin 60: =1500, bin 61: >1500 bytes) for both $U$ (61 bins) and $D$ (61 bins) $\rightarrow$ 122 raw dimensions.
     - `fit_pca()` fits `sklearn.decomposition.PCA(n_components=20)` **only on the training set**. `transform_box_pca()` reduces the 122 raw bin counts down to **20 PCA components**.
  4. **SEQ Sub-vector (15 dimensions):**
     - Extracts the first 15 signed packet lengths $s_1 \dots s_{15}$. Padded with zeros if shorter than 15, truncated if longer.
  5. **Concatenation:** Concatenates $\text{STAT} (62) + \text{BOX}_{\text{PCA}} (20) + \text{SEQ} (15) = \mathbf{97\text{-dimensional vector } V}$.

---

## 4. Classifier Engine — `src/netrack/model.py`
- **File Path:** [`src/netrack/model.py`](file:///home/nandhu/projects/mlproject%28netrack%29/src/netrack/model.py)
- **Role:** Random Forest Classifier wrapper adhering strictly to paper Section 5.1 hyperparameters.
- **Key Class: `NetTrackRandomForest`**:
  - **Hyperparameters:**
    - `n_estimators=55`: 55 decision trees in the ensemble.
    - `criterion='gini'`: Gini impurity metric for split decisions.
    - `max_depth=None`: Trees expanded until leaves are pure.
    - `max_features='sqrt'`: $\sqrt{97} \approx 9$ features sampled randomly per split.
  - **Inference Timing:** `evaluate_with_inference_time()` uses high-resolution timers (`time.perf_counter()`) to measure total evaluation runtime and calculates average per-trace inference latency ($\text{ms/trace}$).

---

## 5. Experiment & Visualization Suite — `src/netrack/experiments.py`
- **File Path:** [`src/netrack/experiments.py`](file:///home/nandhu/projects/mlproject%28netrack%29/src/netrack/experiments.py)
- **Role:** Contains all evaluation suites, ablation studies, false positive auditing, and figure rendering.
- **Key Components:**
  1. **`FeatureSelectionAblations`**:
     - Runs 10-fold cross-validation **exclusively on the 20% validation set** to reproduce Figures 4, 5, and 6:
       - **Fig 4:** Ranks STAT features by Gini importance on training folds and tests accuracy vs top-$k$ ($k \in [5 \dots 62]$).
       - **Fig 5:** Evaluates accuracy vs number of BOX PCA components ($n \in [2 \dots 50]$).
       - **Fig 6:** Evaluates accuracy vs sequence feature length ($L \in [5 \dots 73]$).
  2. **`RandomForestEvaluation`**:
     - Runs 10-fold cross-validation on the **80% test set** to evaluate final model metrics: Accuracy, Precision, Recall, F1-Score, Training Time, and Per-Trace Inference Time. Generates a markdown comparison table against Table 2 and Table 3.
  3. **`RealTimeDetectionExperiment`**:
     - Reproduces Figure 9 by truncating packet sequences to only the **first $n$ packets** ($n \in [5, 10, 15, \dots, 73]$), extracting features, and evaluating early-stage detection accuracy.
  4. **`UnlistedTrackerAuditor`**:
     - Reproduces Table 4 by sampling false positives (Ground Truth = Benign, RF Model = Tracker). Analyzes requested URLs and packet flow signatures to classify false alarms into 4 types: **Type 1 (Ads)**, **Type 2 (Tracking JS)**, **Type 3 (Telemetry Beacons)**, or **Benign**.
  5. **`NetTrackPlotter`**:
     - Uses Matplotlib's non-interactive `Agg` backend (`matplotlib.use("Agg")`) to safely save high-resolution publication figures (`fig4_stat_ablation.png`, `fig5_pca_ablation.png`, `fig6_seq_ablation.png`, `fig9_realtime_detection.png`) without thread/GUI errors.

---

## 6. CLI & Pipeline Orchestrator — `src/netrack/cli.py`
- **File Path:** [`src/netrack/cli.py`](file:///home/nandhu/projects/mlproject%28netrack%29/src/netrack/cli.py)
- **Role:** Command Line Interface (CLI) entry point and full pipeline orchestrator.
- **How it works:**
  - `run_full_pipeline()` coordinates the entire execution sequence:
    1. Generates/loads dataset & prints scale/class balance table.
    2. Runs validation ablations and saves Figs 4, 5, 6.
    3. Runs 10-fold CV test-set Random Forest evaluation (Table 2 & 3).
    4. Runs real-time first $n$-packet detection (Fig 9).
    5. Audits false positives (Table 4).
    6. Assembles and writes the complete markdown reproduction report artifact (`reports/nettrack_reproduction_report.md`).
  - Supports CLI flags via `argparse` (e.g., `uv run python -m netrack.cli pipeline --benign 2220 --tracker 1266`).

---

## Quick Start & Usage Commands

```bash
# 1. Activate virtual environment
source .venv/bin/activate.fish

# 2. Run unit test suite
uv run python -m unittest discover tests

# 3. Execute full reproduction pipeline
uv run python -m netrack.cli pipeline
```
