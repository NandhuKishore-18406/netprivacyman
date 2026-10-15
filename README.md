# Net-track & Two-Tier Browser Fingerprinting Defense System

> **Sub-Packet Metadata Detection for Stateless Web Tracking and Browser Fingerprinting**  
> Reproducing & Extending *Net-track* (Lee, Joo & Lee, WWW 2023) with a Complementary Deep Learning Attention Network.

---

## 🚀 Key Highlights & Architecture

This project implements a **Two-Tier Complementary Defense System** capable of detecting encrypted third-party web trackers and modern browser fingerprinting vectors (Canvas, AudioContext, Canvas-Font, and WebRTC discovery) **without TLS interception or deep packet payload inspection**.

```
                           Incoming Encrypted Connection Flow
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: Line-Rate Random Forest (Net-track WWW 2023 Algorithm 1)                      │
│ - 97-dim Feature Vector: 62 STAT + 20 BOX (PCA) + 15 SEQ                               │
│ - Sub-millisecond evaluation (< 0.04 ms/trace)                                         │
│ - Filters 99%+ benign traffic at line rate                                             │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │ P(FP) < 0.30                              │ P(FP) >= 0.30 (Suspicious)
                     ▼                                           ▼
            ALLOWED (Benign Flow)       ┌────────────────────────────────────────────────┐
                                        │ TIER 2: Deep Learning Hybrid Attention Network │
                                        │ - Dual-Branch: MLP on STAT + 1D-CNN on SEQ     │
                                        │ - Packet Saliency Heatmaps (Self-Attention)    │
                                        │ - Multi-Class Technique Attribution            │
                                        └────────────────────────┬───────────────────────┘
                                                                 │
                                                                 ▼
                                                  BLOCKED + Forensic Diagnostics
                                                  (e.g., Canvas FP via Packet #8)
```

### 1. Tier 1: Line-Rate Random Forest Screener
- **Algorithm 1 Feature Extractor (97 dimensions)**:
  - **STAT (62 dims)**: Summary statistics (min, max, mean, MAD, std, variance, skewness, kurtosis, percentiles, entropy, byte/packet ratios) across Uplink, Downlink, and All packets.
  - **BOX_PCA (20 dims)**: 122 raw 25-byte length histogram bins reduced to 20 principal components via PCA.
  - **SEQ (15 dims)**: First 15 signed packet lengths ($s_i = \text{direction} \times \text{length}$).
- **Classifier**: Ensemble of 55 decision trees (`criterion='gini'`, `max_features='sqrt'`).
- **Throughput**: $< 0.04\text{ ms}$ per trace ($> 25,000\text{ flows/sec}$ per core).

### 2. Tier 2: Deep Learning Hybrid Attention Network
- **Dual-Branch PyTorch Architecture**:
  - **Branch A (MLP)**: Embeds 62 macroscopic STAT features into a 32-dim tabular context vector.
  - **Branch B (1D-CNN)**: Convolves raw signed packet sequences ($S_1 \dots S_{25}$) into temporal feature maps.
- **Self-Attention Pooling**: Learns attention weights $\alpha_1 \dots \alpha_{25}$ across packets. Pinpoints the exact packet carrying the serialized fingerprint upload without decrypting TLS payloads.
- **Multi-Class Technique Attribution**: Attributes flows into 5 distinct classes:
  1. `Benign Traffic`
  2. `Canvas Fingerprinting`
  3. `AudioContext Fingerprinting`
  4. `Canvas-Font Fingerprinting`
  5. `WebRTC Local IP Discovery`

### 3. Integrated Threat Intelligence Datasets
- **Princeton Web Census Fingerprinting TSVs** (`data/princeton_fingerprinting/`): Indexes 4,342 verified fingerprinting scripts across 516 commercial domains.
- **DuckDuckGo Tracker Radar** (`data/tracker-radar/`): Evaluates domain fingerprinting scores ($0\text{--}3$) directly from commercial tracker signatures.
- **Tranco Top Sites**: Benign domain traffic baseline.

---

## 📊 Performance Benchmarks

Evaluated on hold-out test sets using real-world Princeton & Tracker Radar threat intelligence:

| Evaluation Metric | Tier 1: Line-Rate Random Forest | Tier 2: Hybrid Attention Network |
|---|---|---|
| **Binary Detection Accuracy** | **100.00%** | **100.00%** |
| **Precision / Recall / F1** | **100.00% / 100.00% / 100.00%** | **100.00% / 100.00% / 100.00%** |
| **Binary ROC-AUC** | **1.0000** | **1.0000** |
| **Technique Attribution Accuracy** | *N/A (Binary Screener)* | **95.00%** |
| **Per-Trace Latency (CPU)** | **0.0349 ms** | **0.0076 ms** |
| **Model Artifact** | `models/tier1_rf.joblib` (~68 KB) | `models/tier2_hybrid_attention.pt` (~95 KB) |

#### Technique Attribution Accuracy:
- **Benign Traffic**: $100.00\%$
- **WebRTC Local IP Discovery**: $98.00\%$
- **Canvas-Font Fingerprinting**: $96.00\%$
- **AudioContext Fingerprinting**: $88.00\%$
- **Canvas Fingerprinting**: $78.00\%$

---

## 🛠️ Codebase Structure

```
├── data/                                 # Datasets (ignored in git)
│   ├── princeton_fingerprinting/         # TSVs for Canvas, Audio, Font, WebRTC
│   ├── tracker-radar/                    # DuckDuckGo Tracker Radar repository
│   └── priceton55k3pcblock/              # Princeton 55k blocking census dump
├── models/                               # Serialized model checkpoints
│   ├── tier1_rf.joblib                   # Trained Random Forest + PCA model
│   └── tier2_hybrid_attention.pt         # Trained Hybrid Attention PyTorch model
├── reports/                              # Generated reproduction artifacts
│   ├── figures/                          # Fig 4, 5, 6, 9 ablation charts
│   └── nettrack_reproduction_report.md   # Paper reproduction audit report
├── src/netrack/                          # Core Python package
│   ├── __init__.py                       # Package definition
│   ├── cli.py                            # Unified CLI (train, defense, pipeline)
│   ├── dataset.py                        # PCAP splitter, Princeton/Radar loaders, flow generator
│   ├── deep_model.py                     # HybridAttentionFingerprintNet, TwoTierDefense, Trainer
│   ├── experiments.py                    # Ablation studies & evaluation suites
│   ├── features.py                       # Algorithm 1 feature extractor (STAT, BOX, SEQ)
│   └── model.py                          # NetTrackRandomForest classifier
├── tests/
│   └── test_netrack.py                   # 12-test automated unit test suite
├── pyproject.toml                        # Project dependencies (managed with uv)
├── PROJECT_EXPLANATION.md                # Comprehensive technical guide
└── README.md
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/NandhuKishore-18406/netprivacyman.git
cd netprivacyman

# Set up virtual environment and install dependencies using uv
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Run the Unit Test Suite

```bash
python -m unittest discover tests
```
*(All 12 unit tests verify feature extraction, flow directionality, labelers, RF, deep model, and trainer).*

---

## 💻 CLI Commands

### 1. Train the Two-Tier Defense System
Trains Tier 1 Random Forest and Tier 2 Hybrid Attention Network and saves production weights to `models/`:

```bash
python -m netrack.cli train --epochs 10 --batch-size 32 --benign 1000 --per-technique 250
```

### 2. Run the Real-Time Defense Firewall Demo
Analyzes encrypted connections in real-time, displays Tier 1 latency, Tier 2 technique attribution, and the packet attention saliency heatmap:

```bash
python -m netrack.cli defense
```

**Example Output:**
```text
================================================================================
TWO-TIER COMPLEMENTARY BROWSER FINGERPRINTING DEFENSE
================================================================================
Tier 1: Line-Rate Random Forest (97-dim Algorithm 1 features, <0.1ms)
Tier 2: Deep Learning Attention Network (Technique Attribution & Packet Saliency)
Ground Truth: Princeton Web Census TSVs & DuckDuckGo Tracker Radar
================================================================================

[SCENARIO 1: Real-Time Analysis of an Encrypted Connection to 'https://fpjs.io/api/v3/telemetry']
- Total Packets: 11 | Total Uplink: 4045 B | Total Downlink: 1637 B
- Tier 1 Random Forest: Fingerprint Probability = 100.0% (Latency: 0.034 ms)
- Tier 2 Deep Attribution: Canvas Fingerprinting (Ground Truth: Canvas Fingerprinting, Radar Score: 3/3)
- Saliency Attention: Packet #8 (Uplink (Client -> Server), 907B) triggered peak attention (0.516)
- Final Firewall Decision: BLOCKED (Browser Fingerprinting)

[SCENARIO 2: Real-Time Analysis of an Encrypted Connection to 'https://example.com/index.html']
- Total Packets: 18 | Total Uplink: 1573 B | Total Downlink: 13496 B
- Tier 1 Random Forest: Fingerprint Probability = 0.0% (Latency: 0.031 ms)
- Tier 2 Deep Attribution: Benign (Ground Truth: Benign, Radar Score: 0/3)
- Final Firewall Decision: ALLOWED (Benign)
```

### 3. Run the Full WWW 2023 Reproduction Pipeline
Reproduces the empirical findings, ablation experiments (Figures 4, 5, 6, 9), and Tables 2, 3, 4 from the WWW 2023 paper:

```bash
python -m netrack.cli pipeline --benign 2220 --tracker 1266
```

---

## 📖 Technical Documentation

For an in-depth component-by-component architectural walkthrough explaining every module and mathematical formula, see [PROJECT_EXPLANATION.md](file:///home/nandhu/projects/mlproject%28netrack%29/PROJECT_EXPLANATION.md).

---

## 📜 References
- **Net-track (WWW 2023)**: Hyung-Kyu Lee, Joo-Young Joo, and Dong-Hoon Lee. *"Net-track: Generic Web Tracking Detection Using Packet Metadata."* Proceedings of the ACM Web Conference 2023 (WWW '23).
- **Princeton Web Census**: Steven Englehardt and Arvind Narayanan. *"Online Tracking: A 1-Million-Site Measurement and Analysis."* ACM CCS 2016.
- **DuckDuckGo Tracker Radar**: DuckDuckGo Open Source Tracker Intelligence Dataset.
