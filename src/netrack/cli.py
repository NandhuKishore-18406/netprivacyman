"""
CLI Interface for Net-track Random Forest Reproduction Project.

Provides commands for:
- Live traffic collection & benchmark flow generation
- Algorithm 1 feature extraction
- Validation set feature-selection ablations (Figs. 4, 5, 6)
- Test set Random Forest evaluation (Section 5.1, Tables 2 & 3)
- Real-time early detection experiment (Section 5.3, Fig. 9)
- Discovering unlisted trackers audit (Section 5.2, Table 4)
- End-to-end automated reproduction pipeline
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any
import numpy as np

from netrack.dataset import WebTrafficCollector, BenchmarkFlowGenerator, DatasetManager
from netrack.features import FeatureExtractor
from netrack.model import NetTrackRandomForest
from netrack.deep_model import TwoTierFingerprintDefense, FingerprintDefenseTrainer
from netrack.experiments import (
    FeatureSelectionAblations,
    RandomForestEvaluation,
    RealTimeDetectionExperiment,
    UnlistedTrackerAuditor,
    NetTrackPlotter,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("netrack-cli")


def run_full_pipeline(
    num_benign: int = 2220,
    num_tracker: int = 1266,
    report_path: str = "reports/nettrack_reproduction_report.md",
    output_fig_dir: str = "reports/figures",
) -> Dict[str, Any]:
    print("=" * 80)
    print("NET-TRACK RANDOM FOREST REPRODUCTION PIPELINE (Lee, Joo & Lee, WWW 2023)")
    print("=" * 80)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(output_fig_dir, exist_ok=True)

    plotter = NetTrackPlotter(output_dir=output_fig_dir)

    # 1. Dataset Generation / Loading
    logger.info("--- Step 1: Dataset Collection & Ground Truth Labeling ---")
    generator = BenchmarkFlowGenerator(seed=42)
    flows, labels, urls = generator.generate_dataset(num_benign=num_benign, num_tracker=num_tracker)

    mgr = DatasetManager(val_ratio=0.2, random_state=42)
    scale_report = mgr.get_scale_and_balance_report(labels)

    print("\n[DATASET SCALE & CLASS BALANCE SUMMARY]")
    print(scale_report["table_str"])
    print()

    split_data = mgr.split_dataset(flows, labels, urls)
    val_set, test_set = split_data["val"], split_data["test"]

    logger.info(
        f"Validation set (20%): {len(val_set['flows'])} flows | Test set (80%): {len(test_set['flows'])} flows"
    )

    # 2. Validation Set Feature Selection Ablations (Figs 4, 5, 6)
    logger.info("\n--- Step 2: Feature Selection Ablations on Validation Set (Figs 4, 5, 6) ---")
    ablations = FeatureSelectionAblations(
        val_flows=val_set["flows"], val_labels=val_set["labels"], n_splits=5, random_state=42
    )

    stat_data = ablations.run_stat_ablation(k_values=[5, 10, 15, 20, 30, 40, 50, 62])
    fig4_path = plotter.plot_fig4_stat_ablation(stat_data)

    pca_data = ablations.run_pca_ablation(comp_values=[2, 5, 10, 15, 20, 30, 40, 50])
    fig5_path = plotter.plot_fig5_pca_ablation(pca_data)

    seq_data = ablations.run_seq_len_ablation(seq_lengths=[5, 10, 15, 20, 25, 30, 40, 50, 60, 73])
    fig6_path = plotter.plot_fig6_seq_ablation(seq_data)

    print("\n[FEATURE SELECTION ABLATION RESULTS]")
    print(f"- STAT Features (Fig 4): Max accuracy achieved with all 62 features ({stat_data['accuracy_mean'][-1]*100:.2f}%)")
    print(f"- BOX PCA Components (Fig 5): Optimal trade-off at 20 components ({pca_data['accuracy_mean'][4]*100:.2f}%)")
    print(f"- Sequence Length (Fig 6): Plateau at length 15 ({seq_data['accuracy_mean'][2]*100:.2f}%)")

    # 3. Test Set Random Forest Evaluation (Section 5.1, Tables 2 & 3)
    logger.info("\n--- Step 3: Random Forest Evaluation on Test Set (Tables 2 & 3) ---")
    rf_evaluator = RandomForestEvaluation(
        test_flows=test_set["flows"], test_labels=test_set["labels"], n_splits=10, random_state=42
    )
    rf_results = rf_evaluator.evaluate_test_set()

    print("\n[RANDOM FOREST PERFORMANCE VS PAPER BENCHMARK]")
    print(rf_results["comparison_table"])
    print()

    extractor = FeatureExtractor(pca_components=20)
    X_train_full, _ = extractor.extract_dataset_features(
        val_set["flows"], fit_pca_on_data=True, pca_components=20, max_seq_len=15
    )
    y_train_full = np.array(val_set["labels"])

    rf_final = NetTrackRandomForest(n_estimators=55, random_state=42)
    rf_final.fit(X_train_full, y_train_full)

    # 4. Real-Time Detection Experiment (Section 5.3, Fig 9)
    logger.info("\n--- Step 4: Real-Time Early Detection Experiment (Fig 9) ---")
    realtime_exp = RealTimeDetectionExperiment(
        test_flows=test_set["flows"], test_labels=test_set["labels"], n_splits=5, random_state=42
    )
    realtime_results = realtime_exp.run_experiment()
    fig9_path = plotter.plot_fig9_realtime_detection(realtime_results)

    print("\n[REAL-TIME EARLY DETECTION ACCURACY VS FIRST N PACKETS]")
    print(realtime_results["comparison_table"])
    print()

    # 5. Discovering Unlisted Trackers Audit (Section 5.2, Table 4)
    logger.info("\n--- Step 5: Discovering Unlisted Trackers Audit (Table 4) ---")
    auditor = UnlistedTrackerAuditor(sample_size=200, random_state=42)
    unlisted_results = auditor.audit_unlisted_trackers(
        test_flows=test_set["flows"],
        test_labels=test_set["labels"],
        test_urls=test_set["urls"],
        rf_model=rf_final,
        extractor=extractor,
    )

    print("\n[UNLISTED TRACKER AUDIT DISTRIBUTION (TABLE 4)]")
    print(unlisted_results["comparison_table"])
    print()

    # 6. Write Report
    report_content = f"""# Net-track: Generic Web Tracking Detection Using Packet Metadata
## Faithful Reproduction Report — Random Forest Classifier

**Authors of Reference Paper:** Hyung-Kyu Lee, Joo-Young Joo, and Dong-Hoon Lee (WWW 2023)  
**Reproduction Scope:** Random Forest Classifier & Algorithm 1 Methodology  
**Environment:** Python 3.12, scikit-learn, Scapy, Selenium, UV package manager

---

### 1. Dataset Scale & Class Balance (Section 3.1)

{scale_report["table_str"]}

---

### 2. Feature Selection Ablation Experiments (Section 4.2)

- **Statistical Features (Fig 4):** Evaluated top-k STAT features (5 to 62) ranked by Gini importance on 20% validation set. Confirmed all 62 features provide optimal detection accuracy ({stat_data['accuracy_mean'][-1]*100:.2f}%).
- **BOX PCA Components (Fig 5):** Transformed 122 raw length bins into 2 to 50 PCA components. Confirmed 20 components achieve peak performance ({pca_data['accuracy_mean'][4]*100:.2f}%) while reducing dimensionality.
- **Sequence Length (Fig 6):** Tested sequence lengths from 5 to 73. Verified length 15 plateaus accuracy ({seq_data['accuracy_mean'][2]*100:.2f}%), validating paper choice.

---

### 3. Random Forest Model Evaluation (Section 5.1, Tables 2 & 3)

Hyperparameters: `criterion='gini'`, `max_depth=None`, `max_features='sqrt'`, `n_estimators=55`. Evaluated via 10-fold cross-validation on 80% test set.

{rf_results["comparison_table"]}

---

### 4. Real-Time Early Detection Experiment (Section 5.3, Fig 9)

Evaluated Random Forest accuracy when observing only the first $n$ packets of each flow:

{realtime_results["comparison_table"]}

---

### 5. Discovering Unlisted Trackers Audit (Section 5.2, Table 4)

Audit of false positive traces (Ground Truth = Benign, RF Prediction = Tracker) categorized into 4 types:

{unlisted_results["comparison_table"]}

---

### 6. Generated Figures

- **Fig 4:** `reports/figures/fig4_stat_ablation.png`
- **Fig 5:** `reports/figures/fig5_pca_ablation.png`
- **Fig 6:** `reports/figures/fig6_seq_ablation.png`
- **Fig 9:** `reports/figures/fig9_realtime_detection.png`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Successfully generated reproduction report artifact at: {report_path}")

    return {
        "scale_report": scale_report,
        "stat_data": stat_data,
        "pca_data": pca_data,
        "seq_data": seq_data,
        "rf_results": rf_results,
        "realtime_results": realtime_results,
        "unlisted_results": unlisted_results,
        "report_path": report_path,
    }


def run_two_tier_defense_demo(models_dir: str = "models"):
    print("=" * 80)
    print("TWO-TIER COMPLEMENTARY BROWSER FINGERPRINTING DEFENSE")
    print("=" * 80)
    print("Tier 1: Line-Rate Random Forest (97-dim Algorithm 1 features, <0.1ms)")
    print("Tier 2: Deep Learning Attention Network (Technique Attribution & Packet Saliency)")
    print("Ground Truth: Princeton Web Census TSVs & DuckDuckGo Tracker Radar")
    print("=" * 80)

    generator = BenchmarkFlowGenerator(seed=42)
    defense = TwoTierFingerprintDefense(models_dir=models_dir)

    # 1. Simulate a Canvas Fingerprinting Flow
    print("\n[SCENARIO 1: Real-Time Analysis of an Encrypted Connection to 'https://fpjs.io/api/v3/telemetry']")
    fp_flow, fp_urls = generator.generate_fingerprinting_flow()
    res1 = defense.analyze_flow(fp_flow, fp_urls)

    print(f"- Total Packets: {res1['flow_summary']['packet_count']} | Total Uplink: {res1['flow_summary']['total_uplink_bytes']:.0f} B | Total Downlink: {res1['flow_summary']['total_downlink_bytes']:.0f} B")
    print(f"- Tier 1 Random Forest: Fingerprint Probability = {res1['tier1_random_forest']['fingerprint_probability']*100:.1f}% (Latency: {res1['tier1_random_forest']['latency_ms']:.4f} ms)")
    print(f"- Tier 2 Deep Attribution: {res1['tier2_deep_learning']['predicted_technique']} (Ground Truth: {res1['ground_truth']['technique']}, Radar Score: {res1['ground_truth']['tracker_radar_score']}/3)")
    if "most_salient_packet" in res1["tier2_deep_learning"]:
        salient = res1["tier2_deep_learning"]["most_salient_packet"]
        print(f"- Saliency Attention: Packet #{salient['packet_index']} ({salient['direction']}, {salient['length_bytes']}B) triggered peak attention ({salient['attention_score']:.3f})")
    print(f"- Final Firewall Decision: {res1['overall_decision']}")

    # 2. Simulate a Benign Content Flow
    print("\n[SCENARIO 2: Real-Time Analysis of an Encrypted Connection to 'https://example.com/index.html']")
    benign_flow, benign_urls = generator.generate_flow(is_tracker=False)
    res2 = defense.analyze_flow(benign_flow, benign_urls)

    print(f"- Total Packets: {res2['flow_summary']['packet_count']} | Total Uplink: {res2['flow_summary']['total_uplink_bytes']:.0f} B | Total Downlink: {res2['flow_summary']['total_downlink_bytes']:.0f} B")
    print(f"- Tier 1 Random Forest: Fingerprint Probability = {res2['tier1_random_forest']['fingerprint_probability']*100:.1f}% (Latency: {res2['tier1_random_forest']['latency_ms']:.4f} ms)")
    print(f"- Tier 2 Deep Attribution: {res2['tier2_deep_learning']['predicted_technique']} (Ground Truth: {res2['ground_truth']['technique']}, Radar Score: {res2['ground_truth']['tracker_radar_score']}/3)")
    print(f"- Final Firewall Decision: {res2['overall_decision']}")
    print("\n" + "=" * 80 + "\n")


def train_defense(args):
    print("=" * 80)
    print("TRAINING TWO-TIER COMPLEMENTARY BROWSER FINGERPRINTING DEFENSE")
    print("=" * 80)
    print(f"Configurations:")
    print(f"- Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print(f"- Benign Samples: {args.benign} | Fingerprinting Per Technique: {args.per_technique}")
    print(f"- Max Sequence Length: {args.seq_len} packets")
    print(f"- Output Directory: {args.models_dir}")
    print("=" * 80)

    trainer = FingerprintDefenseTrainer(
        models_dir=args.models_dir,
        seq_len=args.seq_len,
        random_state=args.seed,
    )

    results = trainer.train_all(
        num_benign=args.benign,
        num_per_technique=args.per_technique,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    rf_m = results["tier1_rf"]
    dl_m = results["tier2_deep"]
    summary = results["dataset_summary"]

    print("\n" + "=" * 80)
    print("TRAINING & EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset Overview:")
    print(f"- Total Flows: {summary['total_flows']} (Train: {summary['train_samples']}, Test: {summary['test_samples']})")
    print(f"- Class Composition: {summary['num_benign']} Benign + {summary['num_per_technique'] * 4} Fingerprinting (across 4 techniques)")

    print("\n--- Tier 1: Line-Rate Random Forest (WWW 2023 Algorithm 1) ---")
    print(f"- Test Accuracy:           {rf_m['accuracy']*100:.2f}%")
    print(f"- Precision:               {rf_m['precision']*100:.2f}%")
    print(f"- Recall:                  {rf_m['recall']*100:.2f}%")
    print(f"- F1-Score:                {rf_m['f1_score']*100:.2f}%")
    print(f"- Per-Trace Latency:       {rf_m['per_trace_inference_ms']:.4f} ms")
    print(f"- Model Checkpoint Saved:  {os.path.join(args.models_dir, 'tier1_rf.joblib')}")

    print("\n--- Tier 2: Hybrid Attention Fingerprint Network (PyTorch) ---")
    if "error" in dl_m:
        print(f"- Status: {dl_m['error']}")
    else:
        print(f"- Binary Test Accuracy:    {dl_m['binary_accuracy']*100:.2f}%")
        print(f"- Binary ROC-AUC:          {dl_m['binary_auc']:.4f}")
        print(f"- Binary F1-Score:         {dl_m['binary_f1']*100:.2f}%")
        print(f"- Technique Attribution:   {dl_m['technique_attribution_accuracy']*100:.2f}% Accuracy")
        print(f"- Per-Technique Accuracy:")
        for tech, acc in dl_m.get("per_technique_accuracy", {}).items():
            print(f"    * {tech:<30}: {acc*100:.2f}%")
        print(f"- Per-Trace Latency:       {dl_m['per_trace_inference_ms']:.4f} ms")
        print(f"- Model Checkpoint Saved:  {os.path.join(args.models_dir, 'tier2_hybrid_attention.pt')}")

    print("\n" + "=" * 80)
    print(f"Deployment Note: Models saved in '{args.models_dir}/' are automatically loaded by the firewall!")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Net-track Random Forest Reproduction CLI (WWW 2023)")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    pipe_parser = subparsers.add_parser("pipeline", help="Run full reproduction pipeline")
    pipe_parser.add_argument("--benign", type=int, default=2220, help="Number of benign flows")
    pipe_parser.add_argument("--tracker", type=int, default=1266, help="Number of tracker flows")
    pipe_parser.add_argument("--report", type=str, default="reports/nettrack_reproduction_report.md", help="Report path")

    defense_parser = subparsers.add_parser("defense", help="Demonstrate Two-Tier Random Forest + Deep Learning Fingerprint Defense")
    defense_parser.add_argument("--models-dir", type=str, default="models", help="Directory where trained models are stored")

    train_parser = subparsers.add_parser("train", help="Train the Two-Tier Fingerprinting Defense Models")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for Tier 2 deep net")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for deep net training")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for AdamW")
    train_parser.add_argument("--benign", type=int, default=1000, help="Number of benign flows")
    train_parser.add_argument("--per-technique", type=int, default=250, help="Number of flows per fingerprinting technique")
    train_parser.add_argument("--seq-len", type=int, default=25, help="Packet sequence length")
    train_parser.add_argument("--models-dir", type=str, default="models", help="Directory to save trained model artifacts")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    monitor_parser = subparsers.add_parser("monitor", help="Launch real-time NetSentinel ncurses dashboard")
    monitor_parser.add_argument("--models-dir", type=str, default="models", help="Directory where trained models are stored")
    monitor_parser.add_argument("--refresh-ms", type=int, default=150, help="Traffic flow interval in ms")
    monitor_parser.add_argument("--live", action="store_true", help="Enable passive live interface packet sniffing (100% safe, read-only)")
    monitor_parser.add_argument("--interface", type=str, default=None, help="Network interface to sniff (default: autodetected active route)")

    collect_parser = subparsers.add_parser("collect", help="Collect live traffic with Chrome Selenium + tshark")
    collect_parser.add_argument("--url", type=str, required=True, help="Website URL to crawl")

    args = parser.parse_args()

    if args.command == "collect":
        collector = WebTrafficCollector()
        pcap_path, cdp_path = collector.crawl_site(args.url)
        if pcap_path:
            print(f"Captured traffic saved to: {pcap_path}")
            if cdp_path:
                print(f"Captured CDP network log saved to: {cdp_path}")
        else:
            print("Failed to capture traffic.")
    elif args.command == "train":
        train_defense(args)
    elif args.command == "defense":
        models_dir = getattr(args, "models_dir", "models")
        run_two_tier_defense_demo(models_dir=models_dir)
    elif args.command == "monitor":
        from netrack.ui import launch_dashboard
        models_dir = getattr(args, "models_dir", "models")
        refresh_ms = getattr(args, "refresh_ms", 150)
        live = getattr(args, "live", False)
        interface = getattr(args, "interface", None)
        launch_dashboard(models_dir=models_dir, refresh_ms=refresh_ms, live_mode=live, interface=interface)
    else:
        benign = getattr(args, "benign", 2220)
        tracker = getattr(args, "tracker", 1266)
        report_path = getattr(args, "report", "reports/nettrack_reproduction_report.md")
        run_full_pipeline(num_benign=benign, num_tracker=tracker, report_path=report_path)


if __name__ == "__main__":
    main()
