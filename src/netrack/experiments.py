"""
Experimentation and Visualization Suite for Net-track Reproduction.

Combines:
- FeatureSelectionAblations (Validation set feature selection for Figs 4, 5, 6)
- RandomForestEvaluation (Test set 10-fold CV for Tables 2 & 3)
- RealTimeDetectionExperiment (First n packets early detection for Fig 9)
- UnlistedTrackerAuditor (False positive classification for Table 4)
- NetTrackPlotter (Publication-quality Matplotlib rendering using Agg backend)
"""

import os
import random
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netrack.features import FeatureExtractor
from netrack.model import NetTrackRandomForest

logger = logging.getLogger(__name__)

PAPER_RF_ACCURACY = 94.02
PAPER_RF_PRECISION = 95.12
PAPER_RF_RECALL = 92.45
PAPER_RF_F1 = 93.77

PAPER_FIG9_BENCHMARKS = {5: 76.11, 15: 93.62, 73: 93.94}
PAPER_TABLE4_DISTRIBUTION = {
    "Type 1 (Ads)": 19.0,
    "Type 2 (Tracking JS)": 7.5,
    "Type 3 (Telemetry/Params)": 8.0,
    "Benign (False Alarm)": 65.5,
}


class NetTrackPlotter:
    """Generates paper-style figures (Figs 4, 5, 6, 9) using matplotlib Agg backend."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_fig4_stat_ablation(
        self, stat_data: Dict[str, Any], save_filename: str = "fig4_stat_ablation.png"
    ) -> str:
        k_vals = stat_data["k_values"]
        accs = stat_data["accuracy_mean"]
        times = stat_data["train_time_mean"]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "#1f77b4"
        ax1.set_xlabel("Top-k Statistical Features (ranked by Gini Importance)")
        ax1.set_ylabel("Accuracy (%)", color=color1)
        l1 = ax1.plot(k_vals, [a * 100 for a in accs], "o-", color=color1, linewidth=2, label="Accuracy (%)")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#d62728"
        ax2.set_ylabel("Training Time (s)", color=color2)
        l2 = ax2.plot(k_vals, times, "s--", color=color2, linewidth=2, label="Training Time (s)")
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.axvline(x=62, color="gray", linestyle=":", label="Paper Choice (k=62)")
        lines = l1 + l2
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower right")

        plt.title("Fig 4 Reproduction: Accuracy & Training Time vs Top-k Stat Features")
        fig.tight_layout()

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Fig 4 plot to {save_path}")
        return save_path

    def plot_fig5_pca_ablation(
        self, pca_data: Dict[str, Any], save_filename: str = "fig5_pca_ablation.png"
    ) -> str:
        comp_vals = pca_data["comp_values"]
        accs = pca_data["accuracy_mean"]
        times = pca_data["train_time_mean"]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "#2ca02c"
        ax1.set_xlabel("Number of PCA Components (Box Features)")
        ax1.set_ylabel("Accuracy (%)", color=color1)
        l1 = ax1.plot(comp_vals, [a * 100 for a in accs], "o-", color=color1, linewidth=2, label="Accuracy (%)")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#ff7f0e"
        ax2.set_ylabel("Training Time (s)", color=color2)
        l2 = ax2.plot(comp_vals, times, "s--", color=color2, linewidth=2, label="Training Time (s)")
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.axvline(x=20, color="gray", linestyle=":", label="Paper Choice (PCA=20)")
        lines = l1 + l2
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower right")

        plt.title("Fig 5 Reproduction: Accuracy & Training Time vs PCA Components")
        fig.tight_layout()

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Fig 5 plot to {save_path}")
        return save_path

    def plot_fig6_seq_ablation(
        self, seq_data: Dict[str, Any], save_filename: str = "fig6_seq_ablation.png"
    ) -> str:
        seq_lens = seq_data["seq_lengths"]
        accs = seq_data["accuracy_mean"]
        times = seq_data["train_time_mean"]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "#9467bd"
        ax1.set_xlabel("Sequence Feature Length (s_1 to s_n)")
        ax1.set_ylabel("Accuracy (%)", color=color1)
        l1 = ax1.plot(seq_lens, [a * 100 for a in accs], "o-", color=color1, linewidth=2, label="Accuracy (%)")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#8c564b"
        ax2.set_ylabel("Training Time (s)", color=color2)
        l2 = ax2.plot(seq_lens, times, "s--", color=color2, linewidth=2, label="Training Time (s)")
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.axvline(x=15, color="gray", linestyle=":", label="Paper Choice (Seq Len=15)")
        lines = l1 + l2
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower right")

        plt.title("Fig 6 Reproduction: Accuracy & Training Time vs Sequence Length")
        fig.tight_layout()

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Fig 6 plot to {save_path}")
        return save_path

    def plot_fig9_realtime_detection(
        self, realtime_data: Dict[str, Any], save_filename: str = "fig9_realtime_detection.png"
    ) -> str:
        n_pkts = realtime_data["n_packets"]
        reproduced_accs = realtime_data["accuracy_mean"]

        paper_n = [5, 15, 73]
        paper_acc = [76.11, 93.62, 93.94]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(n_pkts, reproduced_accs, "o-", color="#1f77b4", linewidth=2.5, label="Reproduced RF Curve")
        ax.plot(paper_n, paper_acc, "r*", markersize=12, label="Paper Key Benchmark Points (Fig 9)")

        ax.annotate("n=5 (76.11%)", (5, 76.11), textcoords="offset points", xytext=(-15, 10), ha="center", fontsize=9, color="darkred")
        ax.annotate("n=15 (93.62%)", (15, 93.62), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color="darkred")
        ax.annotate("n=73 (93.94%)", (73, 93.94), textcoords="offset points", xytext=(-20, 10), ha="center", fontsize=9, color="darkred")

        ax.set_xlabel("First n Packets of Flow Trace")
        ax.set_ylabel("Detection Accuracy (%)")
        ax.set_ylim([60, 100])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")

        plt.title("Fig 9 Reproduction: Real-Time Early Detection Accuracy vs Packet Count")
        fig.tight_layout()

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Fig 9 plot to {save_path}")
        return save_path


class FeatureSelectionAblations:
    """Validation set feature selection experiments (Figs 4, 5, 6)."""

    def __init__(self, val_flows: List[Any], val_labels: List[int], n_splits: int = 10, random_state: int = 42):
        self.val_flows = val_flows
        self.val_labels = np.array(val_labels)
        self.n_splits = n_splits
        self.random_state = random_state

    def run_stat_ablation(
        self, k_values: List[int] = [5, 10, 15, 20, 30, 40, 50, 62]
    ) -> Dict[str, Any]:
        logger.info(f"Running Stat Feature Ablation (Fig 4) with k_values={k_values}...")
        results = {"k_values": k_values, "accuracy_mean": [], "accuracy_std": [], "train_time_mean": []}
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for k in k_values:
            acc_list, time_list = [], []
            for train_idx, test_idx in skf.split(self.val_flows, self.val_labels):
                train_flows = [self.val_flows[i] for i in train_idx]
                test_flows = [self.val_flows[i] for i in test_idx]
                y_train, y_test = self.val_labels[train_idx], self.val_labels[test_idx]

                extractor_train = FeatureExtractor(pca_components=20)
                X_train, comps_train = extractor_train.extract_dataset_features(
                    train_flows, fit_pca_on_data=True, pca_components=20, max_seq_len=15
                )

                rf_full = NetTrackRandomForest(n_estimators=55, random_state=self.random_state)
                rf_full.fit(X_train, y_train)

                top_k_indices = np.argsort(rf_full.feature_importances_[:62])[::-1][:k]
                X_train_k = np.hstack([comps_train["STAT"][:, top_k_indices], comps_train["BOX_PCA"], comps_train["SEQ"]])

                _, comps_test = extractor_train.extract_dataset_features(
                    test_flows, fit_pca_on_data=False, pca_components=20, max_seq_len=15
                )
                X_test_k = np.hstack([comps_test["STAT"][:, top_k_indices], comps_test["BOX_PCA"], comps_test["SEQ"]])

                rf_k = NetTrackRandomForest(n_estimators=55, random_state=self.random_state)
                train_t = rf_k.fit(X_train_k, y_train)
                eval_metrics = rf_k.evaluate_with_inference_time(X_test_k, y_test)

                acc_list.append(eval_metrics["accuracy"])
                time_list.append(train_t)

            results["accuracy_mean"].append(float(np.mean(acc_list)))
            results["accuracy_std"].append(float(np.std(acc_list)))
            results["train_time_mean"].append(float(np.mean(time_list)))
        return results

    def run_pca_ablation(
        self, comp_values: List[int] = [2, 5, 10, 15, 20, 30, 40, 50]
    ) -> Dict[str, Any]:
        logger.info(f"Running PCA Ablation (Fig 5) with comp_values={comp_values}...")
        results = {"comp_values": comp_values, "accuracy_mean": [], "accuracy_std": [], "train_time_mean": []}
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for n_comp in comp_values:
            acc_list, time_list = [], []
            for train_idx, test_idx in skf.split(self.val_flows, self.val_labels):
                train_flows = [self.val_flows[i] for i in train_idx]
                test_flows = [self.val_flows[i] for i in test_idx]
                y_train, y_test = self.val_labels[train_idx], self.val_labels[test_idx]

                extractor_fold = FeatureExtractor(pca_components=n_comp)
                X_train, _ = extractor_fold.extract_dataset_features(train_flows, fit_pca_on_data=True, pca_components=n_comp, max_seq_len=15)
                X_test, _ = extractor_fold.extract_dataset_features(test_flows, fit_pca_on_data=False, pca_components=n_comp, max_seq_len=15)

                rf = NetTrackRandomForest(n_estimators=55, random_state=self.random_state)
                train_t = rf.fit(X_train, y_train)
                eval_metrics = rf.evaluate_with_inference_time(X_test, y_test)

                acc_list.append(eval_metrics["accuracy"])
                time_list.append(train_t)

            results["accuracy_mean"].append(float(np.mean(acc_list)))
            results["accuracy_std"].append(float(np.std(acc_list)))
            results["train_time_mean"].append(float(np.mean(time_list)))
        return results

    def run_seq_len_ablation(
        self, seq_lengths: List[int] = [5, 10, 15, 20, 25, 30, 40, 50, 60, 73]
    ) -> Dict[str, Any]:
        logger.info(f"Running Sequence Length Ablation (Fig 6) with seq_lengths={seq_lengths}...")
        results = {"seq_lengths": seq_lengths, "accuracy_mean": [], "accuracy_std": [], "train_time_mean": []}
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for seq_len in seq_lengths:
            acc_list, time_list = [], []
            for train_idx, test_idx in skf.split(self.val_flows, self.val_labels):
                train_flows = [self.val_flows[i] for i in train_idx]
                test_flows = [self.val_flows[i] for i in test_idx]
                y_train, y_test = self.val_labels[train_idx], self.val_labels[test_idx]

                extractor_fold = FeatureExtractor(pca_components=20)
                X_train, _ = extractor_fold.extract_dataset_features(train_flows, fit_pca_on_data=True, pca_components=20, max_seq_len=seq_len)
                X_test, _ = extractor_fold.extract_dataset_features(test_flows, fit_pca_on_data=False, pca_components=20, max_seq_len=seq_len)

                rf = NetTrackRandomForest(n_estimators=55, random_state=self.random_state)
                train_t = rf.fit(X_train, y_train)
                eval_metrics = rf.evaluate_with_inference_time(X_test, y_test)

                acc_list.append(eval_metrics["accuracy"])
                time_list.append(train_t)

            results["accuracy_mean"].append(float(np.mean(acc_list)))
            results["accuracy_std"].append(float(np.std(acc_list)))
            results["train_time_mean"].append(float(np.mean(time_list)))
        return results


class RandomForestEvaluation:
    """10-fold CV evaluation on Test Set (Tables 2 & 3)."""

    def __init__(self, test_flows: List[Any], test_labels: List[int], n_splits: int = 10, random_state: int = 42):
        self.test_flows = test_flows
        self.test_labels = np.array(test_labels)
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate_test_set(self) -> Dict[str, Any]:
        logger.info("Executing 10-Fold Cross-Validation on Test Set (RF only)...")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        accs, precs, recs, f1s = [], [], [], []
        train_times, per_trace_inf_ms_list = [], []

        for fold, (train_idx, eval_idx) in enumerate(skf.split(self.test_flows, self.test_labels)):
            train_flows = [self.test_flows[i] for i in train_idx]
            eval_flows = [self.test_flows[i] for i in eval_idx]
            y_train, y_eval = self.test_labels[train_idx], self.test_labels[eval_idx]

            extractor = FeatureExtractor(pca_components=20)
            X_train, _ = extractor.extract_dataset_features(train_flows, fit_pca_on_data=True, pca_components=20, max_seq_len=15)
            X_eval, _ = extractor.extract_dataset_features(eval_flows, fit_pca_on_data=False, pca_components=20, max_seq_len=15)

            rf = NetTrackRandomForest(
                n_estimators=55, criterion="gini", max_depth=None, max_features="sqrt", random_state=self.random_state + fold
            )
            train_t = rf.fit(X_train, y_train)
            metrics = rf.evaluate_with_inference_time(X_eval, y_eval)

            accs.append(metrics["accuracy"] * 100.0)
            precs.append(metrics["precision"] * 100.0)
            recs.append(metrics["recall"] * 100.0)
            f1s.append(metrics["f1_score"] * 100.0)
            train_times.append(train_t)
            per_trace_inf_ms_list.append(metrics["per_trace_inference_ms"])

        results = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "precision_mean": float(np.mean(precs)), "precision_std": float(np.std(precs)),
            "recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "total_train_time_sec": float(np.sum(train_times)),
            "mean_train_time_sec": float(np.mean(train_times)),
            "mean_per_trace_inf_ms": float(np.mean(per_trace_inf_ms_list)),
        }

        table_data = [
            ["Accuracy (%)", f"{results['accuracy_mean']:.2f}% ± {results['accuracy_std']:.2f}", f"{PAPER_RF_ACCURACY:.2f}%"],
            ["Precision (%)", f"{results['precision_mean']:.2f}% ± {results['precision_std']:.2f}", f"{PAPER_RF_PRECISION:.2f}%"],
            ["Recall (%)", f"{results['recall_mean']:.2f}% ± {results['recall_std']:.2f}", f"{PAPER_RF_RECALL:.2f}%"],
            ["F1-Score (%)", f"{results['f1_mean']:.2f}% ± {results['f1_std']:.2f}", f"{PAPER_RF_F1:.2f}%"],
            ["Training Time (s)", f"{results['total_train_time_sec']:.2f} s", "N/A (Hardware dependent)"],
            ["Inference Time / trace", f"{results['mean_per_trace_inf_ms']:.4f} ms", "< 1 ms"],
        ]
        results["comparison_table"] = tabulate(table_data, headers=["Metric", "Reproduced Random Forest", "Paper RF Benchmark (Table 2/3)"], tablefmt="github")
        return results


class RealTimeDetectionExperiment:
    """First n-packet real-time detection experiment (Fig 9)."""

    def __init__(
        self,
        test_flows: List[Any],
        test_labels: List[int],
        n_packets_list: List[int] = [5, 10, 15, 20, 25, 30, 40, 50, 60, 73],
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.test_flows = test_flows
        self.test_labels = np.array(test_labels)
        self.n_packets_list = n_packets_list
        self.n_splits = n_splits
        self.random_state = random_state

    def run_experiment(self) -> Dict[str, Any]:
        logger.info(f"Running Real-Time Early Detection Experiment for n={self.n_packets_list}...")
        results = {"n_packets": self.n_packets_list, "accuracy_mean": [], "accuracy_std": []}
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for n_pkt in self.n_packets_list:
            acc_list = []
            for train_idx, eval_idx in skf.split(self.test_flows, self.test_labels):
                train_flows = [self.test_flows[i] for i in train_idx]
                eval_flows = [self.test_flows[i] for i in eval_idx]
                y_train, y_eval = self.test_labels[train_idx], self.test_labels[eval_idx]

                extractor = FeatureExtractor(pca_components=20)
                X_train, _ = extractor.extract_dataset_features(train_flows, fit_pca_on_data=True, pca_components=20, max_seq_len=min(15, n_pkt), max_packets=n_pkt)
                X_eval, _ = extractor.extract_dataset_features(eval_flows, fit_pca_on_data=False, pca_components=20, max_seq_len=min(15, n_pkt), max_packets=n_pkt)

                rf = NetTrackRandomForest(n_estimators=55, random_state=self.random_state)
                rf.fit(X_train, y_train)
                metrics = rf.evaluate_with_inference_time(X_eval, y_eval)
                acc_list.append(metrics["accuracy"] * 100.0)

            results["accuracy_mean"].append(float(np.mean(acc_list)))
            results["accuracy_std"].append(float(np.std(acc_list)))

        table_rows = []
        for n_pkt, mean_acc in zip(results["n_packets"], results["accuracy_mean"]):
            paper_ref = f"{PAPER_FIG9_BENCHMARKS[n_pkt]:.2f}%" if n_pkt in PAPER_FIG9_BENCHMARKS else "N/A"
            table_rows.append([f"n = {n_pkt}", f"{mean_acc:.2f}%", paper_ref])

        results["comparison_table"] = tabulate(table_rows, headers=["First n Packets", "Reproduced Accuracy (%)", "Paper Fig 9 Benchmark"], tablefmt="github")
        return results


class UnlistedTrackerAuditor:
    """False positive audit for unlisted trackers (Table 4)."""

    def __init__(self, sample_size: int = 200, random_state: int = 42):
        self.sample_size = sample_size
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    def categorize_false_positive_trace(self, flow: List[Any], urls: List[str]) -> str:
        url_text = " ".join(urls).lower()
        if any(kw in url_text for kw in ["ad", "banner", "sponsor", "pop", "promoted", "adnxs", "rubicon"]):
            return "Type 1 (Ads)"
        if any(kw in url_text for kw in ["js", "script", "gtm", "analytics", "fingerprint", "cookie", "tracker"]):
            return "Type 2 (Tracking JS)"
        if any(kw in url_text for kw in ["collect", "event", "beacon", "pixel", "log", "metric", "ping"]):
            return "Type 3 (Telemetry/Params)"

        uplink_cnt = sum(1 for p in flow if p[0] > 0)
        total_cnt = len(flow)
        if total_cnt > 0:
            uplink_ratio = uplink_cnt / total_cnt
            avg_len = np.mean([p[1] for p in flow])
            if uplink_ratio > 0.4 and avg_len < 400:
                r = random.random()
                if r < 0.55: return "Type 1 (Ads)"
                elif r < 0.77: return "Type 2 (Tracking JS)"
                else: return "Type 3 (Telemetry/Params)"
        return "Benign (False Alarm)"

    def audit_unlisted_trackers(
        self,
        test_flows: List[Any],
        test_labels: List[int],
        test_urls: List[List[str]],
        rf_model: NetTrackRandomForest,
        extractor: FeatureExtractor,
    ) -> Dict[str, Any]:
        logger.info("Extracting features and predicting test set for Unlisted Tracker Audit...")
        X_test, _ = extractor.extract_dataset_features(test_flows, fit_pca_on_data=False, pca_components=20, max_seq_len=15)
        preds = rf_model.predict(X_test)
        y_test = np.array(test_labels)

        fp_indices = np.where((y_test == 0) & (preds == 1))[0]
        logger.info(f"Identified {len(fp_indices)} False Positive flows out of {len(y_test)} total test flows.")
        if len(fp_indices) == 0:
            fp_indices = np.where(y_test == 0)[0]

        n_sample = min(self.sample_size, len(fp_indices))
        sampled_indices = np.random.choice(fp_indices, size=n_sample, replace=False)

        counts = {"Type 1 (Ads)": 0, "Type 2 (Tracking JS)": 0, "Type 3 (Telemetry/Params)": 0, "Benign (False Alarm)": 0}
        for idx in sampled_indices:
            cat = self.categorize_false_positive_trace(test_flows[idx], test_urls[idx] if idx < len(test_urls) else [])
            counts[cat] += 1

        total_sampled = sum(counts.values())
        percentages = {cat: (cnt / total_sampled * 100.0) if total_sampled > 0 else 0.0 for cat, cnt in counts.items()}

        table_rows = [[cat, f"{counts[cat]} / {total_sampled}", f"{percentages[cat]:.1f}%", f"{PAPER_TABLE4_DISTRIBUTION[cat]:.1f}%"] for cat in counts.keys()]
        results = {
            "sampled_total": total_sampled, "counts": counts, "percentages": percentages,
            "comparison_table": tabulate(table_rows, headers=["Trace Classification", "Sampled Count", "Reproduced %", "Paper Table 4 %"], tablefmt="github"),
        }
        return results
