"""
Two-Tier Complementary Defense System for Browser Fingerprinting Detection.

Combines:
- Tier 1: Fast Line-Rate Random Forest (97-dim Algorithm 1 Features, <0.1 ms inference)
- Tier 2: Hybrid Dual-Branch Neural Network with Self-Attention (MLP on 62 STAT + 1D-CNN on Sequence)
  Provides:
  1. Packet-Level Attention Heatmaps (pinpoints the exact packet containing the fingerprint payload)
  2. Multi-Class Technique Attribution (Canvas, AudioContext, Canvas-Font, WebRTC)
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from netrack.model import NetTrackRandomForest
from netrack.features import FeatureExtractor
from netrack.dataset import PrincetonTSVLoader, TrackerRadarLabeler, BenchmarkFlowGenerator

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class HybridAttentionFingerprintNet(nn.Module):
        """Dual-Branch Neural Network with Self-Attention.

        - Branch A: Processes 62 STAT features via a multi-layer perceptron.
        - Branch B: Processes the raw signed packet sequence via 1D convolutions and
                    computes attention weights across packets to pinpoint the fingerprint burst.
        - Fusion: Merges tabular context and sequence embeddings into dual output heads.
        """

        def __init__(
            self,
            stat_dim: int = 62,
            seq_len: int = 25,
            num_techniques: int = 5,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.num_techniques = num_techniques

            # Branch A: Tabular Statistical Embeddings
            self.tabular_net = nn.Sequential(
                nn.Linear(stat_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
            )

            # Branch B: Sequence Feature Convolutions (1D-CNN)
            self.seq_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

            # Self-Attention Pooling Layer
            self.attention_query = nn.Linear(64, 1)

            # Fusion Layer
            self.fusion = nn.Sequential(
                nn.Linear(32 + 64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.25),
            )

            # Output Heads
            self.binary_head = nn.Linear(64, 1)  # 0: Benign, 1: Fingerprinting
            self.technique_head = nn.Linear(64, num_techniques)  # 0: Benign, 1: Canvas, 2: Audio, 3: Font, 4: WebRTC

        def forward(
            self, stat_x: torch.Tensor, seq_x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                stat_x: (Batch, 62) tabular tensor
                seq_x: (Batch, seq_len) signed packet length tensor

            Returns:
                binary_logits: (Batch, 1)
                technique_logits: (Batch, 5)
                attention_weights: (Batch, seq_len)
            """
            batch_size = stat_x.size(0)

            # Tabular branch
            tab_feat = self.tabular_net(stat_x)  # (Batch, 32)

            # Sequence branch
            seq_in = seq_x.unsqueeze(1)  # (Batch, 1, seq_len)
            conv_feat = self.seq_conv(seq_in)  # (Batch, 64, seq_len)

            # Attention weights over packet sequence
            # (Batch, seq_len, 64) -> (Batch, seq_len, 1) -> (Batch, seq_len)
            conv_transposed = conv_feat.transpose(1, 2)
            attn_scores = self.attention_query(conv_transposed).squeeze(-1)  # (Batch, seq_len)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (Batch, seq_len)

            # Weighted context pooling: (Batch, 64, seq_len) @ (Batch, seq_len, 1) -> (Batch, 64)
            context = torch.bmm(conv_feat, attn_weights.unsqueeze(-1)).squeeze(-1)

            # Concatenation and fusion
            merged = torch.cat([tab_feat, context], dim=1)  # (Batch, 96)
            fused = self.fusion(merged)  # (Batch, 64)

            binary_logits = self.binary_head(fused)
            technique_logits = self.technique_head(fused)

            return binary_logits, technique_logits, attn_weights
else:
    class HybridAttentionFingerprintNet:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to initialize HybridAttentionFingerprintNet.")


class TwoTierFingerprintDefense:
    """Two-Tier Complementary Architecture:

    - Tier 1 (Random Forest): Evaluates flows at line rate (< 0.1 ms per trace).
    - Tier 2 (Deep Learning): Triggered for suspicious/fingerprinting flows to produce
      exact packet-level attention heatmaps and attribute the specific fingerprinting vector.
    """

    TECHNIQUES = [
        "Benign",
        "Canvas Fingerprinting",
        "AudioContext Fingerprinting",
        "Canvas-Font Fingerprinting",
        "WebRTC Local IP Discovery",
    ]

    def __init__(
        self,
        rf_model: Optional[NetTrackRandomForest] = None,
        dl_model: Optional[Any] = None,
        stat_mean: Optional[np.ndarray] = None,
        stat_scale: Optional[np.ndarray] = None,
        seq_len: int = 25,
        suspicious_threshold: float = 0.30,
        random_state: int = 42,
        models_dir: str = "models",
    ):
        self.seq_len = seq_len
        self.suspicious_threshold = suspicious_threshold
        self.extractor = FeatureExtractor(pca_components=20)
        self.princeton_loader = PrincetonTSVLoader()
        self.tracker_radar = TrackerRadarLabeler()
        self.models_dir = models_dir
        self.stat_mean = stat_mean
        self.stat_scale = stat_scale

        # Auto-load Tier 1 if available
        if rf_model is not None:
            self.rf_model = rf_model
        else:
            rf_path = os.path.join(models_dir, "tier1_rf.joblib")
            if os.path.exists(rf_path):
                try:
                    loaded = joblib.load(rf_path)
                    if isinstance(loaded, dict) and "model" in loaded:
                        self.rf_model = loaded["model"]
                        if loaded.get("pca_model") is not None:
                            self.extractor.pca_model = loaded["pca_model"]
                    else:
                        self.rf_model = loaded
                    logger.info(f"Loaded Tier 1 Random Forest from {rf_path}")
                except Exception as e:
                    logger.warning(f"Could not load {rf_path}: {e}")
                    self.rf_model = NetTrackRandomForest(n_estimators=55, random_state=random_state)
            else:
                self.rf_model = NetTrackRandomForest(n_estimators=55, random_state=random_state)

        # Auto-load Tier 2 if available
        if dl_model is not None:
            self.dl_model = dl_model
        else:
            dl_path = os.path.join(models_dir, "tier2_hybrid_attention.pt")
            if TORCH_AVAILABLE and os.path.exists(dl_path):
                try:
                    checkpoint = torch.load(dl_path, map_location="cpu", weights_only=False)
                    net = HybridAttentionFingerprintNet(
                        stat_dim=checkpoint.get("stat_dim", 62),
                        seq_len=checkpoint.get("seq_len", self.seq_len),
                        num_techniques=checkpoint.get("num_techniques", 5),
                    )
                    net.load_state_dict(checkpoint["model_state"])
                    net.eval()
                    self.dl_model = net
                    if self.stat_mean is None:
                        self.stat_mean = checkpoint.get("stat_mean")
                    if self.stat_scale is None:
                        self.stat_scale = checkpoint.get("stat_scale")
                    logger.info(f"Loaded Tier 2 Hybrid Attention Net from {dl_path}")
                except Exception as e:
                    logger.warning(f"Could not load {dl_path}: {e}")
                    self.dl_model = None
            else:
                self.dl_model = None

    def analyze_flow(
        self,
        packets: List[Tuple[int, int]],
        request_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs the two-tier defense on a single flow trace.

        Args:
            packets: List of (direction, length) tuples
            request_urls: Optional list of URLs mapped to this connection

        Returns:
            Comprehensive diagnostic dict including Tier 1 decision, Tier 2 attention heatmap,
            and technique attribution.
        """
        request_urls = request_urls or []
        start_t = time.perf_counter()

        # Step 1: Feature Extraction
        feat_dict = self.extractor.process_flow(packets, max_seq_len=self.seq_len)
        stat_feat = feat_dict["STAT"]

        # Synthetic/dummy PCA transform for 97-dim RF evaluation if PCA model is fitted
        if self.extractor.pca_model is not None:
            box_pca = self.extractor.transform_box_pca(feat_dict["BOX_RAW"].reshape(1, -1))[0]
        else:
            box_pca = feat_dict["BOX_RAW"][:20]

        X_rf = np.hstack([stat_feat, box_pca, feat_dict["SEQ"][:15]]).reshape(1, -1)

        # Step 2: Tier 1 - Random Forest Fast Evaluation
        t1_start = time.perf_counter()
        rf_proba = float(self.rf_model.predict_proba(X_rf)[0, 1]) if hasattr(self.rf_model.model, "classes_") else 0.5
        t1_latency_ms = (time.perf_counter() - t1_start) * 1000.0

        is_suspicious = rf_proba >= self.suspicious_threshold
        tier2_triggered = is_suspicious

        # Ground truth check from Princeton TSVs & Tracker Radar if URLs provided
        gt_tech_id, gt_tech_name = self.princeton_loader.label_flow_technique(request_urls)
        radar_score = max([self.tracker_radar.get_fingerprinting_score(u) for u in request_urls]) if request_urls else 0

        # Step 3: Tier 2 - Deep Learning Attention & Technique Attribution
        tier2_results: Dict[str, Any] = {}
        t2_latency_ms = 0.0

        if tier2_triggered and TORCH_AVAILABLE and self.dl_model is not None:
            t2_start = time.perf_counter()
            with torch.no_grad():
                seq_tensor = torch.tensor(feat_dict["SEQ"], dtype=torch.float32).unsqueeze(0)
                if self.stat_mean is not None and self.stat_scale is not None:
                    norm_stat = (stat_feat - self.stat_mean) / (self.stat_scale + 1e-7)
                    stat_tensor = torch.tensor(norm_stat, dtype=torch.float32).unsqueeze(0)
                else:
                    stat_tensor = torch.tensor(stat_feat, dtype=torch.float32).unsqueeze(0)

                bin_logits, tech_logits, attn_weights = self.dl_model(stat_tensor, seq_tensor)

                dl_proba = float(torch.sigmoid(bin_logits).item())
                tech_probs = F.softmax(tech_logits, dim=-1)[0].numpy()
                pred_tech_idx = int(np.argmax(tech_probs))
                weights = attn_weights[0].numpy().tolist()

            t2_latency_ms = (time.perf_counter() - t2_start) * 1000.0

            # Find the most salient packet that triggered the attention peak
            max_attn_idx = int(np.argmax(weights[:len(packets)])) if len(packets) > 0 else 0
            salient_pkt = packets[max_attn_idx] if max_attn_idx < len(packets) else (0, 0)

            tier2_results = {
                "dl_confidence": dl_proba,
                "predicted_technique": self.TECHNIQUES[pred_tech_idx],
                "technique_distribution": {
                    self.TECHNIQUES[i]: float(tech_probs[i]) for i in range(len(self.TECHNIQUES))
                },
                "packet_attention_weights": weights[:min(len(packets), self.seq_len)],
                "most_salient_packet": {
                    "packet_index": max_attn_idx + 1,
                    "direction": "Uplink (Client -> Server)" if salient_pkt[0] > 0 else "Downlink (Server -> Client)",
                    "length_bytes": salient_pkt[1],
                    "attention_score": float(weights[max_attn_idx]),
                },
                "t2_latency_ms": t2_latency_ms,
            }
        else:
            # Fallback heuristic / attribution from Princeton loader
            assigned_tech = gt_tech_name if gt_tech_id > 0 else ("Fingerprinting" if rf_proba > 0.5 else "Benign")
            tier2_results = {
                "dl_confidence": rf_proba,
                "predicted_technique": assigned_tech,
                "t2_latency_ms": 0.0,
            }

        total_latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "flow_summary": {
                "packet_count": len(packets),
                "total_uplink_bytes": float(np.sum([p[1] for p in packets if p[0] > 0])),
                "total_downlink_bytes": float(np.sum([p[1] for p in packets if p[0] < 0])),
            },
            "ground_truth": {
                "technique": gt_tech_name,
                "tracker_radar_score": radar_score,
            },
            "tier1_random_forest": {
                "fingerprint_probability": rf_proba,
                "is_suspicious": is_suspicious,
                "latency_ms": t1_latency_ms,
            },
            "tier2_deep_learning": tier2_results,
            "overall_decision": "BLOCKED (Browser Fingerprinting)" if (rf_proba > 0.5 or tier2_results.get("dl_confidence", 0) > 0.5) else "ALLOWED (Benign)",
            "total_latency_ms": total_latency_ms,
        }


class FingerprintDefenseTrainer:
    """End-to-End Trainer for Two-Tier Fingerprinting Defense:
    1. Tier 1: Fast Line-Rate Random Forest (97-dim Algorithm 1 features).
    2. Tier 2: Hybrid Dual-Branch Neural Network with Self-Attention.
    """

    def __init__(
        self,
        models_dir: str = "models",
        random_state: int = 42,
        seq_len: int = 25,
    ):
        self.models_dir = models_dir
        self.random_state = random_state
        self.seq_len = seq_len
        self.extractor = FeatureExtractor(pca_components=20)
        self.princeton_loader = PrincetonTSVLoader()
        self.tracker_radar = TrackerRadarLabeler()
        os.makedirs(self.models_dir, exist_ok=True)

    def prepare_dataset(
        self,
        num_benign: int = 1000,
        num_per_technique: int = 250,
        test_size: float = 0.20,
    ) -> Dict[str, Any]:
        """Generates flows and extracts representations for both models."""
        generator = BenchmarkFlowGenerator(seed=self.random_state)
        flows, y_bin, y_tech, urls_list = generator.generate_fingerprinting_dataset(
            num_benign=num_benign,
            num_per_technique=num_per_technique,
            loader=self.princeton_loader,
        )

        logger.info(f"Extracting Algorithm 1 features for {len(flows)} flows...")
        X_rf, components = self.extractor.extract_dataset_features(
            flows, fit_pca_on_data=True, pca_components=20, max_seq_len=15
        )
        stat_feats = components["STAT"]
        seq_feats = np.array(
            [self.extractor.process_flow(f, max_seq_len=self.seq_len)["SEQ"] for f in flows],
            dtype=np.float32,
        )

        indices = np.arange(len(flows))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=y_tech, random_state=self.random_state
        )

        return {
            "flows": flows,
            "urls_list": urls_list,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "X_rf": X_rf,
            "stat_feats": stat_feats,
            "seq_feats": seq_feats,
            "y_bin": np.array(y_bin, dtype=np.int64),
            "y_tech": np.array(y_tech, dtype=np.int64),
        }

    def train_tier1_rf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[NetTrackRandomForest, Dict[str, Any]]:
        """Trains and evaluates Tier 1 Random Forest."""
        logger.info("Training Tier 1 Random Forest (N=55, Gini criterion, sqrt features)...")
        rf = NetTrackRandomForest(n_estimators=55, random_state=self.random_state)
        train_time = rf.fit(X_train, y_train)
        metrics = rf.evaluate_with_inference_time(X_test, y_test)
        metrics["train_time_sec"] = train_time

        # Save model and fitted PCA transformer
        rf_path = os.path.join(self.models_dir, "tier1_rf.joblib")
        joblib.dump({"model": rf, "pca_model": self.extractor.pca_model}, rf_path)
        logger.info(f"Tier 1 Random Forest saved to: {rf_path}")
        return rf, metrics

    def train_tier2_deep(
        self,
        stat_train: np.ndarray,
        seq_train: np.ndarray,
        y_bin_train: np.ndarray,
        y_tech_train: np.ndarray,
        stat_test: np.ndarray,
        seq_test: np.ndarray,
        y_bin_test: np.ndarray,
        y_tech_test: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Trains and evaluates Tier 2 Dual-Branch Neural Network with Self-Attention."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed. Skipping Tier 2 deep model training.")
            return None, {"error": "PyTorch not available"}

        logger.info(
            f"Training Tier 2 Hybrid Attention Network ({epochs} epochs, batch size {batch_size}, lr={lr})..."
        )
        stat_mean = np.mean(stat_train, axis=0)
        stat_scale = np.std(stat_train, axis=0) + 1e-7

        stat_train_norm = (stat_train - stat_mean) / stat_scale
        stat_test_norm = (stat_test - stat_mean) / stat_scale

        train_ds = TensorDataset(
            torch.tensor(stat_train_norm, dtype=torch.float32),
            torch.tensor(seq_train, dtype=torch.float32),
            torch.tensor(y_bin_train, dtype=torch.float32),
            torch.tensor(y_tech_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = HybridAttentionFingerprintNet(
            stat_dim=62, seq_len=self.seq_len, num_techniques=5
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion_bin = nn.BCEWithLogitsLoss()
        criterion_tech = nn.CrossEntropyLoss()

        history = []
        start_train = time.perf_counter()

        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for bx_stat, bx_seq, by_bin, by_tech in train_loader:
                optimizer.zero_grad()
                bin_logits, tech_logits, _ = model(bx_stat, bx_seq)
                loss_b = criterion_bin(bin_logits.squeeze(1), by_bin)
                loss_t = criterion_tech(tech_logits, by_tech)
                loss = loss_b + 0.5 * loss_t
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(by_bin)

            epoch_loss = total_loss / len(train_ds)
            history.append(epoch_loss)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch:2d}/{epochs:2d} - Loss: {epoch_loss:.4f}")

        total_train_sec = time.perf_counter() - start_train

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            t_stat = torch.tensor(stat_test_norm, dtype=torch.float32)
            t_seq = torch.tensor(seq_test, dtype=torch.float32)

            t0 = time.perf_counter()
            bin_logits, tech_logits, attn_weights = model(t_stat, t_seq)
            inference_sec = time.perf_counter() - t0

            per_trace_ms = (inference_sec / len(stat_test)) * 1000.0
            bin_probs = torch.sigmoid(bin_logits).squeeze(1).numpy()
            bin_preds = (bin_probs >= 0.5).astype(int)

            tech_probs = F.softmax(tech_logits, dim=-1).numpy()
            tech_preds = np.argmax(tech_probs, axis=1)

        bin_acc = float(accuracy_score(y_bin_test, bin_preds))
        bin_prec = float(precision_score(y_bin_test, bin_preds, zero_division=0))
        bin_rec = float(recall_score(y_bin_test, bin_preds, zero_division=0))
        bin_f1 = float(f1_score(y_bin_test, bin_preds, zero_division=0))
        bin_auc = float(roc_auc_score(y_bin_test, bin_probs))
        tech_acc = float(accuracy_score(y_tech_test, tech_preds))

        # Per-technique accuracy
        per_tech_acc = {}
        for idx, tech_name in enumerate(TwoTierFingerprintDefense.TECHNIQUES):
            mask = (y_tech_test == idx)
            if np.sum(mask) > 0:
                per_tech_acc[tech_name] = float(accuracy_score(y_tech_test[mask], tech_preds[mask]))

        dl_metrics = {
            "binary_accuracy": bin_acc,
            "binary_precision": bin_prec,
            "binary_recall": bin_rec,
            "binary_f1": bin_f1,
            "binary_auc": bin_auc,
            "technique_attribution_accuracy": tech_acc,
            "per_technique_accuracy": per_tech_acc,
            "per_trace_inference_ms": per_trace_ms,
            "train_time_sec": total_train_sec,
            "loss_history": history,
        }

        # Save checkpoint
        dl_path = os.path.join(self.models_dir, "tier2_hybrid_attention.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "stat_mean": stat_mean,
                "stat_scale": stat_scale,
                "seq_len": self.seq_len,
                "stat_dim": 62,
                "num_techniques": 5,
                "metrics": dl_metrics,
            },
            dl_path,
        )
        logger.info(f"Tier 2 Hybrid Attention Network saved to: {dl_path}")

        return model, dl_metrics

    def train_all(
        self,
        num_benign: int = 1000,
        num_per_technique: int = 250,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> Dict[str, Any]:
        """Runs the complete training workflow across both tiers."""
        data = self.prepare_dataset(num_benign=num_benign, num_per_technique=num_per_technique)
        train_idx, test_idx = data["train_idx"], data["test_idx"]

        # Train Tier 1 RF
        rf_model, rf_metrics = self.train_tier1_rf(
            data["X_rf"][train_idx],
            data["y_bin"][train_idx],
            data["X_rf"][test_idx],
            data["y_bin"][test_idx],
        )

        # Train Tier 2 Deep Network
        dl_model, dl_metrics = self.train_tier2_deep(
            stat_train=data["stat_feats"][train_idx],
            seq_train=data["seq_feats"][train_idx],
            y_bin_train=data["y_bin"][train_idx],
            y_tech_train=data["y_tech"][train_idx],
            stat_test=data["stat_feats"][test_idx],
            seq_test=data["seq_feats"][test_idx],
            y_bin_test=data["y_bin"][test_idx],
            y_tech_test=data["y_tech"][test_idx],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        # Create deployed two-tier defense instance
        defense = TwoTierFingerprintDefense(
            rf_model=rf_model,
            dl_model=dl_model,
            models_dir=self.models_dir,
            seq_len=self.seq_len,
            random_state=self.random_state,
        )

        return {
            "tier1_rf": rf_metrics,
            "tier2_deep": dl_metrics,
            "dataset_summary": {
                "total_flows": len(data["flows"]),
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "num_benign": num_benign,
                "num_per_technique": num_per_technique,
            },
            "defense_instance": defense,
        }
