"""
Algorithm 1 Feature Extractor for Net-track (Lee et al., WWW 2023).

Extracts:
1. STAT (62 dimensions): 18 statistical metrics per direction (U, D, A) + 8 flow-level metrics
2. BOX (122 raw dimensions -> 20 PCA components): 61 length bins for U and D
3. SEQ (15 dimensions): First 15 signed packet lengths
Final feature vector V: 97 dimensions (STAT + BOX_PCA + SEQ)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA


class FeatureExtractor:
    def __init__(self, pca_components: int = 20):
        self.pca_components = pca_components
        self.pca_model: Optional[PCA] = None

    @staticmethod
    def _compute_list_stats(arr: np.ndarray) -> np.ndarray:
        res = np.zeros(18, dtype=np.float64)
        count = len(arr)
        res[17] = float(count)

        if count == 0:
            return res

        res[0] = np.min(arr)
        res[1] = np.max(arr)
        res[2] = np.mean(arr)

        med = np.median(arr)
        res[3] = np.median(np.abs(arr - med))
        res[4] = np.std(arr)
        res[5] = np.var(arr)

        if count >= 3 and res[4] > 1e-9:
            res[6] = float(stats.skew(arr))
            res[7] = float(stats.kurtosis(arr))
            if np.isnan(res[6]):
                res[6] = 0.0
            if np.isnan(res[7]):
                res[7] = 0.0
        else:
            res[6] = 0.0
            res[7] = 0.0

        percentiles = np.percentile(arr, np.arange(10, 100, 10))
        res[8:17] = percentiles
        return res

    @staticmethod
    def _compute_entropy(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return 0.0
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / np.sum(counts)
        return float(stats.entropy(probs, base=2))

    def extract_stat_features(
        self, U: np.ndarray, D: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        stats_U = self._compute_list_stats(U)
        stats_D = self._compute_list_stats(D)
        stats_A = self._compute_list_stats(A)

        sum_U = float(np.sum(U)) if len(U) > 0 else 0.0
        sum_D = float(np.sum(D)) if len(D) > 0 else 0.0
        sum_A = float(np.sum(A)) if len(A) > 0 else 0.0

        ent_U = self._compute_entropy(U)
        ent_D = self._compute_entropy(D)
        ent_A = self._compute_entropy(A)

        cnt_U, cnt_D = float(len(U)), float(len(D))

        len_ratio = sum_U / (sum_D + 1e-9) if sum_D > 0 else (sum_U if sum_U > 0 else 0.0)
        cnt_ratio = cnt_U / (cnt_D + 1e-9) if cnt_D > 0 else (cnt_U if cnt_U > 0 else 0.0)

        flow_level = np.array(
            [sum_U, sum_D, sum_A, ent_U, ent_D, ent_A, len_ratio, cnt_ratio],
            dtype=np.float64,
        )

        return np.concatenate([stats_U, stats_D, stats_A, flow_level])

    @staticmethod
    def extract_raw_box_features(U: np.ndarray, D: np.ndarray) -> np.ndarray:
        bins = np.arange(0, 1525, 25)

        def _bin_lengths(arr: np.ndarray) -> np.ndarray:
            counts = np.zeros(61, dtype=np.float64)
            if len(arr) == 0:
                return counts
            in_range = arr[arr <= 1500]
            out_range = arr[arr > 1500]
            if len(in_range) > 0:
                hist, _ = np.histogram(in_range, bins=bins)
                counts[:60] = hist
            counts[60] = float(len(out_range))
            return counts

        return np.concatenate([_bin_lengths(U), _bin_lengths(D)])

    @staticmethod
    def extract_seq_features(S: np.ndarray, max_seq_len: int = 15) -> np.ndarray:
        seq = np.zeros(max_seq_len, dtype=np.float64)
        n = min(len(S), max_seq_len)
        if n > 0:
            seq[:n] = S[:n]
        return seq

    def process_flow(
        self,
        packets: List[Tuple[int, int]],
        max_seq_len: int = 15,
        max_packets: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if max_packets is not None and max_packets > 0:
            packets = packets[:max_packets]

        if not packets:
            U = np.array([], dtype=np.float64)
            D = np.array([], dtype=np.float64)
            A = np.array([], dtype=np.float64)
            S = np.array([], dtype=np.float64)
        else:
            directions = np.array([p[0] for p in packets], dtype=np.float64)
            lengths = np.array([p[1] for p in packets], dtype=np.float64)

            U = lengths[directions > 0]
            D = lengths[directions < 0]
            A = lengths
            S = directions * lengths

        return {
            "STAT": self.extract_stat_features(U, D, A),
            "BOX_RAW": self.extract_raw_box_features(U, D),
            "SEQ": self.extract_seq_features(S, max_seq_len=max_seq_len),
            "S": S,
        }

    def fit_pca(self, box_raw_matrix: np.ndarray) -> PCA:
        n_samples, n_features = box_raw_matrix.shape
        n_comp = min(self.pca_components, n_samples, n_features)
        if n_comp <= 0:
            n_comp = 1
        self.pca_model = PCA(n_components=n_comp, random_state=42)
        self.pca_model.fit(box_raw_matrix)
        return self.pca_model

    def transform_box_pca(self, box_raw_matrix: np.ndarray) -> np.ndarray:
        if self.pca_model is None:
            raise ValueError("PCA model has not been fitted yet. Call fit_pca first.")
        transformed = self.pca_model.transform(box_raw_matrix)
        if transformed.shape[1] < self.pca_components:
            padding = np.zeros((transformed.shape[0], self.pca_components - transformed.shape[1]), dtype=np.float64)
            transformed = np.hstack([transformed, padding])
        return transformed

    def extract_dataset_features(
        self,
        flows: List[List[Tuple[int, int]]],
        fit_pca_on_data: bool = False,
        pca_components: int = 20,
        top_k_stat: Optional[int] = None,
        stat_indices: Optional[np.ndarray] = None,
        max_seq_len: int = 15,
        max_packets: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        stat_list, box_raw_list, seq_list = [], [], []

        for flow in flows:
            feat_dict = self.process_flow(flow, max_seq_len=max_seq_len, max_packets=max_packets)
            stat_list.append(feat_dict["STAT"])
            box_raw_list.append(feat_dict["BOX_RAW"])
            seq_list.append(feat_dict["SEQ"])

        stat_matrix = np.array(stat_list, dtype=np.float64)
        box_raw_matrix = np.array(box_raw_list, dtype=np.float64)
        seq_matrix = np.array(seq_list, dtype=np.float64)

        if top_k_stat is not None and stat_indices is not None:
            stat_matrix = stat_matrix[:, stat_indices[:top_k_stat]]

        if fit_pca_on_data:
            self.pca_components = pca_components
            self.fit_pca(box_raw_matrix)

        if self.pca_model is not None:
            box_pca_matrix = self.transform_box_pca(box_raw_matrix)
        else:
            box_pca_matrix = box_raw_matrix[:, :pca_components]

        X = np.hstack([stat_matrix, box_pca_matrix, seq_matrix])
        return X, {
            "STAT": stat_matrix,
            "BOX_RAW": box_raw_matrix,
            "BOX_PCA": box_pca_matrix,
            "SEQ": seq_matrix,
        }
