"""
Consolidated Unit Test Suite for Net-track Reproduction Project.
"""

import os
import unittest
import numpy as np
from netrack.features import FeatureExtractor
from netrack.dataset import FilterListLabeler, DatasetManager
from netrack.model import NetTrackRandomForest


class TestNetTrack(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(pca_components=20)
        self.sample_flow = [
            (1, 100), (-1, 1400), (1, 50), (-1, 1200), (1, 200),
            (-1, 1500), (-1, 1600), (1, 80), (-1, 500), (1, 300)
        ]

    def test_feature_extractor_dimensions(self):
        feat = self.extractor.process_flow(self.sample_flow, max_seq_len=15)
        self.assertEqual(len(feat["STAT"]), 62)
        self.assertEqual(len(feat["BOX_RAW"]), 122)
        self.assertEqual(len(feat["SEQ"]), 15)

    def test_empty_flow_handling(self):
        feat = self.extractor.process_flow([], max_seq_len=15)
        self.assertEqual(len(feat["STAT"]), 62)
        self.assertEqual(len(feat["BOX_RAW"]), 122)
        self.assertEqual(len(feat["SEQ"]), 15)
        self.assertTrue(np.all(feat["STAT"] == 0.0))

    def test_dataset_pca_and_concatenation(self):
        flows = [[(1, 100 + i), (-1, 1400 - i * 2), (1, 50 + i * 3)] for i in range(30)]
        X, components = self.extractor.extract_dataset_features(
            flows, fit_pca_on_data=True, pca_components=20, max_seq_len=15
        )
        self.assertEqual(X.shape, (30, 97))
        self.assertEqual(components["BOX_PCA"].shape, (30, 20))

    def test_filter_list_labeler(self):
        labeler = FilterListLabeler(auto_download=False)
        self.assertTrue(labeler.is_tracker_url("https://google-analytics.com/collect?v=1"))
        self.assertFalse(labeler.is_tracker_url("https://example.com/logo.png"))
        self.assertEqual(labeler.label_flow(["https://doubleclick.net/ads"]), 1)
        self.assertEqual(labeler.label_flow(["https://example.com/app.js"]), 0)

    def test_dataset_manager_splits(self):
        mgr = DatasetManager(val_ratio=0.2, random_state=42)
        flows = [[(1, 100)] for _ in range(50)]
        labels = [0] * 30 + [1] * 20
        split = mgr.split_dataset(flows, labels)
        self.assertEqual(len(split["val"]["labels"]), 10)
        self.assertEqual(len(split["test"]["labels"]), 40)

    def test_random_forest_model(self):
        rf = NetTrackRandomForest(n_estimators=10, random_state=42)
        X = np.random.randn(40, 97)
        y = np.random.choice([0, 1], size=40)
        train_time = rf.fit(X, y)
        self.assertGreater(train_time, 0.0)

        preds = rf.predict(X)
        self.assertEqual(len(preds), 40)
        metrics = rf.evaluate_with_inference_time(X, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("per_trace_inference_ms", metrics)

    def test_flow_splitter_directionality(self):
        from scapy.all import IP, TCP
        from netrack.dataset import FlowSplitter

        splitter = FlowSplitter(client_ip_prefix="192.168.")
        client_ip, server_ip = "192.168.1.100", "93.184.216.34"
        client_port, server_port = 54321, 443

        # Simulated handshake: Client SYN -> Server SYN-ACK -> Client ACK
        pkt1 = IP(src=client_ip, dst=server_ip) / TCP(sport=client_port, dport=server_port, flags="S")
        pkt2 = IP(src=server_ip, dst=client_ip) / TCP(sport=server_port, dport=client_port, flags="SA")
        pkt3 = IP(src=client_ip, dst=server_ip) / TCP(sport=client_port, dport=server_port, flags="A")

        flows = splitter.extract_flows_from_packets([pkt1, pkt2, pkt3])
        self.assertEqual(len(flows), 1)
        flow_packets = list(flows.values())[0]

        # First packet is uplink (+1) from client
        self.assertEqual(flow_packets[0][0], 1)
        # Second packet is downlink (-1) from server
        self.assertEqual(flow_packets[1][0], -1)
        # Third packet is uplink (+1) from client
        self.assertEqual(flow_packets[2][0], 1)

    def test_network_trace_aligner(self):
        from netrack.dataset import NetworkTraceAligner, FilterListLabeler

        aligner = NetworkTraceAligner(labeler=FilterListLabeler(auto_download=False))
        flows = {
            ("192.168.1.10", 50000, "142.250.190.46", 443): [(1, 200), (-1, 500)],
            ("192.168.1.10", 50001, "93.184.216.34", 443): [(1, 300), (-1, 1400)],
        }
        cdp_events = [
            {"method": "Network.responseReceived", "response": {"url": "https://google-analytics.com/collect", "remoteIPAddress": "142.250.190.46", "remotePort": 443}},
            {"method": "Network.responseReceived", "response": {"url": "https://example.com/index.html", "remoteIPAddress": "93.184.216.34", "remotePort": 443}},
        ]
        aligned_flows, labels, urls = aligner.align_flows(flows, cdp_events)
        self.assertEqual(len(aligned_flows), 2)
        # Flow to analytics should be labeled 1 (tracker)
        self.assertEqual(labels[0], 1)
        # Flow to example.com should be labeled 0 (benign)
        self.assertEqual(labels[1], 0)

    def test_tracker_radar_labeler(self):
        from netrack.dataset import TrackerRadarLabeler, BenchmarkFlowGenerator

        radar_labeler = TrackerRadarLabeler(min_fingerprint_score=2)
        # fpjs.io has fingerprinting score 3
        self.assertTrue(radar_labeler.is_fingerprinting("https://fpjs.io/api/v3/telemetry"))
        self.assertEqual(radar_labeler.get_fingerprinting_score("https://fpjs.io/api"), 3)
        # Normal content domain has score 0
        self.assertFalse(radar_labeler.is_fingerprinting("https://example.com/index.html"))
        self.assertEqual(radar_labeler.get_fingerprinting_score("https://example.com/logo.png"), 0)

        # Flow test
        flow, urls = BenchmarkFlowGenerator(seed=42).generate_fingerprinting_flow()
        self.assertGreater(len(flow), 5)
        self.assertEqual(radar_labeler.label_flow(urls), 1)

    def test_princeton_tsv_loader(self):
        from netrack.dataset import PrincetonTSVLoader

        loader = PrincetonTSVLoader()
        # Verify that scripts from the downloaded TSVs are correctly indexed
        tech_id = loader.get_technique("https://www.cdn-net.com/cc.js")
        self.assertIn(tech_id, [1, 2, 3, 4])
        self.assertTrue(loader.is_fingerprinting("https://www.cdn-net.com/cc.js"))

        # Verify benign script is 0
        self.assertEqual(loader.get_technique("https://example.com/main.js"), 0)
        self.assertFalse(loader.is_fingerprinting("https://example.com/main.js"))

    def test_two_tier_defense(self):
        from netrack.deep_model import TwoTierFingerprintDefense
        from netrack.dataset import BenchmarkFlowGenerator

        defense = TwoTierFingerprintDefense()
        flow, urls = BenchmarkFlowGenerator(seed=42).generate_fingerprinting_flow()
        result = defense.analyze_flow(flow, urls)

        self.assertIn("tier1_random_forest", result)
        self.assertIn("tier2_deep_learning", result)
        self.assertIn("overall_decision", result)
        self.assertIn("flow_summary", result)
        self.assertGreater(result["flow_summary"]["packet_count"], 0)

    def test_fingerprint_defense_trainer(self):
        import shutil
        import tempfile
        from netrack.deep_model import FingerprintDefenseTrainer

        temp_dir = tempfile.mkdtemp()
        try:
            trainer = FingerprintDefenseTrainer(models_dir=temp_dir, seq_len=15, random_state=42)
            results = trainer.train_all(
                num_benign=40,
                num_per_technique=10,
                epochs=2,
                batch_size=16,
                lr=0.01,
            )
            self.assertIn("tier1_rf", results)
            self.assertIn("tier2_deep", results)
            self.assertGreater(results["tier1_rf"]["accuracy"], 0.70)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "tier1_rf.joblib")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "tier2_hybrid_attention.pt")))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


    def test_netsentinel_dashboard(self):
        from netrack.ui import NetSentinelDashboard

        dashboard = NetSentinelDashboard()
        dashboard.process_new_flow(is_fingerprint=True, tech_id=1)
        dashboard.process_new_flow(is_fingerprint=False)

        self.assertEqual(dashboard.total_scanned, 2)
        self.assertGreaterEqual(dashboard.total_blocked, 1)
        self.assertEqual(len(dashboard.feed_logs), 2)
        self.assertIsNotNone(dashboard.selected_flow)

    def test_live_sniffer_and_tracker(self):
        from netrack.live import get_default_interface, LiveFlowTracker, LivePacketSniffer

        iface, local_ip = get_default_interface()
        self.assertTrue(len(iface) > 0)
        self.assertTrue(len(local_ip) > 0)

        # Test flow tracker
        key = ("192.168.1.100", 45000, "93.184.216.34", 443)
        tracker = LiveFlowTracker(key=key, client_ip="192.168.1.100", domain="example.com")
        tracker.add_packet("192.168.1.100", 120)  # Uplink (+1)
        tracker.add_packet("93.184.216.34", 1420)  # Downlink (-1)
        tracker.add_packet("93.184.216.34", 64, is_fin=True)  # Downlink FIN

        self.assertEqual(len(tracker.packets), 3)
        self.assertEqual(tracker.packets[0], (1, 120))
        self.assertEqual(tracker.packets[1], (-1, 1420))
        self.assertTrue(tracker.is_finished)

        # Test sniffer init and queue
        sniffer = LivePacketSniffer(interface=iface)
        self.assertFalse(sniffer.running)
        self.assertIsNone(sniffer.get_next_flow(timeout=0.001))

    def test_dashboard_live_flow_processing(self):
        from netrack.ui import NetSentinelDashboard

        dashboard = NetSentinelDashboard(live_mode=True)
        self.assertTrue(dashboard.live_mode)
        self.assertIsNotNone(dashboard.sniffer)

        live_packets = [(1, 150), (-1, 1400), (1, 200), (-1, 800)]
        dashboard.process_flow(
            flow=live_packets,
            urls=["https://wikipedia.org/wiki/Main_Page"],
            domain="wikipedia.org",
            source="LIVE",
        )
        self.assertEqual(dashboard.total_scanned, 1)
        self.assertEqual(dashboard.feed_logs[-1]["domain"], "wikipedia.org")
        self.assertEqual(dashboard.feed_logs[-1]["source"], "LIVE")


if __name__ == "__main__":
    unittest.main()
