"""
Dataset Collection, Flow Splitting, Filter List Ground-Truth Labeling, and Split Management.

Combines:
- WebTrafficCollector (Selenium + TShark live website crawling)
- BenchmarkFlowGenerator (Synthetic/Sample flow trace dataset generation)
- FlowSplitter (Scapy 4-tuple PCAP flow extractor)
- FilterListLabeler (EasyList & EasyPrivacy rule matcher)
- DatasetManager (20% validation / 80% test splits & scale/balance reporting)
-
"""

import os
import re
import json
import time
import random
import logging
import ipaddress
import subprocess
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import requests
from tabulate import tabulate
from adblockparser import AdblockRules
from scapy.all import rdpcap, IP, IPv6, TCP, UDP

logger = logging.getLogger(__name__)

EASYLIST_URL = "https://easylist.to/easylist/easylist.txt"
EASYPRIVACY_URL = "https://easylist.to/easylist/easyprivacy.txt"

PAPER_BENIGN_COUNT = 222009
PAPER_TRACKER_COUNT = 126664
PAPER_TOTAL_COUNT = PAPER_BENIGN_COUNT + PAPER_TRACKER_COUNT
PAPER_BENIGN_RATIO = PAPER_BENIGN_COUNT / PAPER_TOTAL_COUNT
PAPER_TRACKER_RATIO = PAPER_TRACKER_COUNT / PAPER_TOTAL_COUNT


class WebTrafficCollector:
    """Visits website homepages with Chrome via Selenium and captures traffic with tshark for 120s."""

    def __init__(self, output_dir: str = "data/raw_pcaps", capture_duration: int = 120, interface: str = "any"):
        self.output_dir = output_dir
        self.capture_duration = capture_duration
        self.interface = interface
        os.makedirs(output_dir, exist_ok=True)

    def _start_tshark_capture(self, pcap_filename: str) -> Optional[subprocess.Popen]:
        pcap_path = os.path.join(self.output_dir, pcap_filename)
        cmd = ["tshark", "-i", self.interface, "-w", pcap_path, "-a", f"duration:{self.capture_duration}"]
        try:
            logger.info(f"Starting tshark capture: {' '.join(cmd)}")
            return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"Failed to launch tshark: {e}")
            return None

    def crawl_site(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        clean_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
        timestamp = int(time.time())
        pcap_name = f"{clean_name}_{timestamp}.pcap"
        cdp_log_name = f"{clean_name}_{timestamp}_cdp.json"
        pcap_path = os.path.join(self.output_dir, pcap_name)
        cdp_log_path = os.path.join(self.output_dir, cdp_log_name)

        tshark_proc = self._start_tshark_capture(pcap_name)

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        driver = None
        cdp_events: List[Dict[str, Any]] = []
        try:
            logger.info(f"Launching Chrome to visit: {url}")
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            driver.get(url)
            time.sleep(self.capture_duration)

            try:
                perf_logs = driver.get_log("performance")
                for entry in perf_logs:
                    try:
                        msg = json.loads(entry["message"])["message"]
                        method = msg.get("method", "")
                        if method in ("Network.requestWillBeSent", "Network.responseReceived"):
                            cdp_events.append(msg)
                    except Exception:
                        pass
                with open(cdp_log_path, "w", encoding="utf-8") as f:
                    json.dump(cdp_events, f, indent=2)
                logger.info(f"Saved {len(cdp_events)} CDP network events to {cdp_log_path}")
            except Exception as e:
                logger.warning(f"Could not retrieve CDP logs: {e}")
        except Exception as e:
            logger.warning(f"Error visiting {url} with Selenium: {e}")
        finally:
            if driver:
                driver.quit()
            if tshark_proc:
                tshark_proc.wait()

        pcap_ret = pcap_path if os.path.exists(pcap_path) and os.path.getsize(pcap_path) > 0 else None
        cdp_ret = cdp_log_path if os.path.exists(cdp_log_path) and os.path.getsize(cdp_log_path) > 0 else None
        return pcap_ret, cdp_ret


class BenchmarkFlowGenerator:
    """Generates synthetic flow traces matching empirical distributions of Net-track dataset."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_flow(self, is_tracker: bool) -> Tuple[List[Tuple[int, int]], List[str]]:
        if is_tracker:
            num_packets = random.randint(5, 75)
            urls = ["https://google-analytics.com/collect", "https://doubleclick.net/gampad/ads"]
        else:
            num_packets = random.randint(5, 100)
            urls = ["https://example.com/index.html", "https://cdn.example.com/app.js"]

        packets = []
        noisy_label = is_tracker if (random.random() > 0.06) else (not is_tracker)

        for i in range(num_packets):
            if i < 5:
                is_uplink = random.random() < (0.50 if noisy_label else 0.40)
                if is_uplink:
                    length = int(np.clip(np.random.normal(300, 150), 54, 1460))
                    direction = 1
                else:
                    length = int(np.clip(np.random.normal(800, 400), 54, 1500))
                    direction = -1
            else:
                if noisy_label:
                    is_uplink = random.random() < 0.35
                    if is_uplink:
                        direction = 1
                        length = int(np.clip(np.random.normal(450, 200), 54, 1460))
                    else:
                        direction = -1
                        if random.random() < 0.65:
                            length = int(np.clip(np.random.normal(250, 120), 54, 800))
                        else:
                            length = int(np.clip(np.random.normal(1200, 250), 54, 1500))
                else:
                    is_uplink = random.random() < 0.18
                    if is_uplink:
                        direction = 1
                        length = int(np.clip(np.random.normal(200, 100), 54, 1200))
                    else:
                        direction = -1
                        if random.random() < 0.75:
                            length = int(np.clip(np.random.normal(1440, 60), 1000, 1514))
                        else:
                            length = int(np.clip(np.random.normal(400, 200), 54, 1000))

            packets.append((direction, length))

        return packets, urls

    def generate_fingerprinting_flow(self) -> Tuple[List[Tuple[int, int]], List[str]]:
        """Simulates the physical network signature of a browser fingerprinting session:
        - Handshake (SYN, SYN-ACK, ACK, TLS Client/Server Hello)
        - Heavy client uplink burst containing serialized JSON canvas/audio/font hash (~600-1400B)
        - Minimal server acknowledgment (HTTP 204 No Content / small cookie, 54-150B)
        - Short overall sequence (8-25 packets).
        """
        num_packets = random.randint(8, 25)
        urls = ["https://fpjs.io/api/v3/telemetry", "https://threatmetrix.com/fp/check"]
        packets = []

        # Handshake
        packets.append((1, random.choice([64, 74])))
        packets.append((-1, random.choice([64, 74])))
        packets.append((1, 54))
        packets.append((1, int(np.clip(np.random.normal(517, 80), 300, 800))))
        packets.append((-1, int(np.clip(np.random.normal(1460, 50), 1200, 1500))))

        # Fingerprint POST payload burst (Uplink dominant)
        payload_packets = random.randint(2, 5)
        for _ in range(payload_packets):
            packets.append((1, int(np.clip(np.random.normal(950, 180), 500, 1460))))

        # Server response: HTTP 204 No Content / empty response
        packets.append((-1, int(np.clip(np.random.normal(120, 40), 54, 250))))
        packets.append((1, 54))

        while len(packets) < num_packets:
            direction = 1 if random.random() < 0.6 else -1
            length = random.choice([54, 60, 120])
            packets.append((direction, length))

        return packets[:num_packets], urls

    def generate_technique_flow(
        self, technique_id: int, loader: Optional[Any] = None
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        """Simulates physical flow packet traces and attaches ground-truth URLs for specific techniques:
        0: Benign
        1: Canvas Fingerprinting
        2: AudioContext Fingerprinting
        3: Canvas-Font Fingerprinting
        4: WebRTC Local IP Discovery
        """
        if technique_id == 0:
            return self.generate_flow(is_tracker=False)

        packets = []
        # Handshake: Client SYN (64/74B), Server SYN-ACK, Client ACK, TLS Client Hello, TLS Server Hello
        packets.append((1, random.choice([64, 74])))
        packets.append((-1, random.choice([64, 74])))
        packets.append((1, 54))
        packets.append((1, int(np.clip(np.random.normal(517, 80), 300, 800))))
        packets.append((-1, int(np.clip(np.random.normal(1460, 50), 1200, 1500))))

        if technique_id == 1:
            # Canvas Fingerprinting: 2 to 4 consecutive large uplink packets (800-1420 bytes)
            num_payload = random.randint(2, 4)
            for _ in range(num_payload):
                packets.append((1, int(np.clip(np.random.normal(1050, 150), 750, 1460))))
            packets.append((-1, int(np.clip(np.random.normal(120, 30), 54, 250))))
            packets.append((1, 54))
            default_urls = ["https://fpjs.io/api/v3/telemetry", "https://threatmetrix.com/fp/check"]

        elif technique_id == 2:
            # AudioContext Fingerprinting: 1 to 2 medium uplink packets (500-900 bytes)
            num_payload = random.randint(1, 2)
            for _ in range(num_payload):
                packets.append((1, int(np.clip(np.random.normal(680, 100), 450, 950))))
            packets.append((-1, int(np.clip(np.random.normal(100, 25), 54, 200))))
            packets.append((1, 54))
            default_urls = ["https://audio-fingerprint.io/collect", "https://ligatus.com/audiodata"]

        elif technique_id == 3:
            # Canvas-Font Fingerprinting: 3 to 6 large uplink packets (1000-1460 bytes)
            num_payload = random.randint(3, 6)
            for _ in range(num_payload):
                packets.append((1, int(np.clip(np.random.normal(1250, 120), 950, 1460))))
            packets.append((-1, int(np.clip(np.random.normal(150, 40), 54, 300))))
            packets.append((1, 54))
            default_urls = ["https://fontmetrics.iovation.com/metrics", "https://cdn.perimeterx.net/font.js"]

        elif technique_id == 4:
            # WebRTC Local IP Discovery: 1 small uplink packet (180-450 bytes)
            packets.append((1, int(np.clip(np.random.normal(320, 60), 180, 480))))
            packets.append((-1, int(np.clip(np.random.normal(80, 20), 54, 150))))
            packets.append((1, 54))
            default_urls = ["https://webrtc-leak.privacy.net/candidate", "https://ip-check.kasada.io/rtc"]

        else:
            return self.generate_flow(is_tracker=False)

        # Retrieve actual ground truth script URLs from Princeton TSV loader if available
        urls = []
        if loader and hasattr(loader, "get_sample_urls"):
            sampled = loader.get_sample_urls(technique_id, count=1)
            if sampled:
                urls = sampled
        if not urls:
            urls = default_urls

        # Add tail jitter/termination packets
        extra = random.randint(1, 4)
        for _ in range(extra):
            packets.append((1 if random.random() < 0.5 else -1, random.choice([54, 60, 120])))

        return packets, urls

    def generate_fingerprinting_dataset(
        self,
        num_benign: int = 1000,
        num_per_technique: int = 250,
        loader: Optional[Any] = None,
    ) -> Tuple[List[List[Tuple[int, int]]], List[int], List[int], List[List[str]]]:
        """Generates a multi-class dataset for training both Tier 1 and Tier 2 models.

        Returns:
            flows: list of packet sequences
            binary_labels: 0 (benign) or 1 (fingerprinting)
            technique_labels: 0 (benign), 1 (canvas), 2 (audio), 3 (font), 4 (webrtc)
            urls_list: list of request URLs per flow
        """
        flows, bin_labels, tech_labels, urls_list = [], [], [], []
        logger.info(
            f"Generating dataset: {num_benign} benign flows, {num_per_technique} flows for each of 4 fingerprinting techniques..."
        )

        # 0: Benign
        for _ in range(num_benign):
            flow, urls = self.generate_flow(is_tracker=False)
            flows.append(flow)
            bin_labels.append(0)
            tech_labels.append(0)
            urls_list.append(urls)

        # 1..4: Fingerprinting techniques
        for tech_id in range(1, 5):
            for _ in range(num_per_technique):
                flow, urls = self.generate_technique_flow(tech_id, loader=loader)
                flows.append(flow)
                bin_labels.append(1)
                tech_labels.append(tech_id)
                urls_list.append(urls)

        # Shuffle consistently
        combined = list(zip(flows, bin_labels, tech_labels, urls_list))
        random.shuffle(combined)
        flows, bin_labels, tech_labels, urls_list = zip(*combined)

        return list(flows), list(bin_labels), list(tech_labels), list(urls_list)

    def generate_dataset(
        self, num_benign: int = 2220, num_tracker: int = 1266
    ) -> Tuple[List[List[Tuple[int, int]]], List[int], List[List[str]]]:
        flows, labels, urls_list = [], [], []
        logger.info(f"Generating dataset with {num_benign} benign flows and {num_tracker} tracker flows...")

        for _ in range(num_benign):
            flow, urls = self.generate_flow(is_tracker=False)
            flows.append(flow)
            labels.append(0)
            urls_list.append(urls)

        for _ in range(num_tracker):
            flow, urls = self.generate_flow(is_tracker=True)
            flows.append(flow)
            labels.append(1)
            urls_list.append(urls)

        combined = list(zip(flows, labels, urls_list))
        random.shuffle(combined)
        flows, labels, urls_list = zip(*combined)
        return list(flows), list(labels), list(urls_list)


class FlowSplitter:
    """Splits multi-connection PCAP traces into distinct per-connection flow traces

    with proper client-initiator directionality (+1 uplink, -1 downlink).
    """

    def __init__(self, client_ip_prefix: Optional[str] = None, filter_pure_acks: bool = False):
        self.client_ip_prefix = client_ip_prefix
        self.filter_pure_acks = filter_pure_acks

    @staticmethod
    def _is_private_ip(ip_str: str) -> bool:
        try:
            return ipaddress.ip_address(ip_str).is_private
        except Exception:
            return False

    @staticmethod
    def _get_flow_key(pkt) -> Optional[Tuple[str, int, str, int]]:
        if IP in pkt:
            src_ip, dst_ip = str(pkt[IP].src), str(pkt[IP].dst)
        elif IPv6 in pkt:
            src_ip, dst_ip = str(pkt[IPv6].src), str(pkt[IPv6].dst)
        else:
            return None

        if TCP in pkt:
            src_port, dst_port = int(pkt[TCP].sport), int(pkt[TCP].dport)
        elif UDP in pkt:
            src_port, dst_port = int(pkt[UDP].sport), int(pkt[UDP].dport)
        else:
            return None

        return (src_ip, src_port, dst_ip, dst_port)

    def extract_flows_from_pcap(
        self, pcap_path: str
    ) -> Dict[Tuple[str, int, str, int], List[Tuple[int, int]]]:
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        logger.info(f"Reading PCAP file: {pcap_path}...")
        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            logger.error(f"Failed to parse PCAP {pcap_path}: {e}")
            return {}

        return self.extract_flows_from_packets(packets)

    def extract_flows_from_packets(
        self, packets: List[Any]
    ) -> Dict[Tuple[str, int, str, int], List[Tuple[int, int]]]:
        canonical_flows: Dict[Tuple, List[Tuple[int, int]]] = {}
        client_endpoints: Dict[Tuple, Tuple[str, int]] = {}

        for pkt in packets:
            key = self._get_flow_key(pkt)
            if key is None:
                continue
            src_ip, src_port, dst_ip, dst_port = key

            # Canonical bidirectional pair key
            if (src_ip, src_port) < (dst_ip, dst_port):
                pair_key = (src_ip, src_port, dst_ip, dst_port)
            else:
                pair_key = (dst_ip, dst_port, src_ip, src_port)

            # Determine or confirm client initiator
            if pair_key not in client_endpoints:
                is_syn = (TCP in pkt) and (pkt[TCP].flags & 0x02) and not (pkt[TCP].flags & 0x10)
                if is_syn:
                    client_endpoints[pair_key] = (src_ip, src_port)
                elif self.client_ip_prefix and src_ip.startswith(self.client_ip_prefix):
                    client_endpoints[pair_key] = (src_ip, src_port)
                elif self.client_ip_prefix and dst_ip.startswith(self.client_ip_prefix):
                    client_endpoints[pair_key] = (dst_ip, dst_port)
                elif self._is_private_ip(src_ip) and not self._is_private_ip(dst_ip):
                    client_endpoints[pair_key] = (src_ip, src_port)
                elif self._is_private_ip(dst_ip) and not self._is_private_ip(src_ip):
                    client_endpoints[pair_key] = (dst_ip, dst_port)
                elif dst_port in {80, 443, 8080, 8443} and src_port > 1024:
                    client_endpoints[pair_key] = (src_ip, src_port)
                elif src_port in {80, 443, 8080, 8443} and dst_port > 1024:
                    client_endpoints[pair_key] = (dst_ip, dst_port)
                else:
                    client_endpoints[pair_key] = (src_ip, src_port)

            # Pure ACK filtering (0 payload and ACK flag set)
            if self.filter_pure_acks and TCP in pkt:
                payload_len = len(pkt[TCP].payload) if hasattr(pkt[TCP], "payload") else 0
                if payload_len == 0 and (pkt[TCP].flags & 0x10):
                    continue

            # Direction: +1 for client -> server (uplink), -1 for server -> client (downlink)
            client_ip, client_port = client_endpoints[pair_key]
            if (src_ip, src_port) == (client_ip, client_port):
                direction = +1
            else:
                direction = -1

            length = len(pkt)
            if pair_key not in canonical_flows:
                canonical_flows[pair_key] = []
            canonical_flows[pair_key].append((direction, length))

        logger.info(f"Extracted {len(canonical_flows)} distinct flows.")
        return canonical_flows


class NetworkTraceAligner:
    """Aligns raw PCAP connection flows with Chrome DevTools Protocol (CDP) / HAR logs

    to solve the encrypted TLS-to-URL ground truth labeling gap.
    """

    def __init__(self, labeler: Optional["FilterListLabeler"] = None):
        self.labeler = labeler

    def parse_cdp_log(self, cdp_log_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(cdp_log_path):
            return []
        try:
            with open(cdp_log_path, "r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception as e:
            logger.error(f"Error reading CDP log {cdp_log_path}: {e}")
            return []

        requests_info = []
        for ev in events:
            method = ev.get("method")
            params = ev.get("params", {})
            if method == "Network.requestWillBeSent":
                req = params.get("request", {})
                url = req.get("url")
                if url:
                    requests_info.append({
                        "url": url,
                        "timestamp": params.get("timestamp", 0),
                        "wallTime": params.get("wallTime", 0),
                    })
            elif method == "Network.responseReceived":
                resp = params.get("response", {})
                url = resp.get("url")
                remote_ip = resp.get("remoteIPAddress")
                remote_port = resp.get("remotePort")
                if url:
                    requests_info.append({
                        "url": url,
                        "remoteIPAddress": remote_ip,
                        "remotePort": remote_port,
                        "timestamp": params.get("timestamp", 0),
                    })
        return requests_info

    def align_flows(
        self,
        flows: Dict[Tuple[str, int, str, int], List[Tuple[int, int]]],
        cdp_events: List[Dict[str, Any]],
        labeler: Optional["FilterListLabeler"] = None,
    ) -> Tuple[List[List[Tuple[int, int]]], List[int], List[List[str]]]:
        """Maps URLs from CDP network logs to each 4-tuple flow by remote IP and port,

        then applies FilterListLabeler to assign ground-truth labels.
        """
        active_labeler = labeler or self.labeler or FilterListLabeler(auto_download=False)
        ip_port_to_urls: Dict[Tuple[str, int], List[str]] = {}
        ip_to_urls: Dict[str, List[str]] = {}

        # Normalize raw or flattened CDP events
        flattened_events: List[Dict[str, Any]] = []
        for ev in cdp_events:
            if "url" in ev:
                flattened_events.append(ev)
            elif "params" in ev:
                params = ev.get("params", {})
                resp = params.get("response", {})
                req = params.get("request", {})
                url = resp.get("url") or req.get("url")
                if url:
                    flattened_events.append({
                        "url": url,
                        "remoteIPAddress": resp.get("remoteIPAddress"),
                        "remotePort": resp.get("remotePort"),
                    })
            elif "response" in ev:
                resp = ev.get("response", {})
                url = resp.get("url")
                if url:
                    flattened_events.append({
                        "url": url,
                        "remoteIPAddress": resp.get("remoteIPAddress"),
                        "remotePort": resp.get("remotePort"),
                    })

        for req in flattened_events:
            url = req.get("url")
            if not url:
                continue
            remote_ip = req.get("remoteIPAddress")
            remote_port = req.get("remotePort")
            if remote_ip and remote_port:
                key = (str(remote_ip), int(remote_port))
                ip_port_to_urls.setdefault(key, []).append(url)
            if remote_ip:
                ip_to_urls.setdefault(str(remote_ip), []).append(url)

        aligned_flows, labels, urls_list = [], [], []

        for flow_key, packet_list in flows.items():
            src_ip, src_port, dst_ip, dst_port = flow_key
            matched_urls: List[str] = []

            if (dst_ip, dst_port) in ip_port_to_urls:
                matched_urls.extend(ip_port_to_urls[(dst_ip, dst_port)])
            elif (src_ip, src_port) in ip_port_to_urls:
                matched_urls.extend(ip_port_to_urls[(src_ip, src_port)])
            elif dst_ip in ip_to_urls:
                matched_urls.extend(ip_to_urls[dst_ip])
            elif src_ip in ip_to_urls:
                matched_urls.extend(ip_to_urls[src_ip])

            unique_urls = list(dict.fromkeys(matched_urls))
            label = active_labeler.label_flow(unique_urls) if unique_urls else 0

            aligned_flows.append(packet_list)
            labels.append(label)
            urls_list.append(unique_urls)

        return aligned_flows, labels, urls_list


class TrancoDatasetLoader:
    """Manages Tranco Top Sites lists and real flow dataset imports."""

    DEFAULT_TRANCO_SITES = [
        "https://google.com", "https://youtube.com", "https://facebook.com", "https://amazon.com",
        "https://yahoo.com", "https://wikipedia.org", "https://reddit.com", "https://twitter.com",
        "https://netflix.com", "https://linkedin.com", "https://cnn.com", "https://nytimes.com",
        "https://forbes.com", "https://espn.com", "https://weather.com", "https://huffpost.com",
        "https://buzzfeed.com", "https://theverge.com", "https://techcrunch.com", "https://ebay.com",
    ]

    def __init__(self, cache_dir: str = "data/tranco"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_top_sites(self, limit: int = 20) -> List[str]:
        return self.DEFAULT_TRANCO_SITES[:limit]


class PrincetonTSVLoader:
    """Loads and indexes Princeton Web Census fingerprinting script TSV files.

    Extracts ground-truth URLs and domains across 4 techniques:
    1: Canvas Fingerprinting
    2: AudioContext Fingerprinting
    3: Canvas-Font Fingerprinting
    4: WebRTC Local IP Discovery
    """

    TECHNIQUE_NAMES = {
        0: "Benign",
        1: "Canvas Fingerprinting",
        2: "AudioContext Fingerprinting",
        3: "Canvas-Font Fingerprinting",
        4: "WebRTC Local IP Discovery",
    }

    def __init__(self, tsv_dir: str = "data/princeton_fingerprinting"):
        self.tsv_dir = tsv_dir
        self.script_to_technique: Dict[str, int] = {}
        self.domain_to_technique: Dict[str, int] = {}
        self.technique_scripts: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}
        self.stats: Dict[str, int] = {}
        self._load_all_tsvs()

    def _clean_domain(self, url: str) -> str:
        clean = url.lower().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
        return clean

    def _load_single_tsv(self, filename: str, technique_id: int):
        filepath = os.path.join(self.tsv_dir, filename)
        if not os.path.exists(filepath):
            return
        count = 0
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        first_party, script_url = parts[0].strip(), parts[1].strip()
                        if script_url:
                            script_lower = script_url.lower()
                            self.script_to_technique[script_lower] = technique_id
                            if script_url not in self.technique_scripts[technique_id]:
                                self.technique_scripts[technique_id].append(script_url)
                            domain = self._clean_domain(script_url)
                            if domain:
                                self.domain_to_technique[domain] = technique_id
                            count += 1
            self.stats[self.TECHNIQUE_NAMES[technique_id]] = count
            logger.info(f"Loaded {count} entries for {self.TECHNIQUE_NAMES[technique_id]} from {filename}")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    def get_sample_urls(self, technique_id: int, count: int = 1) -> List[str]:
        """Returns sample ground-truth script URLs for a specific fingerprinting technique."""
        scripts = self.technique_scripts.get(technique_id, [])
        if scripts:
            return random.sample(scripts, min(count, len(scripts)))
        return []

    def _load_all_tsvs(self):
        self._load_single_tsv("canvas_fingerprinting.tsv", 1)
        self._load_single_tsv("audio_fingerprinting.tsv", 2)
        self._load_single_tsv("font_fingerprinting.tsv", 3)
        self._load_single_tsv("webrtc_ip_retrieval.tsv", 4)
        if self.script_to_technique:
            logger.info(
                f"Indexed {len(self.script_to_technique)} fingerprinting scripts across {len(self.domain_to_technique)} domains."
            )

    def get_technique(self, url: str) -> int:
        if not url:
            return 0
        url_lower = url.lower()
        if url_lower in self.script_to_technique:
            return self.script_to_technique[url_lower]
        domain = self._clean_domain(url)
        if any(d in domain for d in ["fpjs.io", "fingerprintjs", "threatmetrix"]):
            return 1
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            sub = ".".join(parts[i:])
            if sub in self.domain_to_technique:
                return self.domain_to_technique[sub]
        return 0

    def is_fingerprinting(self, url: str) -> bool:
        return self.get_technique(url) > 0

    def label_flow_technique(self, request_urls: List[str]) -> Tuple[int, str]:
        """Returns (technique_id, technique_name) for a flow's URLs."""
        for url in request_urls:
            tech_id = self.get_technique(url)
            if tech_id > 0:
                return tech_id, self.TECHNIQUE_NAMES.get(tech_id, "Unknown")
        return 0, "Benign"


class TrackerRadarLabeler:
    """Labels flows using DuckDuckGo Tracker Radar domain intelligence with fingerprinting scores (0 to 3)."""

    KNOWN_FINGERPRINTING_DOMAINS = {
        "fingerprintjs.com": 3,
        "fpjs.io": 3,
        "threatmetrix.com": 3,
        "iovation.com": 3,
        "perimeterx.net": 3,
        "kasada.io": 3,
        "augur.io": 3,
        "datadoghq.com": 2,
        "branch.io": 2,
        "appsflyer.com": 2,
        "adjust.com": 2,
        "criteo.com": 2,
        "scorecardresearch.com": 2,
        "hotjar.com": 2,
        "clarity.ms": 2,
    }

    def __init__(self, radar_data_dir: Optional[str] = None, min_fingerprint_score: int = 2):
        if radar_data_dir is None and os.path.exists("data/tracker-radar"):
            radar_data_dir = "data/tracker-radar"
        self.radar_data_dir = radar_data_dir
        self.min_fingerprint_score = min_fingerprint_score
        self.domain_scores: Dict[str, int] = dict(self.KNOWN_FINGERPRINTING_DOMAINS)
        if self.radar_data_dir and os.path.exists(self.radar_data_dir):
            self._load_radar_directory(self.radar_data_dir)

    def _load_radar_directory(self, path: str):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            domain = data.get("domain", "")
                            fp_score = data.get("fingerprinting", 0)
                            if domain and fp_score:
                                self.domain_scores[domain.lower()] = int(fp_score)
                    except Exception:
                        pass

    def get_fingerprinting_score(self, url: str) -> int:
        if not url:
            return 0
        clean = url.lower().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
        parts = clean.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self.domain_scores:
                return self.domain_scores[parent]
        if any(term in url.lower() for term in ["fingerprint", "fp.js", "canvas", "device_id", "client_hints", "telemetry"]):
            return 2
        return 0

    def is_fingerprinting(self, url: str) -> bool:
        return self.get_fingerprinting_score(url) >= self.min_fingerprint_score

    def label_flow(self, request_urls: List[str]) -> int:
        for url in request_urls:
            if self.is_fingerprinting(url):
                return 1
        return 0


class FilterListLabeler:
    """Labels flow trace ground truth using EasyList and EasyPrivacy filter lists."""

    def __init__(
        self,
        easylist_path: Optional[str] = None,
        easyprivacy_path: Optional[str] = None,
        auto_download: bool = True,
    ):
        self.rules: Optional[AdblockRules] = None
        self._load_or_download_rules(easylist_path, easyprivacy_path, auto_download)

    def _load_or_download_rules(
        self, easylist_path: Optional[str], easyprivacy_path: Optional[str], auto_download: bool
    ):
        raw_rules: List[str] = []

        def get_rules_from_source(path: Optional[str], url: str, name: str) -> List[str]:
            if path and os.path.exists(path):
                logger.info(f"Loading {name} from local path: {path}")
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.readlines()
            elif auto_download:
                logger.info(f"Downloading {name} from {url}...")
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    return resp.text.splitlines()
                except Exception as e:
                    logger.warning(f"Failed to download {name}: {e}. Using fallback rule set.")
                    return []
            return []

        easylist_lines = get_rules_from_source(easylist_path, EASYLIST_URL, "EasyList")
        easyprivacy_lines = get_rules_from_source(easyprivacy_path, EASYPRIVACY_URL, "EasyPrivacy")
        all_lines = easylist_lines + easyprivacy_lines

        if not all_lines:
            logger.info("Using embedded default tracker & ad filter rules for fallback.")
            all_lines = [
                "||google-analytics.com^", "||googlesyndication.com^", "||doubleclick.net^",
                "||facebook.com/tr/*", "||scorecardresearch.com^", "||criteo.com^", "||outbrain.com^",
                "||taboola.com^", "||adnxs.com^", "||pubmatic.com^", "||rubiconproject.com^",
                "||openx.net^", "||amazon-adsystem.com^", "||clarity.ms^", "||hotjar.com^",
                "||segment.io^", "||mixpanel.com^", "||chartbeat.com^", "||quantserve.com^", "||omtrdc.net^",
            ]

        clean_rules = [l.strip() for l in all_lines if l.strip() and not l.startswith("!") and not l.startswith("[")]

        try:
            self.rules = AdblockRules(clean_rules, use_re2=False, skip_unsupported_rules=True)
        except Exception as e:
            logger.error(f"Error initializing AdblockRules: {e}. Falling back to basic regex.")
            self.rules = None
            self.fallback_patterns = [
                re.compile(r"analytics|doubleclick|googlesyndication|facebook\.com/tr|scorecardresearch|criteo|outbrain|taboola|adnxs|pubmatic|rubiconproject|openx|amazon-adsystem|clarity\.ms|hotjar|segment\.io|mixpanel|chartbeat|quantserve|omtrdc", re.IGNORECASE)
            ]

    def is_tracker_url(self, url: str, options: Optional[dict] = None) -> bool:
        if not url:
            return False
        if self.rules is not None:
            opts = options or {"third-party": True}
            try:
                return self.rules.should_block(url, opts)
            except Exception:
                pass
        if hasattr(self, "fallback_patterns"):
            for pattern in self.fallback_patterns:
                if pattern.search(url):
                    return True
        return False

    def label_flow(self, request_urls: List[str]) -> int:
        for url in request_urls:
            if self.is_tracker_url(url):
                return 1
        return 0


class DatasetManager:
    """Manages 20% validation / 80% test dataset splits and reports scale/class balance."""

    def __init__(self, val_ratio: float = 0.2, random_state: int = 42):
        self.val_ratio = val_ratio
        self.random_state = random_state

    def split_dataset(
        self, flows: List[Any], labels: List[int], urls: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        from sklearn.model_selection import train_test_split

        labels_arr = np.array(labels)
        indices = np.arange(len(labels))
        if urls is None:
            urls = [[] for _ in range(len(labels))]

        val_idx, test_idx = train_test_split(
            indices,
            test_size=(1.0 - self.val_ratio),
            stratify=labels_arr,
            random_state=self.random_state,
        )

        return {
            "val": {"flows": [flows[i] for i in val_idx], "labels": [labels[i] for i in val_idx], "urls": [urls[i] for i in val_idx], "indices": val_idx},
            "test": {"flows": [flows[i] for i in test_idx], "labels": [labels[i] for i in test_idx], "urls": [urls[i] for i in test_idx], "indices": test_idx},
        }

    @staticmethod
    def get_scale_and_balance_report(labels: List[int]) -> Dict[str, Any]:
        total = len(labels)
        benign_cnt = int(sum(1 for l in labels if l == 0))
        tracker_cnt = int(sum(1 for l in labels if l == 1))

        benign_pct = (benign_cnt / total * 100) if total > 0 else 0.0
        tracker_pct = (tracker_cnt / total * 100) if total > 0 else 0.0

        table_data = [
            ["Benign Flows", f"{benign_cnt:,}", f"{benign_pct:.2f}%", f"{PAPER_BENIGN_COUNT:,}", f"{PAPER_BENIGN_RATIO * 100:.2f}%"],
            ["Tracker Flows", f"{tracker_cnt:,}", f"{tracker_pct:.2f}%", f"{PAPER_TRACKER_COUNT:,}", f"{PAPER_TRACKER_RATIO * 100:.2f}%"],
            ["Total Dataset Scale", f"{total:,}", "100.00%", f"{PAPER_TOTAL_COUNT:,}", "100.00%"],
        ]

        headers = ["Flow Category", "Reproduced Count", "Reproduced %", "Paper Reference", "Paper %"]
        table_str = tabulate(table_data, headers=headers, tablefmt="github")

        return {
            "total": total,
            "benign_count": benign_cnt,
            "tracker_count": tracker_cnt,
            "benign_pct": benign_pct,
            "tracker_pct": tracker_pct,
            "table_str": table_str,
            "paper_reference": {
                "total": PAPER_TOTAL_COUNT,
                "benign": PAPER_BENIGN_COUNT,
                "tracker": PAPER_TRACKER_COUNT,
            },
        }
