"""
Live Packet Sniffer & Real-Time Flow Aggregator.

Passive, 100% read-only network sniffer:
- Captures live TCP/TLS connections from active interfaces (e.g., wlan0, eno1)
- Aggregates packets by canonical 4-tuple into bidirectional flows (+1 uplink, -1 downlink)
- Extracts TLS SNI (Server Name Indication) domain names and resolves hostnames
- Feeds live flow traces directly into the Two-Tier Defense model
- Strictly passive (NO iptables rules, NO packet drops, NO connection interference)
"""

import os
import sys
import time
import socket
import logging
import queue
import threading
import subprocess
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def get_default_interface() -> Tuple[str, str]:
    """Returns (interface_name, local_ip) for the default outbound route."""
    try:
        out = subprocess.check_output(["ip", "route", "get", "8.8.8.8"], text=True).split()
        dev_idx = out.index("dev") + 1
        src_idx = out.index("src") + 1
        return out[dev_idx], out[src_idx]
    except Exception:
        # Fallback
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = "127.0.0.1"
        finally:
            s.close()
        return "wlan0", local_ip


class LiveFlowTracker:
    """Tracks a single bidirectional TCP connection flow."""

    def __init__(self, key: Tuple[str, int, str, int], client_ip: str, domain: Optional[str] = None):
        self.key = key
        self.client_ip = client_ip
        self.domain = domain
        self.packets: List[Tuple[int, int]] = []
        self.start_time = time.time()
        self.last_seen = time.time()
        self.is_finished = False
        self.emitted = False

    def add_packet(self, src_ip: str, frame_len: int, is_fin: bool = False, is_rst: bool = False):
        direction = 1 if src_ip == self.client_ip else -1
        self.packets.append((direction, frame_len))
        self.last_seen = time.time()
        if is_fin or is_rst:
            self.is_finished = True


class LivePacketSniffer:
    """Passively sniffs TCP packets and emits completed flow traces."""

    def __init__(
        self,
        interface: Optional[str] = None,
        bpf_filter: str = "tcp port 443 or tcp port 80",
        min_packets: int = 8,
        max_packets: int = 35,
        flow_timeout_sec: float = 2.0,
    ):
        detected_iface, local_ip = get_default_interface()
        self.interface = interface or detected_iface
        self.local_ip = local_ip
        self.bpf_filter = bpf_filter
        self.min_packets = min_packets
        self.max_packets = max_packets
        self.flow_timeout_sec = flow_timeout_sec

        self.flow_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.active_flows: Dict[Tuple[str, int, str, int], LiveFlowTracker] = {}
        self.dns_cache: Dict[str, str] = {}

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.proc: Optional[subprocess.Popen] = None

    def _resolve_ip(self, ip: str) -> str:
        if ip in self.dns_cache:
            return self.dns_cache[ip]
        try:
            name = socket.gethostbyaddr(ip)[0]
            self.dns_cache[ip] = name
            return name
        except Exception:
            self.dns_cache[ip] = ip
            return ip

    def _capture_worker(self):
        """Worker thread running tshark line-buffered parser."""
        cmd = [
            "tshark",
            "-i", self.interface,
            "-l",  # line buffered
            "-n",  # don't resolve network names (faster)
            "-f", self.bpf_filter,
            "-T", "fields",
            "-e", "ip.src",
            "-e", "tcp.srcport",
            "-e", "ip.dst",
            "-e", "tcp.dstport",
            "-e", "frame.len",
            "-e", "tcp.flags.fin",
            "-e", "tcp.flags.reset",
            "-e", "tls.handshake.extensions_server_name",
        ]

        logger.info(f"Starting passive live sniffer on interface '{self.interface}' (local IP: {self.local_ip})...")
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            logger.error(f"Failed to start tshark process: {e}")
            return

        last_sweep = time.time()

        for line in iter(self.proc.stdout.readline, ""):
            if not self.running:
                break
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            src_ip = parts[0].strip()
            src_port_s = parts[1].strip()
            dst_ip = parts[2].strip()
            dst_port_s = parts[3].strip()
            len_s = parts[4].strip()

            if not src_ip or not dst_ip or not src_port_s or not dst_port_s or not len_s:
                continue

            try:
                src_port = int(src_port_s)
                dst_port = int(dst_port_s)
                frame_len = int(len_s)
            except ValueError:
                continue

            is_fin = ("True" in parts[5]) if len(parts) > 5 else False
            is_rst = ("True" in parts[6]) if len(parts) > 6 else False
            sni = parts[7].strip() if len(parts) > 7 else ""

            # Canonical 4-tuple key
            if (src_ip, src_port) < (dst_ip, dst_port):
                key = (src_ip, src_port, dst_ip, dst_port)
            else:
                key = (dst_ip, dst_port, src_ip, src_port)

            tracker = self.active_flows.get(key)
            if tracker is None:
                # New connection
                remote_ip = dst_ip if src_ip == self.local_ip else src_ip
                domain = sni if sni else self._resolve_ip(remote_ip)
                tracker = LiveFlowTracker(key=key, client_ip=self.local_ip, domain=domain)
                self.active_flows[key] = tracker

            if sni and not tracker.domain:
                tracker.domain = sni

            tracker.add_packet(src_ip, frame_len, is_fin, is_rst)

            # Ready to emit if reached observation window or closed
            if (len(tracker.packets) >= self.min_packets and not tracker.emitted) or tracker.is_finished:
                self._emit_flow(tracker)

            # Sweep expired flows every 1 second
            now = time.time()
            if now - last_sweep > 1.0:
                self._sweep_expired_flows(now)
                last_sweep = now

        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _emit_flow(self, tracker: LiveFlowTracker):
        """Emits a flow to the consumer queue."""
        packets = list(tracker.packets[:self.max_packets])
        domain = tracker.domain or "unknown-remote.net"
        urls = [f"https://{domain}/live"]
        try:
            self.flow_queue.put_nowait({
                "packets": packets,
                "urls": urls,
                "domain": domain,
                "key": tracker.key,
                "packet_count": len(packets),
                "is_finished": tracker.is_finished,
            })
            tracker.emitted = True
        except queue.Full:
            pass

    def _sweep_expired_flows(self, now: float):
        """Checks for timed out flows and emits them."""
        to_del = []
        for key, tracker in list(self.active_flows.items()):
            if tracker.is_finished or (now - tracker.last_seen > self.flow_timeout_sec):
                if not tracker.emitted and len(tracker.packets) >= 4:
                    self._emit_flow(tracker)
                to_del.append(key)
        for k in to_del:
            self.active_flows.pop(k, None)

    def start(self):
        """Starts passive capture in a background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops passive capture cleanly."""
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=0.5)
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)

    def get_next_flow(self, timeout: float = 0.05) -> Optional[Dict[str, Any]]:
        """Retrieves the next captured flow from the live interface."""
        try:
            return self.flow_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
