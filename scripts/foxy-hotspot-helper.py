#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ipaddress
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import List, Optional


def log(msg: str) -> None:
    print(f"[foxy-hotspot] {msg}", flush=True)


def run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and result.returncode != 0:
        output = result.stdout.strip()
        raise RuntimeError(f"{' '.join(cmd)} failed with {result.returncode}" + (f":\n{output}" if output else ""))
    return result


def require_tools(names: List[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"Missing required hotspot tools: {', '.join(missing)}")


def find_wifi_interface() -> str:
    if shutil.which("iw"):
        result = run(["iw", "dev"], check=False)
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Interface "):
                return line.split(None, 1)[1]
    if shutil.which("nmcli"):
        result = run(["nmcli", "-t", "-f", "DEVICE,TYPE", "device"], check=False)
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[1] == "wifi":
                return parts[0]
    raise SystemExit("Could not autodetect a Wi-Fi interface. Pass --interface wlanX.")


def wlan_network(address: str) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(f"{address}/24", strict=False)


def hostapd_config(args: argparse.Namespace, mode: str) -> str:
    if mode == "5ghz":
        hw_mode = "a"
        channel = "36"
        label = "5 GHz"
    else:
        hw_mode = "g"
        channel = "6"
        label = "2.4 GHz"

    lines = [
        f"interface={args.interface}",
        "driver=nl80211",
        f"ssid={args.ssid}",
        f"hw_mode={hw_mode}",
        f"channel={channel}",
        "ieee80211n=1",
        "wmm_enabled=1",
        "auth_algs=1",
        "ignore_broadcast_ssid=0",
    ]
    if args.country:
        lines.extend([
            f"country_code={args.country.upper()}",
            "ieee80211d=1",
        ])
    if args.password:
        lines.extend([
            "wpa=2",
            f"wpa_passphrase={args.password}",
            "wpa_key_mgmt=WPA-PSK",
            "rsn_pairwise=CCMP",
        ])

    log(f"Trying {label} AP on channel {channel}")
    return "\n".join(lines) + "\n"


def dnsmasq_config(args: argparse.Namespace) -> str:
    net = wlan_network(args.address)
    start = ipaddress.ip_address(int(net.network_address) + 50)
    end = ipaddress.ip_address(int(net.network_address) + 150)
    return "\n".join([
        f"interface={args.interface}",
        "bind-interfaces",
        f"listen-address={args.address}",
        "no-hosts",
        "no-resolv",
        "domain-needed",
        "bogus-priv",
        "dhcp-authoritative",
        f"dhcp-range={start},{end},{net.netmask},12h",
        f"dhcp-option=option:dns-server,{args.address}",
        f"address=/{args.domain}/{args.address}",
        "address=/#/0.0.0.0",
    ]) + "\n"


class Hotspot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tmp: Optional[tempfile.TemporaryDirectory[str]] = None
        self.hostapd: Optional[subprocess.Popen[bytes]] = None
        self.dnsmasq: Optional[subprocess.Popen[bytes]] = None
        self.old_ip_forward: Optional[str] = None
        self.nm_was_touched = False

    def prepare_interface(self) -> None:
        if shutil.which("nmcli"):
            run(["nmcli", "device", "set", self.args.interface, "managed", "no"], check=False)
            self.nm_was_touched = True
        run(["ip", "link", "set", self.args.interface, "down"], check=False)
        run(["ip", "addr", "flush", "dev", self.args.interface], check=False)
        run(["ip", "addr", "add", f"{self.args.address}/24", "dev", self.args.interface])
        run(["ip", "link", "set", self.args.interface, "up"])
        forward_path = pathlib.Path("/proc/sys/net/ipv4/ip_forward")
        try:
            self.old_ip_forward = forward_path.read_text(encoding="utf-8").strip()
            forward_path.write_text("0\n", encoding="utf-8")
        except Exception as e:
            log(f"Warning: could not force IPv4 forwarding off: {type(e).__name__}: {e}")

    def start_hostapd(self, mode: str) -> bool:
        assert self.tmp is not None
        conf = pathlib.Path(self.tmp.name) / f"hostapd-{mode}.conf"
        conf.write_text(hostapd_config(self.args, mode), encoding="utf-8")
        self.hostapd = subprocess.Popen(["hostapd", str(conf)])
        time.sleep(3.0)
        if self.hostapd.poll() is None:
            return True
        log(f"{mode} AP failed; hostapd exited with {self.hostapd.returncode}")
        self.hostapd = None
        return False

    def start_dnsmasq(self) -> None:
        assert self.tmp is not None
        conf = pathlib.Path(self.tmp.name) / "dnsmasq.conf"
        conf.write_text(dnsmasq_config(self.args), encoding="utf-8")
        self.dnsmasq = subprocess.Popen(["dnsmasq", "--no-daemon", "-C", str(conf)])
        time.sleep(1.0)
        if self.dnsmasq.poll() is not None:
            raise RuntimeError(f"dnsmasq exited with {self.dnsmasq.returncode}")

    def start(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="foxy-hotspot-")
        self.prepare_interface()
        if not self.start_hostapd("5ghz"):
            run(["ip", "link", "set", self.args.interface, "down"], check=False)
            run(["ip", "link", "set", self.args.interface, "up"], check=False)
            if not self.start_hostapd("2ghz"):
                raise RuntimeError("Could not start a 5 GHz or 2.4 GHz AP with hostapd")
        self.start_dnsmasq()
        if self.args.ready_file:
            pathlib.Path(self.args.ready_file).write_text("ready\n", encoding="utf-8")
        log(f"SSID {self.args.ssid!r} is up with local DNS {self.args.domain} -> {self.args.address}")
        log(f"Open https://{self.args.domain}:{self.args.server_port} on the Quest")
        log("Internet forwarding is disabled for this helper.")

    def stop(self) -> None:
        if self.args.ready_file:
            try:
                pathlib.Path(self.args.ready_file).unlink()
            except FileNotFoundError:
                pass
        for proc in (self.dnsmasq, self.hostapd):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        run(["ip", "addr", "flush", "dev", self.args.interface], check=False)
        run(["ip", "link", "set", self.args.interface, "down"], check=False)
        if self.old_ip_forward is not None:
            try:
                pathlib.Path("/proc/sys/net/ipv4/ip_forward").write_text(self.old_ip_forward + "\n", encoding="utf-8")
            except Exception as e:
                log(f"Warning: could not restore IPv4 forwarding: {type(e).__name__}: {e}")
        if self.nm_was_touched:
            run(["nmcli", "device", "set", self.args.interface, "managed", "yes"], check=False)
        if self.tmp:
            self.tmp.cleanup()

    def wait(self) -> int:
        while True:
            if self.hostapd and self.hostapd.poll() is not None:
                return int(self.hostapd.returncode or 0)
            if self.dnsmasq and self.dnsmasq.poll() is not None:
                return int(self.dnsmasq.returncode or 0)
            time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an isolated Foxy Wi-Fi AP with local-only DNS.")
    p.add_argument("--interface", default="")
    p.add_argument("--ssid", default="foxy")
    p.add_argument("--password", default="foxy")
    p.add_argument("--domain", default="foxy.local")
    p.add_argument("--address", default="10.42.0.1")
    p.add_argument("--server-port", type=int, default=8766)
    p.add_argument("--country", default="")
    p.add_argument("--ready-file", default="")
    args = p.parse_args()
    args.domain = args.domain.lower().rstrip(".")
    if args.password and not (8 <= len(args.password) <= 63):
        raise SystemExit(
            "Wi-Fi WPA2 passphrases must be 8..63 characters. "
            "The requested default password 'foxy' is too short for WPA2; "
            "pass --hotspot-password with at least 8 characters, or pass an empty password for an open AP."
        )
    if args.country and len(args.country) != 2:
        raise SystemExit("--country must be a two-letter regulatory code, for example US")
    ipaddress.ip_address(args.address)
    if not args.interface:
        args.interface = find_wifi_interface()
    return args


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        raise SystemExit("The hotspot helper must run as root.")
    require_tools(["hostapd", "dnsmasq", "ip"])
    hotspot = Hotspot(args)
    stopping = False

    def stop_handler(signum, frame) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        log("Stopping hotspot")
        hotspot.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        hotspot.start()
        return hotspot.wait()
    finally:
        hotspot.stop()


if __name__ == "__main__":
    raise SystemExit(main())
