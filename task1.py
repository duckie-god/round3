#!/usr/bin/env python3
"""CanSat Ground Control System - live 3D telemetry tracker."""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

# --------------------------------------------------------------------------- #
# 1. Packet model + parser
# --------------------------------------------------------------------------- #

STATE_NAMES = {
    0: "IDLE", 1: "ASCENT", 2: "APOGEE", 3: "DESCENT", 4: "LANDED",
}


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class TelemetryPacket:
    timestamp: int
    state: int
    temperature: float
    pressure: float
    altitude: float
    battery_voltage: float
    battery_current: float
    latitude: float
    longitude: float
    prev_cmd_echo: str

    FIELDS = (
        ("timestamp", lambda s: int(float(s))),
        ("state", lambda s: int(float(s))),
        ("temperature", float),
        ("pressure", float),
        ("altitude", float),
        ("battery_voltage", float),
        ("battery_current", float),
        ("latitude", float),
        ("longitude", float),
        ("prev_cmd_echo", str),
    )

    @classmethod
    def parse(cls, line: str) -> "TelemetryPacket":
        parts = [p.strip() for p in line.strip().split(",")]
        n = len(cls.FIELDS)

        if len(parts) < n:
            raise ParseError(f"expected {n} fields, got {len(parts)}")
        if len(parts) > n:
            parts = parts[: n - 1] + [",".join(parts[n - 1:])]

        values = {}
        for (name, conv), raw in zip(cls.FIELDS, parts):
            if raw == "" and conv is not str:
                raise ParseError(f"empty field '{name}'")
            try:
                values[name] = conv(raw)
            except ValueError:
                raise ParseError(f"field '{name}' is not valid: {raw!r}") from None

        pkt = cls(**values)
        pkt.validate()
        return pkt

    def validate(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ParseError(f"latitude out of range: {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ParseError(f"longitude out of range: {self.longitude}")
        if not -500.0 <= self.altitude <= 50_000.0:
            raise ParseError(f"altitude out of range: {self.altitude}")
        if self.timestamp < 0:
            raise ParseError("negative timestamp")

    @property
    def has_fix(self) -> bool:
        return not (abs(self.latitude) < 1e-7 and abs(self.longitude) < 1e-7)

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, f"S{self.state}")

    def as_row(self) -> list:
        return [getattr(self, f.name) for f in dataclass_fields(self)]

    @classmethod
    def header(cls) -> list:
        return [name for name, _ in cls.FIELDS]


# --------------------------------------------------------------------------- #
# 2. Byte stream -> lines
# --------------------------------------------------------------------------- #

class LineAssembler:
    _SPLIT = re.compile(r"[\r\n]+")

    def __init__(self, max_buffer: int = 4096):
        self._buf = ""
        self._max = max_buffer

    def feed(self, text: str) -> list[str]:
        self._buf += text
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]
        chunks = self._SPLIT.split(self._buf)
        self._buf = chunks.pop()
        return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# 3. Telemetry sources (real radio / simulator / log replay)
# --------------------------------------------------------------------------- #

class TelemetrySource:
    name = "source"

    def __init__(self, on_line):
        self._on_line = on_line
        self._asm = LineAssembler()

    def _ingest(self, raw: bytes) -> None:
        for line in self._asm.feed(raw.decode("utf-8", errors="replace")):
            self._on_line(line)

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def send(self, text: str) -> None:
        raise NotImplementedError(f"{self.name} does not support uplink")


class XBeeSource(TelemetrySource):
    name = "XBee3"

    def __init__(self, on_line, port: str, baud: int):
        super().__init__(on_line)
        self.port, self.baud = port, baud
        self._device = None

    def start(self) -> None:
        from digi.xbee.devices import XBeeDevice

        self._device = XBeeDevice(self.port, self.baud)
        self._device.open()
        self._device.flush_queues()
        print(f"[xbee ] open on {self.port} @ {self.baud} "
              f"(local addr {self._device.get_64bit_addr()})")

        def _on_message(xbee_message):
            try:
                self._ingest(bytes(xbee_message.data))
            except Exception as exc:
                print(f"[xbee ] callback error: {exc}", file=sys.stderr)

        self._device.add_data_received_callback(_on_message)

    def send(self, text: str) -> None:
        if self._device and self._device.is_open():
            self._device.send_data_broadcast((text + "\n").encode())

    def stop(self) -> None:
        if self._device is not None and self._device.is_open():
            self._device.close()
            print("[xbee ] closed")


class SimSource(TelemetrySource):
    name = "simulator"

    def __init__(self, on_line, rate=4.0, origin=(12.97160, 77.59460),
                 apogee=700.0, error_rate=0.03, speed=1.0):
        super().__init__(on_line)
        self.rate, self.apogee, self.error_rate = rate, apogee, error_rate
        self.speed = speed
        self.lat, self.lon = origin
        self.alt = 0.0
        self._t0 = time.time()
        self._last_cmd = "CXON"
        self._stop = threading.Event()
        self._thread = None

    def _step(self, t: float, dt: float) -> int:
        ascent_rate, descent_rate = 40.0, 7.5
        t_ascent = self.apogee / ascent_rate
        if t < 4.0:
            state, vz, wind = 0, 0.0, 0.0
        elif t < 4.0 + t_ascent:
            state, vz, wind = 1, ascent_rate, 1.5
        elif t < 6.0 + t_ascent:
            state, vz, wind = 2, 0.0, 2.0
        elif self.alt > 0.5:
            state, vz, wind = 3, -descent_rate, 5.0
        else:
            state, vz, wind = 4, 0.0, 0.0

        self.alt = max(0.0, self.alt + vz * dt + random.gauss(0, 0.4))
        east = wind * dt * random.uniform(0.6, 1.2)
        north = wind * dt * random.uniform(-0.4, 0.9)
        self.lon += east / (111_320.0 * math.cos(math.radians(self.lat)))
        self.lat += north / 111_320.0
        return state

    def _packet(self, t: float, state: int) -> str:
        pressure = 1013.25 * (1 - 2.25577e-5 * self.alt) ** 5.25588
        temp = 27.0 - 0.0065 * self.alt + random.gauss(0, 0.15)
        volt = 8.31 - 0.0009 * t + random.gauss(0, 0.01)
        amp = (0.72 if state == 2 else 0.34) + random.gauss(0, 0.02)
        return (f"{int(t * 1000)},{state},{temp:.2f},{pressure:.2f},"
                f"{self.alt:.2f},{volt:.2f},{amp:.2f},"
                f"{self.lat:.6f},{self.lon:.6f},{self._last_cmd}")

    def _corrupt(self, line: str) -> str:
        style = random.choice(("truncate", "noise", "drop_field"))
        if style == "truncate":
            return line[: random.randint(5, max(6, len(line) // 2))]
        if style == "drop_field":
            parts = line.split(",")
            del parts[random.randrange(len(parts))]
            return ",".join(parts)
        i = random.randrange(len(line))
        return line[:i] + random.choice("@#?\x00\xff") + line[i + 1:]

    def _run(self) -> None:
        period = 1.0 / self.rate
        prev = self._t0
        while not self._stop.is_set():
            now = time.time()
            t = (now - self._t0) * self.speed
            state = self._step(t, max(1e-3, (now - prev) * self.speed))
            prev = now
            line = self._packet(t, state)
            if random.random() < self.error_rate:
                line = self._corrupt(line)
            payload = (line + "\n").encode("utf-8", errors="replace")
            if random.random() < 0.25 and len(payload) > 10:
                cut = random.randrange(4, len(payload) - 4)
                self._ingest(payload[:cut])
                self._ingest(payload[cut:])
            else:
                self._ingest(payload)
            self._stop.wait(period / self.speed)

    def start(self) -> None:
        print(f"[sim  ] simulated flight, {self.rate} Hz, "
              f"apogee {self.apogee:.0f} m, {self.error_rate:.0%} corrupt packets")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, text: str) -> None:
        self._last_cmd = text.strip().upper()[:24]

    def stop(self) -> None:
        self._stop.set()


class ReplaySource(TelemetrySource):
    name = "replay"

    def __init__(self, on_line, path: str, rate=4.0, speed=1.0):
        super().__init__(on_line)
        self.path, self.rate, self.speed = path, rate, speed
        self._stop = threading.Event()
        self._thread = None

    def _run(self) -> None:
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        print(f"[replay] {len(lines)} lines from {self.path} at {self.speed}x")
        prev_ts = None
        for line in lines:
            if self._stop.is_set():
                return
            delay = 1.0 / self.rate
            try:
                ts = int(float(line.split(",")[0]))
                if prev_ts is not None and 0 <= ts - prev_ts < 10_000:
                    delay = (ts - prev_ts) / 1000.0
                prev_ts = ts
            except (ValueError, IndexError):
                pass
            self._ingest((line + "\n").encode())
            self._stop.wait(delay / self.speed)
        print("[replay] end of log")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# 4. Telemetry bus: queue, parse, validate, log, keep history
# --------------------------------------------------------------------------- #

class FlightLogger:
    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(directory, f"flight_{stamp}.csv")
        self.raw_path = os.path.join(directory, f"flight_{stamp}.raw.txt")
        self._csv_fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_fh)
        self._writer.writerow(["gcs_utc"] + TelemetryPacket.header())
        self._raw_fh = open(self.raw_path, "w", encoding="utf-8")
        print(f"[log  ] {self.csv_path}\n[log  ] {self.raw_path}")

    def raw(self, line: str) -> None:
        self._raw_fh.write(line + "\n")
        self._raw_fh.flush()

    def packet(self, pkt: TelemetryPacket) -> None:
        self._writer.writerow([datetime.now(timezone.utc).isoformat(timespec="milliseconds")]
                              + pkt.as_row())
        self._csv_fh.flush()

    def close(self) -> None:
        for fh in (self._csv_fh, self._raw_fh):
            try:
                fh.close()
            except OSError:
                pass


class TelemetryBus:
    def __init__(self, logger: FlightLogger | None, history: int = 5000):
        self._q: queue.Queue[tuple[float, str]] = queue.Queue(maxsize=10_000)
        self._logger = logger
        self.packets: deque[TelemetryPacket] = deque(maxlen=history)
        self.rx_ok = 0
        self.rx_bad = 0
        self.last_error = ""
        self.last_rx_wall = None
        self._recent = deque(maxlen=25)

    def submit(self, line: str) -> None:
        try:
            self._q.put_nowait((time.time(), line))
        except queue.Full:
            pass

    def drain(self) -> int:
        new = 0
        while True:
            try:
                arrived, line = self._q.get_nowait()
            except queue.Empty:
                break
            if self._logger:
                self._logger.raw(line)
            try:
                pkt = TelemetryPacket.parse(line)
            except ParseError as exc:
                self.rx_bad += 1
                self.last_error = str(exc)
                continue
            self.rx_ok += 1
            self.last_rx_wall = arrived
            self._recent.append(arrived)
            if self._logger:
                self._logger.packet(pkt)
            if pkt.has_fix:
                self.packets.append(pkt)
                new += 1
        return new

    @property
    def rate_hz(self) -> float:
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1] - self._recent[0]
        return (len(self._recent) - 1) / span if span > 0 else 0.0

    @property
    def link_age(self) -> float:
        return float("inf") if self.last_rx_wall is None else time.time() - self.last_rx_wall


# --------------------------------------------------------------------------- #
# 5. Live 3D plot
# --------------------------------------------------------------------------- #

class LivePlot:
    MIN_DEG_SPAN = 4e-4
    MIN_ALT_SPAN = 20.0

    def __init__(self, bus: TelemetryBus, source: TelemetrySource, interval_ms=200):
        self.bus, self.source = bus, source
        plt.style.use("dark_background")

        self.fig = plt.figure("CanSat GCS - live 3D telemetry", figsize=(11.5, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("Longitude (deg)", labelpad=12)
        self.ax.set_ylabel("Latitude (deg)", labelpad=12)
        self.ax.set_zlabel("Altitude (m)", labelpad=8)
        self.ax.set_title(f"CanSat Ground Control  -  source: {source.name}",
                          pad=14, fontsize=13)
        for axis in (self.ax.xaxis, self.ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
        self.ax.view_init(elev=24, azim=-58)
        self.fig.patch.set_facecolor("#05080d")
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.04, 0.07, 0.11, 1.0))
            axis._axinfo["grid"].update(color=(0.30, 0.40, 0.52, 0.45),
                                        linewidth=0.6)
            axis.set_tick_params(labelsize=8, colors="#9fb3c8")

        self.trail = Line3DCollection([[(0, 0, 0), (0, 0, 0)]],
                                      cmap="turbo", linewidths=2.2)
        self.trail.set_array(np.zeros(1))
        self.ax.add_collection3d(self.trail)
        self.ground, = self.ax.plot([], [], [], color="#8899aa", lw=1.0,
                                    ls="--", alpha=0.8, label="ground track")
        self.drop, = self.ax.plot([], [], [], color="#8899aa", lw=0.8, alpha=0.5)
        self.craft, = self.ax.plot([], [], [], marker="o", ms=9, mfc="#ff3860",
                                   mec="white", mew=1.2, ls="none", label="CanSat")
        self.launch, = self.ax.plot([], [], [], marker="x", ms=9, color="#33ff99",
                                    ls="none", label="first fix")
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.25)

        self.hud = self.ax.text2D(0.015, 0.97, "", transform=self.ax.transAxes,
                                  family="monospace", fontsize=9.5,
                                  va="top", ha="left", color="#e8f0ff",
                                  bbox=dict(boxstyle="round,pad=0.55",
                                            fc="#0d1b2a", ec="#2a4a6a", alpha=0.88))
        self._cbar = None
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.source.stop())
        self._anim = FuncAnimation(self.fig, self._update, interval=interval_ms,
                                   blit=False, cache_frame_data=False)

    def _update(self, _frame):
        self.bus.drain()
        pkts = list(self.bus.packets)
        if not pkts:
            self.hud.set_text(self._hud_text(None))
            return ()

        lon = np.fromiter((p.longitude for p in pkts), float, len(pkts))
        lat = np.fromiter((p.latitude for p in pkts), float, len(pkts))
        alt = np.fromiter((p.altitude for p in pkts), float, len(pkts))

        lo_x, hi_x = self._span(lon, self.MIN_DEG_SPAN)
        lo_y, hi_y = self._span(lat, self.MIN_DEG_SPAN)
        lo_z, hi_z = self._span(alt, self.MIN_ALT_SPAN, pad=0.08, floor=0.0)
        self.ax.set_xlim(lo_x, hi_x)
        self.ax.set_ylim(lo_y, hi_y)
        self.ax.set_zlim(lo_z, hi_z)

        if len(pkts) > 1:
            pts = np.column_stack([lon, lat, alt]).reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            self.trail.set_segments(segs)
            self.trail.set_array(alt[1:])
            self.trail.set_clim(alt.min(), max(alt.max(), alt.min() + 1.0))
            if self._cbar is None:
                self._cbar = self.fig.colorbar(self.trail, ax=self.ax, pad=0.13,
                                               shrink=0.60, aspect=24)
                self._cbar.set_label("Altitude (m)", fontsize=9)

        floor = np.full(len(pkts), lo_z)
        self.ground.set_data_3d(lon, lat, floor)
        self.drop.set_data_3d([lon[-1], lon[-1]], [lat[-1], lat[-1]], [lo_z, alt[-1]])
        self.craft.set_data_3d([lon[-1]], [lat[-1]], [alt[-1]])
        self.launch.set_data_3d([lon[0]], [lat[0]], [lo_z])
        self.hud.set_text(self._hud_text(pkts))
        return ()

    @staticmethod
    def _span(values, min_span, pad=0.12, floor=None):
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < min_span:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - min_span / 2, mid + min_span / 2
        margin = (hi - lo) * pad
        lo, hi = lo - margin, hi + margin
        if floor is not None:
            lo = min(floor, lo)
        return lo, hi

    def _hud_text(self, pkts) -> str:
        bus = self.bus
        total = bus.rx_ok + bus.rx_bad
        good = 100.0 * bus.rx_ok / total if total else 0.0
        age = bus.link_age
        link = "NO SIGNAL" if age == float("inf") else (
            "LIVE" if age < 3 else f"STALE {age:4.1f}s")
        head = (f"LINK {link:<12} {bus.rate_hz:4.1f} Hz\n"
                f"RX   {bus.rx_ok:5d} ok / {bus.rx_bad:3d} bad  ({good:5.1f}% good)")
        if not pkts:
            tail = "\n\nwaiting for the first valid GPS fix..."
            if bus.last_error:
                tail += f"\nlast parse error: {bus.last_error[:42]}"
            return head + tail

        p = pkts[-1]
        vs = self._vertical_speed(pkts)
        dist, brg = self._range_bearing(pkts[0], p)
        return (f"{head}\n"
                f"{'-' * 42}\n"
                f"T+       {p.timestamp / 1000:8.1f} s      STATE  {p.state_name}\n"
                f"ALT      {p.altitude:8.1f} m      V/S  {vs:+6.1f} m/s\n"
                f"LAT      {p.latitude:11.6f}\n"
                f"LON      {p.longitude:11.6f}\n"
                f"RANGE    {dist:8.1f} m      BRG  {brg:5.1f} deg\n"
                f"TEMP     {p.temperature:8.2f} C      PRES {p.pressure:8.2f} hPa\n"
                f"BATT     {p.battery_voltage:8.2f} V      CURR {p.battery_current:6.2f} A\n"
                f"CMD      {p.prev_cmd_echo[:24]}")

    @staticmethod
    def _vertical_speed(pkts, window_ms: int = 2000) -> float:
        if len(pkts) < 2:
            return 0.0
        b, a = pkts[-1], pkts[0]
        for p in reversed(pkts[:-1]):
            a = p
            if b.timestamp - p.timestamp >= window_ms:
                break
        dt = (b.timestamp - a.timestamp) / 1000.0
        return (b.altitude - a.altitude) / dt if dt > 1e-6 else 0.0

    @staticmethod
    def _range_bearing(a: TelemetryPacket, b: TelemetryPacket):
        r = 6_371_000.0
        p1, p2 = math.radians(a.latitude), math.radians(b.latitude)
        dl = math.radians(b.longitude - a.longitude)
        dp = p2 - p1
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        dist = 2 * r * math.asin(min(1.0, math.sqrt(h)))
        y = math.sin(dl) * math.cos(p2)
        x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        return dist, (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    def run(self) -> None:
        self.fig.subplots_adjust(left=0.02, right=0.93, top=0.95, bottom=0.04)
        plt.show()


# --------------------------------------------------------------------------- #
# 6. Optional uplink console
# --------------------------------------------------------------------------- #

def start_uplink_console(source: TelemetrySource) -> None:
    def loop():
        print("[uplink] type a command + Enter to broadcast (e.g. CAL, CXON, SIM_ENABLE)")
        for line in sys.stdin:
            cmd = line.strip()
            if not cmd:
                continue
            try:
                source.send(cmd)
                print(f"[uplink] -> {cmd}")
            except Exception as exc:
                print(f"[uplink] failed: {exc}", file=sys.stderr)
    threading.Thread(target=loop, daemon=True).start()


# --------------------------------------------------------------------------- #
# 7. Entry point
# --------------------------------------------------------------------------- #

def build_source(args, on_line) -> TelemetrySource:
    if args.simulate:
        return SimSource(on_line, rate=args.rate, apogee=args.sim_apogee,
                         error_rate=args.error_rate, speed=args.speed)
    if args.replay:
        return ReplaySource(on_line, args.replay, rate=args.rate, speed=args.speed)
    if not args.port:
        sys.exit("error: give --port (e.g. --port COM7 / --port /dev/ttyUSB0), "
                 "or use --simulate / --replay")
    return XBeeSource(on_line, args.port, args.baud)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CanSat ground control - live 3D telemetry")
    ap.add_argument("-p", "--port", help="serial port of the XBee3 coordinator")
    ap.add_argument("-b", "--baud", type=int, default=9600, help="baud rate (default 9600)")
    ap.add_argument("--simulate", action="store_true", help="run a simulated flight")
    ap.add_argument("--replay", metavar="RAW_LOG", help="replay a .raw.txt log")
    ap.add_argument("--speed", type=float, default=1.0, help="sim/replay speed multiplier")
    ap.add_argument("--rate", type=float, default=4.0, help="sim/replay packet rate (Hz)")
    ap.add_argument("--sim-apogee", type=float, default=700.0, help="sim apogee (m)")
    ap.add_argument("--error-rate", type=float, default=0.03,
                    help="fraction of corrupt sim packets (default 0.03)")
    ap.add_argument("--uplink", action="store_true", help="enable the command console")
    ap.add_argument("--log-dir", default="logs", help="where to write flight logs")
    ap.add_argument("--no-log", action="store_true", help="disable disk logging")
    ap.add_argument("--history", type=int, default=5000, help="points kept on the plot")
    ap.add_argument("--interval", type=int, default=200, help="redraw interval (ms)")
    args = ap.parse_args(argv)

    logger = None if args.no_log else FlightLogger(args.log_dir)
    bus = TelemetryBus(logger, history=args.history)
    source = build_source(args, bus.submit)

    try:
        source.start()
    except Exception as exc:
        print(f"error: could not start {source.name}: {exc}", file=sys.stderr)
        if logger:
            logger.close()
        return 1

    if args.uplink:
        start_uplink_console(source)

    plot = LivePlot(bus, source, interval_ms=args.interval)
    try:
        plot.run()
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        if logger:
            logger.close()
        print(f"\n[done ] {bus.rx_ok} good packets, {bus.rx_bad} rejected, "
              f"{len(bus.packets)} plotted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
