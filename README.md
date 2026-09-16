# Round3
# CanSat Ground Control System

Live 3D telemetry tracker for a CanSat over an XBee3 radio link.

## Install

```bash
pip install digi-xbee matplotlib numpy
```
(`digi-xbee` is only needed for a real radio — `--simulate` and `--replay` work without it.)

## Usage

```bash
# Simulated flight, no hardware needed
python gcs.py --simulate

# Real radio link
python gcs.py --port COM7 --baud 9600           # Windows
python gcs.py --port /dev/ttyUSB0 --baud 9600   # Linux / macOS

# Replay a recorded flight at 4x speed
python gcs.py --replay logs/flight_20260916_101500.raw.txt --speed 4

# Enable uplink (type a command in the terminal to broadcast it)
python gcs.py --port COM7 --uplink
```

## Packet format

One line per packet, comma-separated, newline-terminated:

```
Timestamp(int), State(int), Temperature(float), Pressure(float),
Altitude(float), Battery Voltage(float), Battery Current(float),
Latitude(float), Longitude(float), Prev_CMD_echo(str)
```

## Options

| Flag | Description |
|---|---|
| `-p/--port` | Serial port of the XBee3 coordinator |
| `-b/--baud` | Baud rate (default 9600) |
| `--simulate` | Run a simulated flight |
| `--replay RAW_LOG` | Replay a `.raw.txt` log |
| `--speed` | Sim/replay speed multiplier |
| `--rate` | Sim/replay packet rate (Hz) |
| `--sim-apogee` | Simulated apogee altitude (m) |
| `--error-rate` | Fraction of corrupt sim packets |
| `--uplink` | Enable command console |
| `--log-dir` | Directory for flight logs (default `logs`) |
| `--no-log` | Disable disk logging |
| `--history` | Points kept on the plot |
| `--interval` | Redraw interval (ms) |

## Output

Each run (unless `--no-log`) writes to the log directory:
- `flight_<timestamp>.csv` — parsed, timestamped telemetry
- `flight_<timestamp>.raw.txt` — raw line stream, for replay

## Notes

- Corrupted or out-of-range packets are dropped and counted, never crash the run.
- Radio I/O runs on a background thread; parsing and plotting happen on the main thread.


# Parafoil 6-DOF Dynamics + Autonomous Homing Guidance

Simulates a rigid 6-DOF parafoil (canopy + payload) and steers it to a ground
target using an energy-management + proportional homing guidance law.

Model is from Zhao, Tao, Sun & Sun, "Dynamic modelling of parafoil system
based on aerodynamic coefficients identification", Automatika 64:2 (2023).
Aerodynamic coefficients come from the paper's Tables 1-3; moments of
inertia aren't published, so representative values are assumed (marked in
the code).

## Install

```bash
pip install numpy matplotlib
```

## Run

```bash
python3 parafoil_guidance.py
```

Produces `parafoil_path.png` - a 3D plot of the flight path colored by
guidance phase, plus a printed landing summary (miss distance, flight time).

## How it works

- **Dynamics**: full nonlinear 6-DOF equations of motion, integrated with RK4.
- **Guidance**:
  - **SPIRAL** - if too high to glide directly to the target, hold a
    constant-deflection turn to bleed altitude.
  - **HOMING** - once within the reachable glide cone, steer heading
    proportionally toward the target.
  - **CAPTURE** - stop steering once within the capture radius.
- **Bonus utility**: `lla_to_local_xyz()` converts lat/lon/altitude telemetry
  (e.g. from a CanSat ground station) into the local north/east/down frame
  the guidance model uses.

## Notes

- Control input is `delta_a`, an asymmetric brake deflection in `[-1, 1]`.
- No wind estimation or final flare leg - deliberately a simple, complete
  closed loop, not a flight-ready controller.
