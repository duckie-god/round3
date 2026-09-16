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
