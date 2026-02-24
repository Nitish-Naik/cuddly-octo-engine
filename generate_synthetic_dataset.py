"""

Generate synthetic industrial sensor data
=======================================

This folder contains a small utility to create synthetic time-series sensor data with labeled anomalies.

Quick usage
-----------

Run the generator and produce a CSV:

```bash
python scripts/generate_synthetic_sensor_data.py --out data/synthetic_sensor_data.csv --n-machines 5 --hours 0.1 --freq-s 1
```

Output columns: `timestamp`, `machine_id`, `temperature`, `vibration`, `pressure`, `humidity`, `anomaly`, `anomaly_type`.

Notes
-----
- `anomaly` is 0/1 per timestamp. `anomaly_type` is one of `none`, `spike`, `drift`, `stuck`.
- Use `pandas` to load and prepare windows/feature vectors for classification.


"""
"""Generate synthetic industrial sensor time-series with labeled anomalies.

Output CSV columns:
  - timestamp, machine_id
  - temperature, vibration, pressure, humidity
  - anomaly (0/1), anomaly_type (none/spike/drift/stuck)

Usage example:
  python scripts/generate_synthetic_sensor_data.py --out data/synthetic_sensor_data.csv --n-machines 5 --hours 0.1 --freq-s 1

This script is intended to create classification-ready per-timestamp labels for anomaly detection models.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    n_machines: int = 10
    hours: float = 1.0
    freq_s: int = 1
    seed: Optional[int] = 42
    out_csv: str = "data/synthetic_sensor_data.csv"


def _inject_spike(series: np.ndarray, mag: float, idx: int, dur: int = 1):
    end = min(len(series), idx + dur)
    series[idx:end] += mag


def _inject_drift(series: np.ndarray, start_idx: int, dur: int, slope: float):
    end = min(len(series), start_idx + dur)
    drift = slope * np.arange(end - start_idx)
    series[start_idx:end] += drift


def _inject_stuck(series: np.ndarray, start_idx: int, dur: int):
    end = min(len(series), start_idx + dur)
    if start_idx >= len(series):
        return
    val = series[start_idx]
    series[start_idx:end] = val


def generate_for_machine(machine_id: int, cfg: GeneratorConfig, start_ts: pd.Timestamp):
    n_samples = int(cfg.hours * 3600 / cfg.freq_s)
    if n_samples <= 0:
        return pd.DataFrame()

    time_index = pd.date_range(start=start_ts, periods=n_samples, freq=f"{cfg.freq_s}S")
    t = np.arange(n_samples)

    # Baselines and periodic components
    temperature = 50 + 0.5 * np.sin(2 * math.pi * t / 3600) + 0.2 * np.sin(2 * math.pi * t / 86400) + np.random.normal(0, 0.3, n_samples)
    vibration = 0.5 + 0.05 * np.sin(2 * math.pi * t / 600) + np.random.normal(0, 0.02, n_samples)
    pressure = 101.3 + 0.01 * (t / max(1, n_samples)) + np.random.normal(0, 0.02, n_samples)
    humidity = 30 + 5 * np.sin(2 * math.pi * t / 43200) + np.random.normal(0, 0.5, n_samples)

    anomaly = np.zeros(n_samples, dtype=int)
    anomaly_type = np.array(["none"] * n_samples, dtype=object)

    # Number of anomalies per machine
    n_anoms = np.random.poisson(3)
    for _ in range(n_anoms):
        typ = random.choice(["spike", "drift", "stuck"])
        idx = np.random.randint(0, n_samples)

        if typ == "spike":
            # apply spikes to one or more sensors
            dur = np.random.randint(1, max(2, int(2 / cfg.freq_s) + 1))
            mag_t = np.random.uniform(3, 12)  # temperature spike
            mag_v = np.random.uniform(0.2, 1.0)  # vibration spike
            _inject_spike(temperature, mag_t, idx, dur)
            _inject_spike(vibration, mag_v, idx, dur)
            anomaly[idx: idx + dur] = 1
            anomaly_type[idx: idx + dur] = "spike"

        elif typ == "drift":
            dur = np.random.randint(max(10, int(10 / cfg.freq_s)), max(20, int(200 / cfg.freq_s)))
            slope_t = np.random.uniform(0.01, 0.05)  # temp increases slowly
            slope_p = np.random.uniform(-0.001, 0.001)
            _inject_drift(temperature, idx, dur, slope_t)
            _inject_drift(pressure, idx, dur, slope_p)
            anomaly[idx: idx + dur] = 1
            anomaly_type[idx: idx + dur] = "drift"

        elif typ == "stuck":
            dur = np.random.randint(max(5, int(5 / cfg.freq_s)), max(50, int(500 / cfg.freq_s)))
            _inject_stuck(humidity, idx, dur)
            anomaly[idx: idx + dur] = 1
            anomaly_type[idx: idx + dur] = "stuck"

    df = pd.DataFrame(
        {
            "timestamp": time_index,
            "machine_id": machine_id,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "humidity": humidity,
            "anomaly": anomaly,
            "anomaly_type": anomaly_type,
        }
    )

    return df


def generate(cfg: GeneratorConfig):
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    dfs = []
    start_ts = pd.Timestamp.now().floor("S")
    for mid in range(cfg.n_machines):
        dfm = generate_for_machine(mid, cfg, start_ts + pd.Timedelta(seconds=mid))
        if not dfm.empty:
            dfs.append(dfm)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    # shuffle rows so consecutive rows aren't all the same machine
    out = out.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser(description="Synthetic sensor data generator")
    p.add_argument("--out", dest="out", default="data/synthetic_sensor_data.csv", help="CSV output path")
    p.add_argument("--n-machines", dest="n_machines", type=int, default=10)
    p.add_argument("--hours", dest="hours", type=float, default=1.0)
    p.add_argument("--freq-s", dest="freq_s", type=int, default=1, help="sampling frequency in seconds")
    p.add_argument("--seed", dest="seed", type=int, default=42)
    args = p.parse_args()

    cfg = GeneratorConfig(n_machines=args.n_machines, hours=args.hours, freq_s=args.freq_s, seed=args.seed, out_csv=args.out)
    df = generate(cfg)
    if df.empty:
        print("No data generated (check hours/freq settings).")
        return

    # ensure data directory exists
    import os

    os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
    df.to_csv(cfg.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {cfg.out_csv}")


if __name__ == "__main__":
    main()


