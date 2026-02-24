"""Create windowed/aggregated features from per-timestamp sensor CSV for classification.

Generates one row per window per machine with aggregated stats (mean, std) and a binary label.

Usage:
  from scripts.windowed_features import create_windowed_features
  create_windowed_features('data/synthetic_sensor_data.csv', 'data/windowed_features.csv', window_s=60, step_s=30)
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd


def _window_aggregate(df_machine: pd.DataFrame, window_s: int, step_s: int, features: List[str]):
    df_machine = df_machine.sort_values('timestamp')
    timestamps = pd.to_datetime(df_machine['timestamp']).values
    if len(timestamps) == 0:
        return []

    start = timestamps[0]
    end = timestamps[-1]
    # use numpy datetime64 seconds to slide windows
    start_s = np.datetime64(start, 's')
    end_s = np.datetime64(end, 's')

    results = []
    cur = start_s
    while cur <= end_s:
        window_end = cur + np.timedelta64(window_s, 's')
        mask = (timestamps >= cur) & (timestamps < window_end)
        if mask.sum() > 0:
            seg = df_machine.iloc[mask.nonzero()[0]]
            row = {
                'window_start': pd.Timestamp(cur).isoformat(),
                'n_samples': len(seg),
            }
            for f in features:
                vals = seg[f].values
                row[f'_mean_{f}'] = float(vals.mean())
                row[f'_std_{f}'] = float(vals.std())

            # label window positive if any anomaly occurred in the window
            row['label'] = int(seg['anomaly'].astype(int).sum() > 0)
            results.append(row)

        cur = cur + np.timedelta64(step_s, 's')

    return results


def create_windowed_features(in_csv: str, out_csv: str, window_s: int = 60, step_s: int = 30, features: List[str] = None):
    if features is None:
        features = ['temperature', 'vibration', 'pressure', 'humidity']

    df = pd.read_csv(in_csv)
    # normalize column names (strip whitespace) to be robust to CSV formatting
    df.columns = df.columns.str.strip()
    if 'timestamp' not in df.columns:
        raise ValueError('input CSV must contain timestamp column')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    rows = []
    for mid, g in df.groupby('machine_id'):
        agg = _window_aggregate(g, window_s, step_s, features)
        for r in agg:
            r['machine_id'] = int(mid)
        rows.extend(agg)

    if not rows:
        print('No windows generated.')
        return

    out = pd.DataFrame(rows)
    # order columns: machine_id, window_start, n_samples, features..., label
    cols = ['machine_id', 'window_start', 'n_samples'] + [c for c in out.columns if c.startswith('_mean_') or c.startswith('_std_')] + ['label']
    cols = [c for c in cols if c in out.columns]
    out = out[cols]
    out.to_csv(out_csv, index=False)
    print(f'Wrote {len(out)} windows to {out_csv}')


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_csv', required=True)
    p.add_argument('--out', dest='out_csv', required=True)
    p.add_argument('--window-s', dest='window_s', type=int, default=60)
    p.add_argument('--step-s', dest='step_s', type=int, default=30)
    args = p.parse_args()
    create_windowed_features(args.in_csv, args.out_csv, args.window_s, args.step_s)


if __name__ == '__main__':
    _cli()
