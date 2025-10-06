#!/usr/bin/env python3
"""
Process NYISO palIntegrated CSVs into per-day NY load files and a daily summary.

For each raw CSV (e.g., data/raw/palIntegrated/20050101palIntegrated.csv) the
script will:
 - read the file
 - parse timestamps and the "Integrated Load" column
 - sum the load across all regions for each timestamp (total NY load time series)
 - write a per-day timeseries CSV to data/processed/palIntegrated/<YYYYMMDD>_ny_total.csv
 - append a row to data/processed/palIntegrated/daily_summary.csv with mean, sum,
   interval count and a simple sanity check

Sanity checks performed:
 - Determine interval length (minutes) from timestamps and infer expected
   intervals per day (24 for 60-min, 96 for 15-min). Check actual count matches.
 - Check for negative or missing load values.

Usage:
  python scripts/process_palintegrated.py

"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta


RAW_DIR = Path("data/raw/palIntegrated")
OUT_DIR = Path("data/processed/palIntegrated")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT_DIR / "daily_summary.csv"


def infer_interval_minutes(idx: pd.DatetimeIndex) -> int:
    if len(idx) < 2:
        return 24 * 60
    deltas = np.diff(idx.astype('int64')) // 1_000_000_000  # seconds
    median_sec = int(np.median(deltas))
    return max(1, median_sec // 60)


def expected_intervals_per_day(interval_minutes: int) -> int:
    return (24 * 60) // interval_minutes


def process_file(fp: Path) -> dict:
    logging.info("Processing %s", fp)
    # Read CSV; header likely contains 'Time Stamp' and 'Integrated Load'
    df = pd.read_csv(fp)
    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Find the timestamp and load columns
    ts_col = None
    load_col = None
    for c in df.columns:
        if c.lower().startswith('time'):
            ts_col = c
        if 'integrated' in c.lower() and 'load' in c.lower():
            load_col = c
    if ts_col is None or load_col is None:
        raise ValueError(f"Could not find timestamp/load columns in {fp}")

    # Parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col])

    # Ensure load numeric
    df[load_col] = pd.to_numeric(df[load_col], errors='coerce')

    # Sum across regions for each timestamp
    df_sum = df.groupby(ts_col)[load_col].sum().sort_index()
    df_sum.index.name = 'Time Stamp'
    # Determine date for this file (use first timestamp's date)
    date = df_sum.index[0].date()
    date_str = date.strftime('%Y%m%d')

    # Output time series file
    out_ts_file = OUT_DIR / f"{date_str}_ny_total.csv"
    df_sum.rename('Total Load').to_frame().to_csv(out_ts_file)

    # Summary statistics
    interval_minutes = infer_interval_minutes(df_sum.index)
    expected = expected_intervals_per_day(interval_minutes)
    actual = len(df_sum)
    mean_total = float(df_sum.mean())
    sum_total = float(df_sum.sum())
    has_negative = bool((df_sum < 0).any())
    has_nan = bool(df_sum.isna().any())
    sanity_pass = (actual == expected) and (not has_nan) and (not has_negative)
    note_parts = []
    if actual != expected:
        note_parts.append(f"intervals={actual} expected={expected} (interval_min={interval_minutes})")
    if has_nan:
        note_parts.append("contains NaN")
    if has_negative:
        note_parts.append("contains negative values")
    note = "; ".join(note_parts) if note_parts else "OK"

    summary_row = {
        'date': date.isoformat(),
        'file': fp.name,
        'interval_minutes': interval_minutes,
        'expected_intervals': expected,
        'actual_intervals': actual,
        'mean_total': mean_total,
        'sum_total': sum_total,
        'sanity_pass': sanity_pass,
        'note': note,
    }

    return summary_row


def append_summary(row: dict):
    df_row = pd.DataFrame([row])
    if SUMMARY_FILE.exists():
        df_row.to_csv(SUMMARY_FILE, mode='a', header=False, index=False)
    else:
        df_row.to_csv(SUMMARY_FILE, index=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    files = sorted(RAW_DIR.glob('*palIntegrated*.csv'))
    if not files:
        logging.info('No palIntegrated CSV files found under %s', RAW_DIR)
        return 0
    logging.info('Found %d files', len(files))
    results = []
    for fp in files:
        try:
            summary = process_file(fp)
            append_summary(summary)
            results.append(summary)
            logging.info('Processed %s -> mean %.2f sum %.2f intervals %d', fp.name, summary['mean_total'], summary['sum_total'], summary['actual_intervals'])
        except Exception as exc:
            logging.exception('Failed to process %s: %s', fp, exc)

    # Simple sanity check across processed days: check mean_total reasonable (no NaNs)
    df_summary = pd.read_csv(SUMMARY_FILE)
    if df_summary['mean_total'].isna().any():
        logging.warning('Some days have NaN mean_total')
    # Basic range check: mean_total should be positive and not extremely large
    if (df_summary['mean_total'] <= 0).any():
        logging.warning('Some days have non-positive mean_total')
    if (df_summary['mean_total'] > 1e7).any():
        logging.warning('Some days have very large mean_total (>1e7)')

    logging.info('Processing complete. Summary written to %s', SUMMARY_FILE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
