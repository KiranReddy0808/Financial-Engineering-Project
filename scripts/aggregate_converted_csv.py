#!/usr/bin/env python3
"""
Aggregate converted MISO CSVs into per-column daily summaries.

This script supports two common layout variants observed in the converted CSVs:
 - Variant A (pre-2023): header is on line 6 (1-based), date on line 7, values on lines 8..31 (24 rows)
 - Variant B (2023+): header is on line 5 (1-based), date on line 6, values on lines 7..30 (24 rows)

The detector will inspect the first few lines of each file and choose the header row
that produces a parseable date on the following row. If neither candidate works, it
falls back to a heuristic search within the first 12 rows.

For each column (excluding metadata columns) it computes min/max/avg/count for the
24 interval rows and appends a daily summary to data/processed/miso/columns/<col>.csv
and records first-seen column names in data/processed/miso/column_names_log.csv.
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


RAW_DIR = Path("data/raw/miso/converted_csv")
OUT_DIR = Path("data/processed/miso/columns")
LOG_FILE = Path("data/processed/miso/column_names_log.csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# columns to ignore if present in header
METADATA_COLS = {"Time Stamp", "Time Zone", "Name", "PTID", "Integrated Load"}


def sanitize(name: str) -> str:
    name = name.strip()
    # replace problematic filename chars
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name[:200]


def find_files(base: Path) -> List[Path]:
    if not base.exists():
        return []
    return sorted(base.rglob("*.csv"))


def parse_date_from_row(row: List[str]) -> Optional[pd.Timestamp]:
    for cell in row:
        if pd.isna(cell):
            continue
        s = str(cell).strip()
        # try common date formats
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d", "%Y%m%d"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        # try generic parse
        try:
            return pd.to_datetime(s, errors='coerce')
        except Exception:
            continue
    return None


def _read_all_rows(file: Path) -> pd.DataFrame:
    # read file without assuming a header so we can inspect arbitrary rows
    try:
        df = pd.read_csv(file, header=None, dtype=str, keep_default_na=False, low_memory=False)
        return df
    except Exception:
        # last-resort: try with default settings
        return pd.read_csv(file, header=None, low_memory=False)


def detect_layout_and_slice(df_all: pd.DataFrame, file: Path):
    """Return tuple (header_idx, date_row_series, data24_df) or (None, None, None) on failure.

    header_idx is 0-based index of the header row within df_all. data24_df will contain
    24 rows (header_idx+2 .. header_idx+25 inclusive).
    """
    # candidate header rows in 0-based indexing: 4 (line5) and 5 (line6)
    candidates = [4, 5]
    max_row = df_all.shape[0]
    for idx in candidates:
        if idx + 25 <= max_row - 1:
            # date row is next row
            date_row = df_all.iloc[idx + 1].tolist()
            parsed = parse_date_from_row(date_row)
            if parsed is not None and not pd.isna(parsed):
                return idx, df_all.iloc[idx + 1], df_all.iloc[idx + 2: idx + 26].reset_index(drop=True)

    # fallback: search within first 12 rows for a header that yields a parseable date next
    for idx in range(0, min(12, max_row - 2)):
        if idx + 25 <= max_row - 1:
            date_row = df_all.iloc[idx + 1].tolist()
            parsed = parse_date_from_row(date_row)
            if parsed is not None and not pd.isna(parsed):
                return idx, df_all.iloc[idx + 1], df_all.iloc[idx + 2: idx + 26].reset_index(drop=True)

    # give up
    return None, None, None


def append_column_summary(col_name: str, date: pd.Timestamp, min_v, max_v, avg_v, count_valid: int, src_file: str):
    out_file = OUT_DIR / f"{sanitize(col_name)}.csv"
    row = {
        'Date': pd.to_datetime(date).date().isoformat() if date is not None else '',
        'min': float(min_v) if pd.notna(min_v) else '',
        'max': float(max_v) if pd.notna(max_v) else '',
        'avg': float(avg_v) if pd.notna(avg_v) else '',
    }
    # if file exists, append but avoid duplicate date
    if out_file.exists():
        try:
            existing = pd.read_csv(out_file)
            if 'Date' in existing.columns and row['Date'] in existing['Date'].astype(str).values:
                return
        except Exception:
            pass
    df_row = pd.DataFrame([row])
    if out_file.exists():
        df_row.to_csv(out_file, mode='a', header=False, index=False)
    else:
        df_row.to_csv(out_file, index=False)


def log_column_name(col_name: str, file: Path, date: Optional[pd.Timestamp]):
    # create log if missing
    entry = {'column': col_name, 'first_seen_file': str(file), 'first_seen_date': pd.to_datetime(date).date().isoformat() if date is not None else ''}
    if LOG_FILE.exists():
        try:
            log_df = pd.read_csv(LOG_FILE)
            if col_name in log_df['column'].astype(str).values:
                return
            log_df = pd.concat([log_df, pd.DataFrame([entry])], ignore_index=True)
            log_df.to_csv(LOG_FILE, index=False)
        except Exception:
            # fallback append
            pd.DataFrame([entry]).to_csv(LOG_FILE, mode='a', header=not LOG_FILE.exists(), index=False)
    else:
        pd.DataFrame([entry]).to_csv(LOG_FILE, index=False)


def process_file(file: Path):
    try:
        df_all = _read_all_rows(file)
    except Exception as exc:
        logging.exception('Failed to read file %s: %s', file, exc)
        return

    header_idx, date_row_series, data24 = detect_layout_and_slice(df_all, file)
    if header_idx is None:
        logging.warning('Could not detect layout for %s, skipping', file)
        return
    # header row values
    header = [str(x) for x in df_all.iloc[header_idx].tolist()]
    parsed_date = parse_date_from_row(date_row_series.tolist() if date_row_series is not None else [])
    if parsed_date is None:
        # try to get date from filename (search for 8-digit date)
        m = re.search(r"(\d{8})", file.name)
        if m:
            try:
                parsed_date = pd.to_datetime(m.group(1), format='%Y%m%d')
            except Exception:
                parsed_date = None

    # data24 is already a DataFrame with the 24 rows; align columns
    # ensure numeric conversion per-column
    ncols = min(len(header), data24.shape[1])
    for j in range(ncols):
        col_name = header[j]
        if col_name in METADATA_COLS or not str(col_name).strip():
            continue
        series = pd.to_numeric(data24.iloc[:, j], errors='coerce')
        count_valid = int(series.notna().sum())
        min_v = series.min(skipna=True)
        max_v = series.max(skipna=True)
        avg_v = series.mean(skipna=True)
        if count_valid == 0:
            # nothing to record
            continue
        append_column_summary(col_name, parsed_date, min_v, max_v, avg_v, count_valid, str(file))
        log_column_name(col_name, file, parsed_date)
        if count_valid != 24:
            logging.warning('File %s column %s has %d valid values (expected 24)', file, col_name, count_valid)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    files = find_files(RAW_DIR)
    if not files:
        logging.info('No converted CSV files found in %s', RAW_DIR)
        return
    for f in files:
        logging.info('Processing %s', f)
        process_file(f)


if __name__ == '__main__':
    main()
