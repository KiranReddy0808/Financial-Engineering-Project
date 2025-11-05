from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import List, Optional

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
    try:
        df = pd.read_csv(file, header=None, dtype=str, keep_default_na=False, low_memory=False)
        return df
    except Exception:
        return pd.read_csv(file, header=None, low_memory=False)


def detect_layout_and_slice(df_all: pd.DataFrame, file: Path):
    candidates = [4, 5]
    max_row = df_all.shape[0]
    for idx in candidates:
        if idx + 25 <= max_row - 1:
            date_row = df_all.iloc[idx + 1].tolist()
            parsed = parse_date_from_row(date_row)
            if parsed is not None and not pd.isna(parsed):
                return idx, df_all.iloc[idx + 1], df_all.iloc[idx + 2: idx + 26].reset_index(drop=True)

    for idx in range(0, min(12, max_row - 2)):
        if idx + 25 <= max_row - 1:
            date_row = df_all.iloc[idx + 1].tolist()
            parsed = parse_date_from_row(date_row)
            if parsed is not None and not pd.isna(parsed):
                return idx, df_all.iloc[idx + 1], df_all.iloc[idx + 2: idx + 26].reset_index(drop=True)

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
    
    header = [str(x) for x in df_all.iloc[header_idx].tolist()]
    parsed_date = parse_date_from_row(date_row_series.tolist() if date_row_series is not None else [])
    if parsed_date is None:
        m = re.search(r"(\d{8})", file.name)
        if m:
            try:
                parsed_date = pd.to_datetime(m.group(1), format='%Y%m%d')
            except Exception:
                parsed_date = None

    ncols = min(len(header), data24.shape[1])
    for j in range(ncols):
        col_name = header[j]
        if col_name in METADATA_COLS or not str(col_name).strip():
            continue
        series = pd.to_numeric(data24.iloc[:, j], errors='coerce')
        count_valid = int(series.notna().sum())
        min_v = series.min(skipna=True)
        max_v = series.max(skipna=True)
        avg_v = (min_v + max_v) / 2 if pd.notna(min_v) and pd.notna(max_v) else float('nan')
        if count_valid == 0:
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
