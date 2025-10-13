#!/usr/bin/env python3
"""
Convert MISO .xls/.xlsx reports into CSV or .xlsx files.

Scans an input directory (default: data/raw/miso/daily) for .xls/.xlsx files and
converts them into CSVs (one file per sheet) or into unified .xlsx workbooks.

Usage:
  python scripts/convert_miso_reports.py --format csv
  python scripts/convert_miso_reports.py --in-dir data/raw/miso/daily --out-dir data/raw/miso/converted_xlsx --format xlsx
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Dict

import pandas as pd
from tqdm import tqdm


DEFAULT_IN = Path("data/raw/miso/daily")
DEFAULT_OUT_CSV = Path("data/raw/miso/converted_csv")
DEFAULT_OUT_XLSX = Path("data/raw/miso/converted_xlsx")


def sanitize_sheet_name(name: str) -> str:
    # remove characters not allowed in filenames
    return re.sub(r"[^A-Za-z0-9_\- ]", "_", name)


def convert_file_to_csv(in_path: Path, out_dir: Path, skip_existing: bool = True) -> int:
    """Read all sheets and write each as separate CSV. Returns number written."""
    written = 0
    try:
        x = pd.read_excel(in_path, sheet_name=None)
    except Exception as exc:
        logging.exception("Failed to read %s: %s", in_path, exc)
        return 0
    for sheet_name, df in x.items():
        safe_name = sanitize_sheet_name(sheet_name)[:120]
        out_name = out_dir / f"{in_path.stem}__{safe_name}.csv"
        out_name.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and out_name.exists() and out_name.stat().st_size > 0:
            logging.debug("Skipping existing %s", out_name)
            continue
        try:
            df.to_csv(out_name, index=False)
            written += 1
            logging.info("Wrote %s", out_name)
        except Exception:
            logging.exception("Failed to write %s", out_name)
    return written


def convert_file_to_xlsx(in_path: Path, out_dir: Path, skip_existing: bool = True) -> int:
    """Read all sheets and write into an .xlsx workbook with same sheet names."""
    out_path = out_dir / f"{in_path.stem}.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        logging.debug("Skipping existing %s", out_path)
        return 0
    try:
        x = pd.read_excel(in_path, sheet_name=None)
    except Exception as exc:
        logging.exception("Failed to read %s: %s", in_path, exc)
        return 0
    try:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for sheet_name, df in x.items():
                safe_name = sanitize_sheet_name(sheet_name)[:31]
                try:
                    df.to_excel(writer, sheet_name=safe_name, index=False)
                except Exception:
                    logging.exception("Failed to write sheet %s in %s", sheet_name, out_path)
        logging.info("Wrote %s", out_path)
        return 1
    except Exception:
        logging.exception("Failed to write workbook %s", out_path)
        return 0


def find_excel_files(in_dir: Path):
    if not in_dir.exists():
        return []
    return sorted(in_dir.glob("**/*.*"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MISO xls/xlsx files to csv or xlsx")
    p.add_argument("--in-dir", type=str, default=str(DEFAULT_IN), help="Input directory")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default based on --format)")
    p.add_argument("--format", choices=["csv", "xlsx"], default="csv", help="Output format")
    p.add_argument("--skip-existing", action="store_true", help="Skip files if output exists")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s: %(message)s')
    in_dir = Path(args.in_dir)
    if args.format == 'csv':
        out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_CSV
    else:
        out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_XLSX

    files = [p for p in find_excel_files(in_dir) if p.suffix.lower() in ['.xls', '.xlsx']]
    if not files:
        logging.info('No excel files found in %s', in_dir)
        return 0

    total_written = 0
    for p in tqdm(files, desc='files'):
        try:
            if args.format == 'csv':
                total_written += convert_file_to_csv(p, out_dir, skip_existing=args.skip_existing)
            else:
                total_written += convert_file_to_xlsx(p, out_dir, skip_existing=args.skip_existing)
        except Exception:
            logging.exception('Error processing %s', p)

    logging.info('Done. Converted %d outputs from %d files', total_written, len(files))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
