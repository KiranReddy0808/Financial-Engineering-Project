#!/usr/bin/env python3
"""
Extract NYISO palIntegrated zip files.

Finds zip files (default: data/raw/palIntegrated/*.zip) and extracts any CSV files
into an `extracted/<zip-stem>/` directory under the input folder (default
`data/raw/palIntegrated/extracted`). Supports --skip-existing, --overwrite and --dry-run.

Example:
  python scripts/extract_palintegrated.py
  python scripts/extract_palintegrated.py --in-dir data/raw/palIntegrated --overwrite
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import zipfile
import tempfile
import shutil
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract palIntegrated zip files to extracted folder")
    p.add_argument("--in-dir", type=str, default="data/raw/palIntegrated", help="Directory containing downloaded zip files")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <in-dir>/extracted)")
    p.add_argument("--pattern", type=str, default="*.zip", help="Glob pattern for zip files (default: *.zip)")
    p.add_argument("--skip-existing", action="store_true", help="Skip extracting files that already exist")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted files")
    p.add_argument("--dry-run", action="store_true", help="Show what would be extracted without writing files")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def find_zip_files(in_dir: Path, pattern: str) -> List[Path]:
    if not in_dir.exists():
        return []
    return sorted(in_dir.glob(pattern))


def safe_extract_zip(zip_path: Path, dest_dir: Path, skip_existing: bool = True, overwrite: bool = False, dry_run: bool = False) -> List[Path]:
    """Extract CSV files from zip_path into dest_dir (creates dest_dir).
    Returns list of extracted file paths (or would-be paths if dry_run).
    """
    extracted_files: List[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            # Normalize member name
            if member.endswith('/'):
                continue
            name = Path(member).name
            if not name.lower().endswith('.csv'):
                # skip non-csv files
                continue
            out_path = dest_dir / name
            if out_path.exists() and not overwrite:
                if skip_existing:
                    logging.info('Skipping existing %s', out_path)
                    extracted_files.append(out_path)
                    continue
                # if not skip and not overwrite, still skip to avoid accidental clobber
                logging.info('Not overwriting existing %s (use --overwrite to force)', out_path)
                extracted_files.append(out_path)
                continue
            if dry_run:
                extracted_files.append(out_path)
                continue
            # Extract safely: write to a temporary file then move into place
            with tempfile.NamedTemporaryFile(delete=False, dir=str(dest_dir)) as tmpfh:
                tmpname = Path(tmpfh.name)
                with zf.open(member) as src:
                    shutil.copyfileobj(src, tmpfh)
            # move temp file to final location (replace if overwrite)
            tmpname.replace(out_path)
            extracted_files.append(out_path)
            logging.info('Extracted %s -> %s', member, out_path)
    return extracted_files


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s: %(message)s')
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / 'extracted'
    zips = find_zip_files(in_dir, args.pattern)
    if not zips:
        logging.warning('No zip files found in %s matching %s', in_dir, args.pattern)
        return 0
    logging.info('Found %d zip files', len(zips))
    overall_extracted = 0
    for z in zips:
        # create a folder per zip stem (e.g., 20250701palIntegrated_csv)
        stem = z.stem
        dest_for_zip = out_dir / stem
        logging.info('Processing %s -> %s', z.name, dest_for_zip)
        try:
            extracted = safe_extract_zip(z, dest_for_zip, skip_existing=args.skip_existing, overwrite=args.overwrite, dry_run=args.dry_run)
            overall_extracted += len([p for p in extracted if p and (args.dry_run or p.exists())])
        except zipfile.BadZipFile:
            logging.error('Bad zip file: %s', z)
        except Exception as exc:
            logging.exception('Error extracting %s: %s', z, exc)

    logging.info('Done. Extracted (or would extract in dry-run) %d files', overall_extracted)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
