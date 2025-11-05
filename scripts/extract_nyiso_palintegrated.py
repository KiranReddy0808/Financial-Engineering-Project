from __future__ import annotations

import argparse
import logging
from pathlib import Path
import zipfile
import shutil
from typing import Iterable

from tqdm import tqdm


DEFAULT_IN_DIR = Path("data/raw/palIntegrated")
DEFAULT_OUT_DIR = Path("data/raw/extracted")


def find_zip_files(in_dir: Path) -> Iterable[Path]:
    if not in_dir.exists():
        return []
    return sorted(in_dir.glob("*.zip"))


def extract_zip_to(zip_path: Path, out_dir: Path, skip_existing: bool = True, overwrite: bool = False, dry_run: bool = False) -> int:
    """Extract CSV files from zip_path into out_dir (flat).

    Returns number of files extracted (or that would be extracted in dry-run).
    """
    dest_base = out_dir
    extracted_count = 0
    if dry_run:
        logging.info("Dry run: would extract %s -> %s", zip_path.name, dest_base)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                logging.debug("No CSV members in %s", zip_path)
                return 0
            for member in members:
                # use the basename so internal paths don't get recreated
                dest_file = dest_base / Path(member).name
                if dest_file.exists() and not overwrite:
                    if skip_existing:
                        logging.debug("Skipping existing file %s", dest_file)
                        continue
                    else:
                        logging.info("Not overwriting existing %s (use --overwrite to force)", dest_file)
                        continue
                if dry_run:
                    logging.info("Would extract %s -> %s", member, dest_file)
                    extracted_count += 1
                    continue
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                # write member content to dest_file (overwrites if requested)
                with zf.open(member) as src, open(dest_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_count += 1
                logging.info("Extracted %s -> %s", member, dest_file)
    except zipfile.BadZipFile:
        logging.error("Bad zip file: %s", zip_path)
    except Exception as exc:
        logging.exception("Error extracting %s: %s", zip_path, exc)
    return extracted_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract NYISO palIntegrated zip files to extracted folder")
    p.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR), help="Input directory with zip files (default: data/raw/palIntegrated)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for extracted CSVs (default: data/raw/extracted)")
    p.add_argument("--inplace", action="store_true", help="Extract CSVs directly into the input directory (may cause name collisions)")
    p.add_argument("--skip-existing", action="store_true", help="Skip extraction if destination CSV exists")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted files")
    p.add_argument("--remove-zip", action="store_true", help="Remove zip file after successful extraction")
    p.add_argument("--dry-run", action="store_true", help="Don't write files; just show what would be done")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if args.inplace:
        out_dir = in_dir
        logging.warning("Inplace extraction enabled: CSVs will be written directly into %s — this may overwrite files from other zips.", out_dir)

    zips = find_zip_files(in_dir)
    if not zips:
        logging.info("No zip files found in %s", in_dir)
        return 0

    total_extracted = 0
    for zip_path in tqdm(zips, desc="zips"):
        logging.debug("Processing %s", zip_path)
        try:
            num = extract_zip_to(zip_path, out_dir, skip_existing=args.skip_existing, overwrite=args.overwrite, dry_run=args.dry_run)
            total_extracted += num
            if num > 0 and args.remove_zip and not args.dry_run:
                try:
                    zip_path.unlink()
                    logging.info("Removed zip %s", zip_path)
                except Exception as exc:
                    logging.warning("Failed to remove zip %s: %s", zip_path, exc)
        except Exception as exc:
            logging.exception("Failed processing %s: %s", zip_path, exc)

    logging.info("Extracted %d files from %d zip(s)", total_extracted, len(zips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
