from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterator, Tuple

import requests
from requests.exceptions import RequestException
import zipfile
import shutil


BASE_URL = "https://docs.misoenergy.org/marketreports"
OUT_BASE = Path("data/raw/miso")
MONTHLY_DIR = OUT_BASE / "monthly"
DAILY_DIR = OUT_BASE / "daily"

MIN_MONTHLY = date(2009, 7, 1)
MAX_MONTHLY = date(2022, 12, 31)
MIN_DAILY = date(2023, 1, 1)
MAX_DAILY = date(2024, 12, 31)


def month_iter(start: date, end: date) -> Iterator[Tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def daterange(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_monthly_filename(year: int, month: int) -> str:
    # e.g., 202212_rf_al_xls.zip
    return f"{year}{month:02d}_rf_al_xls.zip"


def build_daily_filename(d: date) -> str:
    # e.g., 20251006_df_al.xls
    return f"{d.strftime('%Y%m%d')}_df_al.xls"


def download(url: str, out_path: Path, max_retries: int = 3, timeout: int = 30) -> bool:
    headers = {"User-Agent": "miso-downloader/1.0"}
    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 404:
                    logging.debug("Not found: %s", url)
                    return False
                r.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(out_path)
                return True
        except RequestException as exc:
            attempt += 1
            wait = 2 ** (attempt - 1)
            logging.warning("Download failed (%d/%d) %s: %s — retrying in %ds", attempt, max_retries, url, exc, wait)
            time.sleep(wait)
    logging.error("Giving up %s after %d attempts", url, max_retries)
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MISO load monthly/daily reports")
    p.add_argument("--start", type=str, default="2009-07-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default="2024-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--out-base", type=str, default=str(OUT_BASE), help="Output base dir")
    p.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start > end:
        logging.error("start must be <= end")
        return 2

    out_base = Path(args.out_base)
    monthly_dir = out_base / "monthly"
    daily_dir = out_base / "daily"

    # Monthly range: intersection of requested range with MIN_MONTHLY..MAX_MONTHLY
    monthly_start = max(start, MIN_MONTHLY)
    monthly_end = min(end, MAX_MONTHLY)
    # Daily range: intersection of requested range with MIN_DAILY..MAX_DAILY
    daily_start = max(start, MIN_DAILY)
    daily_end = min(end, MAX_DAILY)

    total_ok = 0
    total_failed = 0

    # Download monthly files
    if monthly_start <= monthly_end:
        logging.info("Downloading monthly files from %s to %s", monthly_start, monthly_end)
        for y, m in month_iter(monthly_start, monthly_end):
            fname = build_monthly_filename(y, m)
            url = f"{BASE_URL}/{fname}"
            out_path = monthly_dir / fname
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                logging.info("Skipping existing %s", out_path.name)
                total_ok += 1
                continue
            ok = download(url, out_path, max_retries=args.max_retries)
            if ok:
                total_ok += 1
                # extract monthly zip into daily_dir and remove zip
                try:
                    with zipfile.ZipFile(out_path, 'r') as zf:
                        for member in zf.namelist():
                            # prefer csv/xls files
                            if not member.lower().endswith(('.csv', '.xls', '.xlsx')):
                                continue
                            target_name = Path(member).name
                            target_path = daily_dir / target_name
                            if args.skip_existing and target_path.exists() and target_path.stat().st_size > 0:
                                logging.debug("Skipping existing extracted %s", target_path)
                                continue
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(member) as src, target_path.open('wb') as dst:
                                shutil.copyfileobj(src, dst)
                    # remove zip after successful extraction
                    out_path.unlink()
                    logging.info('Extracted and removed zip %s', out_path)
                except zipfile.BadZipFile:
                    logging.warning('Downloaded file is not a valid zip: %s', out_path)
                except Exception as exc:
                    logging.exception('Error extracting %s: %s', out_path, exc)
            else:
                total_failed += 1

    # Download daily files
    if daily_start <= daily_end:
        logging.info("Downloading daily files from %s to %s", daily_start, daily_end)
        for d in daterange(daily_start, daily_end):
            fname = build_daily_filename(d)
            url = f"{BASE_URL}/{fname}"
            out_path = daily_dir / fname
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                logging.info("Skipping existing %s", out_path.name)
                total_ok += 1
                continue
            ok = download(url, out_path, max_retries=args.max_retries)
            if ok:
                total_ok += 1
            else:
                total_failed += 1

    logging.info("Done. Success: %d, Failed: %d", total_ok, total_failed)
    return 0 if total_failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
