from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import requests
from requests.exceptions import RequestException
from tqdm import tqdm


BASE_URL = "https://mis.nyiso.com/public/csv/palIntegrated"
DEFAULT_OUT_DIR = Path("data/raw/palIntegrated")
MIN_DATE = datetime(2005, 1, 1)
MAX_DATE = datetime(2024, 12, 31)


def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_url_for_date(d: datetime) -> str:
    date_str = d.strftime("%Y%m%d")
    return f"{BASE_URL}/{date_str}palIntegrated_csv.zip"


def download_with_retries(url: str, out_path: Path, max_retries: int = 3, backoff: float = 1.0, timeout: int = 30) -> bool:
    """Download url to out_path. Returns True on success, False otherwise."""
    headers = {"User-Agent": "nyiso-downloader/1.0 (+https://github.com)"}
    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 404:
                    logging.debug("Not found: %s", url)
                    return False
                r.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # Write to a temp file then rename
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")
                with tmp_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                tmp_path.replace(out_path)
                return True
        except RequestException as exc:
            attempt += 1
            wait = backoff * (2 ** (attempt - 1))
            logging.warning("Request failed (attempt %d/%d) for %s: %s — retrying in %.1fs", attempt, max_retries, url, exc, wait)
            time.sleep(wait)
    logging.error("Failed to download after %d attempts: %s", max_retries, url)
    return False


def is_valid_date(d: datetime) -> bool:
    return MIN_DATE <= d <= MAX_DATE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download NYISO palIntegrated CSV zip files for a date range.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", type=str, help="Single date to download (YYYY-MM-DD)")
    group.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, help="End date (YYYY-MM-DD). Required when --start is used.")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory (default: data/raw/palIntegrated)")
    p.add_argument("--skip-existing", action="store_true", help="Skip download if file already exists")
    p.add_argument("--max-retries", type=int, default=3, help="Max number of retries for each file")
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep between downloads (default 0.25)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.out_dir)

    dates = []
    if args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d")
        if not is_valid_date(d):
            logging.error("Date %s out of allowed range: %s - %s", args.date, MIN_DATE.date(), MAX_DATE.date())
            return 2
        dates = [d]
    else:
        if not args.end:
            logging.error("--end is required when --start is provided")
            return 2
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        if start > end:
            logging.error("start date must be <= end date")
            return 2
        if not is_valid_date(start) or not is_valid_date(end):
            logging.error("Date range must be within %s - %s", MIN_DATE.date(), MAX_DATE.date())
            return 2
        dates = list(daterange(start, end))

    logging.info("Downloading %d dates to %s", len(dates), out_dir)
    success_count = 0
    fail_count = 0
    for d in tqdm(dates, desc="dates"):
        date_str = d.strftime("%Y%m%d")
        url = build_url_for_date(d)
        out_path = out_dir / f"{date_str}palIntegrated_csv.zip"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            logging.info("Skipping existing: %s", out_path.name)
            success_count += 1
            time.sleep(args.sleep)
            continue
        ok = download_with_retries(url=url, out_path=out_path, max_retries=args.max_retries, backoff=1.0)
        if ok:
            success_count += 1
        else:
            fail_count += 1
        time.sleep(args.sleep)

    logging.info("Done. Success: %d, Failed: %d", success_count, fail_count)
    return 0 if fail_count == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
