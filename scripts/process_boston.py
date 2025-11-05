
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd


RAW_DIR = Path("data/raw/Boston")
OUT_DIR = Path("data/processed/Boston")
# Try both sheet names - older files use NEMASSBOST, newer files use NEMA
SHEET_NAMES = ["NEMASSBOST", "NEMA"]


def find_boston_files(base: Path) -> List[Path]:
    """Find all .xls and .xlsx files in Boston raw directory."""
    if not base.exists():
        return []
    xls_files = sorted(base.glob("*.xls"))
    xlsx_files = sorted(base.glob("*.xlsx"))
    return sorted(xls_files + xlsx_files)


def read_boston_file(file: Path) -> pd.DataFrame:
    """
    Read NEMASSBOST or NEMA sheet from a Boston Excel file.
    
    Returns a DataFrame with columns: Date, Hour, DA_DEMD, DEMAND
    """
    # Try each possible sheet name
    for sheet_name in SHEET_NAMES:
        try:
            df = pd.read_excel(file, sheet_name=sheet_name)
            
            # For newer files, columns may have different names
            # Map: Hr_End -> Hour, DA_Demand -> DA_DEMD, RT_Demand -> DEMAND
            if 'Hr_End' in df.columns:
                df = df.rename(columns={'Hr_End': 'Hour'})
            if 'DA_Demand' in df.columns:
                df = df.rename(columns={'DA_Demand': 'DA_DEMD'})
            if 'RT_Demand' in df.columns:
                df = df.rename(columns={'RT_Demand': 'DEMAND'})
            
            # Check if required columns exist
            required_cols = ['Date', 'Hour', 'DA_DEMD', 'DEMAND']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logging.warning(f"File {file} sheet '{sheet_name}' missing columns: {missing}")
                continue  # Try next sheet name
            
            # Keep only required columns
            df = df[required_cols].copy()
            
            # Parse date column
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Convert Hour to numeric
            df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
            
            # Convert demand columns to numeric
            df['DA_DEMD'] = pd.to_numeric(df['DA_DEMD'], errors='coerce')
            df['DEMAND'] = pd.to_numeric(df['DEMAND'], errors='coerce')
            
            # Drop rows where Date is NaT or Hour is NaN
            df = df.dropna(subset=['Date', 'Hour'])
            
            # Ensure Hour is in valid range (1-24)
            df = df[(df['Hour'] >= 1) & (df['Hour'] <= 24)]
            
            return df
            
        except ValueError as e:
            if "Worksheet named" in str(e):
                # Try next sheet name
                continue
            else:
                logging.exception(f"Error reading {file} sheet '{sheet_name}': {e}")
                return pd.DataFrame()
        except Exception as e:
            logging.exception(f"Error reading {file} sheet '{sheet_name}': {e}")
            return pd.DataFrame()
    
    # If we get here, no valid sheet was found
    logging.warning(f"File {file} does not contain any valid sheet from {SHEET_NAMES}")
    return pd.DataFrame()


def compute_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily min, max, avg for DA_DEMD and DEMAND.
    
    Returns DataFrame with columns:
      - Date
      - ActualLoad_pred_min, ActualLoad_pred_max, ActualLoad_pred_avg
      - ActualLoad_min, ActualLoad_max, ActualLoad_avg
    """
    if df.empty:
        return pd.DataFrame()
    
    # Group by Date
    grouped = df.groupby('Date').agg(
        ActualLoad_pred_min=('DA_DEMD', 'min'),
        ActualLoad_pred_max=('DA_DEMD', 'max'),
        ActualLoad_min=('DEMAND', 'min'),
        ActualLoad_max=('DEMAND', 'max'),
    ).reset_index()
    
    # Calculate avg as (min + max) / 2
    grouped['ActualLoad_pred_avg'] = (grouped['ActualLoad_pred_min'] + grouped['ActualLoad_pred_max']) / 2
    grouped['ActualLoad_avg'] = (grouped['ActualLoad_min'] + grouped['ActualLoad_max']) / 2
    
    # Format Date as YYYY-MM-DD
    grouped['Date'] = grouped['Date'].dt.date
    
    return grouped


def process_all_files(files: List[Path], output_file: Path, verbose: bool = False):
    """Process all Boston files and save combined daily summary."""
    all_summaries = []
    
    for file in files:
        if verbose:
            logging.info(f"Processing {file.name}")
        
        df = read_boston_file(file)
        if df.empty:
            logging.warning(f"Skipped {file.name} (no valid data)")
            continue
        
        summary = compute_daily_summary(df)
        if not summary.empty:
            all_summaries.append(summary)
    
    if not all_summaries:
        logging.error("No data processed from any files")
        return
    
    # Combine all summaries
    combined = pd.concat(all_summaries, ignore_index=True)
    
    # Sort by date
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Remove any duplicate dates (keep first occurrence)
    combined = combined.drop_duplicates(subset=['Date'], keep='first')
    
    # Save to output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)
    
    logging.info(f"Saved {len(combined)} daily records to {output_file}")
    
    # Print summary statistics
    date_range = f"{combined['Date'].min()} to {combined['Date'].max()}"
    logging.info(f"Date range: {date_range}")
    
    # Check for missing dates
    all_dates = pd.date_range(
        start=pd.to_datetime(combined['Date'].min()),
        end=pd.to_datetime(combined['Date'].max()),
        freq='D'
    )
    missing_dates = set(all_dates.date) - set(combined['Date'])
    if missing_dates:
        logging.warning(f"Found {len(missing_dates)} missing dates in range")
        if verbose and len(missing_dates) <= 20:
            for dt in sorted(missing_dates):
                logging.warning(f"  Missing: {dt}")


def main():
    parser = argparse.ArgumentParser(description="Process Boston NEMASSBOST load data")
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=RAW_DIR,
        help='Input directory containing Boston Excel files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=OUT_DIR / 'daily_summary.csv',
        help='Output CSV file for daily summaries'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    # Find all Boston files
    files = find_boston_files(args.input_dir)
    if not files:
        logging.error(f"No Excel files found in {args.input_dir}")
        return
    
    logging.info(f"Found {len(files)} Boston Excel files")
    
    # Process all files
    process_all_files(files, args.output, verbose=args.verbose)


if __name__ == '__main__':
    main()
