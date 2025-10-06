import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

RAW_DIR = Path("data/raw/Houston")
OUT_DIR_HOUSTON = Path("data/processed/Houston")
OUT_DIR_DALLAS = Path("data/processed/Dallas")
OUT_DIR_HOUSTON.mkdir(parents=True, exist_ok=True)
OUT_DIR_DALLAS.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE_HOUSTON = OUT_DIR_HOUSTON / "Houston.csv"
SUMMARY_FILE_DALLAS = OUT_DIR_DALLAS / "Dallas.csv"

def process_houston():
    
    summary_hou = pd.DataFrame(columns=['Date', 'min_load', 'max_load', 'avg_load'])
    summary_dal = pd.DataFrame(columns=['Date', 'min_load', 'max_load', 'avg_load'])
    for file in RAW_DIR.glob("*.xlsx"):
        print(f"Processing {file.name}")

        if "2007" in file.name:
            sheets = pd.read_excel(file, sheet_name=None, skiprows=1)
        else:
            sheets = pd.read_excel(file, sheet_name=None)
            sheets = {name: sheet for name, sheet in sheets.items() if "DRG 60 Days" not in name}
        
        # Merge all sheets into a single DataFrame
        df = pd.concat(sheets.values(), ignore_index=True)
        if 'int_kWh1' in df.columns:
            # forward-fill only NaN values across interval columns (axis=1)
            # keep PType_WZ so we can filter by region later
            df.drop(columns = ['int_kWh97', 'int_kWh98', 'int_kWh99', 'int_kWh100', 'ADDTIME'], inplace=True, errors='ignore')
            # forward-fill missing interval values row-wise (across columns)
            df.loc[:, 'int_kWh1':'int_kWh96'] = df.loc[:, 'int_kWh1':'int_kWh96'].fillna(method='ffill', axis=1)
            for i in range(0, 24):
                df['Hour_' + str(i)] = df['int_kWh' + str(4*i+1)] + df['int_kWh' + str(4*i+2)] + df['int_kWh' + str(4*i+3)] + df['int_kWh' + str(4*i+4)]
                df.drop(columns = ['int_kWh' + str(4*i+1), 'int_kWh' + str(4*i+2), 'int_kWh' + str(4*i+3), 'int_kWh' + str(4*i+4)], inplace=True, errors='ignore')
        
        if 'INT001' in df.columns:
            # forward-fill only NaN values across interval columns (axis=1)
            df.rename(columns={'ERC_TRADE_DATE': 'Date'}, inplace=True)
            df.rename(columns={"Profile Type & Weather Zone" : "PType_WZ"}, inplace=True)
            # keep PType_WZ to allow region filtering later
            df.drop(columns = ['INT097', 'INT098', 'INT099', 'INT100', 'ADDTIME'], inplace=True, errors='ignore')
            df.loc[:, 'INT001':'INT096'] = df.loc[:, 'INT001':'INT096'].fillna(method='ffill', axis=1)
            for i in range(0, 24):
                df['Hour_' + str(i)] = df['INT' + str(4*i+1).zfill(3)] + df['INT' + str(4*i+2).zfill(3)] + df['INT' + str(4*i+3).zfill(3)] + df['INT' + str(4*i+4).zfill(3)]
                df.drop(columns = ['INT' + str(4*i+1).zfill(3), 'INT' + str(4*i+2).zfill(3), 'INT' + str(4*i+3).zfill(3), 'INT' + str(4*i+4).zfill(3)], inplace=True, errors='ignore')
            
        df.groupby('Date').sum()
        
        # Sum across regions per Date for the hourly values, then compute daily stats
        hour_cols = ['Hour_' + str(i) for i in range(0, 24)]
        if 'PType_WZ' in df.columns:
            # ensure consistent uppercase for matching
            df['PType_WZ'] = df['PType_WZ'].astype(str)
            # Houston rows end with _COAST
            hou_df = df[df['PType_WZ'].str.upper().str.endswith('_COAST')]
            if not hou_df.empty:
                grouped_hou = hou_df.groupby('Date')[hour_cols].sum().reset_index()
                grouped_hou['min_load'] = grouped_hou[hour_cols].min(axis=1)
                grouped_hou['max_load'] = grouped_hou[hour_cols].max(axis=1)
                grouped_hou['avg_load'] = grouped_hou[hour_cols].mean(axis=1)
                summary_hou = pd.concat([summary_hou, grouped_hou[['Date','min_load','max_load','avg_load']]], ignore_index=True)
            # Dallas rows end with _NCENT
            dal_df = df[df['PType_WZ'].str.upper().str.endswith('_NCENT')]
            if not dal_df.empty:
                grouped_dal = dal_df.groupby('Date')[hour_cols].sum().reset_index()
                grouped_dal['min_load'] = grouped_dal[hour_cols].min(axis=1)
                grouped_dal['max_load'] = grouped_dal[hour_cols].max(axis=1)
                grouped_dal['avg_load'] = grouped_dal[hour_cols].mean(axis=1)
                summary_dal = pd.concat([summary_dal, grouped_dal[['Date','min_load','max_load','avg_load']]], ignore_index=True)
        else:
            logging.warning('No PType_WZ column in %s — cannot split Houston/Dallas', file.name)

        # cleanup
        df.drop(columns = hour_cols, inplace=True, errors='ignore')
        df.reset_index(inplace=True, drop=True)
        
    # finalize and write summaries
    if not summary_hou.empty:
        summary_hou['Date'] = pd.to_datetime(summary_hou['Date'])
        summary_hou.sort_values('Date', inplace=True)
        summary_hou.to_csv(SUMMARY_FILE_HOUSTON, index=False)
    else:
        SUMMARY_FILE_HOUSTON.write_text('')

    if not summary_dal.empty:
        summary_dal['Date'] = pd.to_datetime(summary_dal['Date'])
        summary_dal.sort_values('Date', inplace=True)
        summary_dal.to_csv(SUMMARY_FILE_DALLAS, index=False)
    else:
        SUMMARY_FILE_DALLAS.write_text('')

process_houston()


def check_missing_dates(summary_path: Path, name: str):
    if not summary_path.exists():
        print(f"No summary file for {name} at {summary_path}")
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        print(f"Summary file for {name} is empty: {summary_path}")
        return
    df['Date'] = pd.to_datetime(df['Date'])
    all_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
    missing_dates = all_dates.difference(df['Date'])
    missing_years = missing_dates.year.value_counts().sort_index()
    print(f"Missing dates by year for {name}:")
    if missing_years.empty:
        print("  None — no missing dates")
    else:
        print(missing_years.to_string())


check_missing_dates(SUMMARY_FILE_HOUSTON, 'Houston')
check_missing_dates(SUMMARY_FILE_DALLAS, 'Dallas')