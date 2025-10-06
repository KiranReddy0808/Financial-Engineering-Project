import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

RAW_DIR = Path("data/raw/palIntegrated")
OUT_DIR = Path("data/processed/NY")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT_DIR / "NewYork.csv"

def process_file(fp: Path) -> dict:
    summary = pd.DataFrame(columns=['date', 'avg_load', 'max_load', 'min_load'])
    for file in RAW_DIR.glob("*.csv"):
        contents = pd.read_csv(file)
        # Convert Time Stamp to datetime for proper grouping
        contents['Time Stamp'] = pd.to_datetime(contents['Time Stamp'])
        contents = contents[(contents['Name'] == 'N.Y.C.') | (contents['Name'] == 'LONGIL') | (contents['Name'] == 'N.Y.C._LONGIL')]

        # Group by Time Stamp and sum the 'Integrated Load' column
        contents = contents.groupby('Time Stamp', as_index=False).agg({'Integrated Load': 'sum'})

        date = contents['Time Stamp'].dt.date.iloc[0]
        avg_load = contents['Integrated Load'].mean()
        max_load = contents['Integrated Load'].max()
        min_load = contents['Integrated Load'].min()
        
        data = {
            'date': date,
            'avg_load': avg_load,
            'max_load': max_load,
            'min_load': min_load
        }

        # Append the data to the summary DataFrame
        summary = pd.concat([summary, pd.DataFrame([data])], ignore_index=True)

    # Write the summary DataFrame to a CSV file
    summary.to_csv(SUMMARY_FILE, index=False)
    content = pd.read_csv(SUMMARY_FILE)
    content.sort_values(by='date', inplace=True)
    content.to_csv(SUMMARY_FILE, index=False)
    
process_file(RAW_DIR)

