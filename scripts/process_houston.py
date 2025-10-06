import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

RAW_DIR = Path("C:/Users/saiki/OneDrive/Documents/GitHub/Financial-Engineering-Project/data/raw/Houston")
OUT_DIR = Path("C:/Users/saiki/OneDrive/Documents/GitHub/Financial-Engineering-Project/data/processed/Houston")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT_DIR / "Houston.csv"

def process_houston():
    count = 0
    for file in RAW_DIR.glob("*.xlsx"):
        logging.info(f"Processing {file.name}")

        df = pd.read_excel(file)
        
        if count == 0:
            print(df.head())
        count += 1

process_houston()