import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os


# Select ONE representative ETF per category (6 total)
ENERGY_ETFS = {
    # Natural Gas (Primary fuel - ~43% of US generation)
    'UNG': 'United States Natural Gas Fund',
    
    # Coal (Secondary fuel - ~16% of US generation)
    'KOL': 'VanEck Coal ETF',
    
    # Nuclear/Uranium (~18% of US generation)
    'URA': 'Global X Uranium ETF',
    
    # Renewables (~21% of US generation - solar, wind, hydro)
    'ICLN': 'iShares Global Clean Energy ETF',
    
    # Electric Utilities (Power generation companies)
    'XLU': 'Utilities Select Sector SPDR Fund',
    
    # Oil (Backup/peak generation - ~1% but important for price signals)
    'USO': 'United States Oil Fund',
}

def download_etf_data(ticker, start_date, end_date, description):
    try:
        print(f"Downloading {ticker}: {description}...")
        
        # Download data
        etf = yf.Ticker(ticker)
        df = etf.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"  ⚠️  No data available for {ticker}")
            return None
        
        # Add ticker column
        df['Ticker'] = ticker
        df['Description'] = description
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        print(f"  ✓ Downloaded {len(df)} days of data for {ticker}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {str(e)}")
        return None


def download_all_energy_etfs(start_date='2014-01-01', end_date=None):

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 80)
    print("DOWNLOADING ELECTRICITY GENERATION FUEL ETF DATA")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total ETFs: {len(ENERGY_ETFS)}")
    print("Focus: Natural Gas, Coal, Nuclear, Renewables, Utilities\n")
    
    all_data = {}
    combined_data = []
    
    for ticker, description in ENERGY_ETFS.items():
        df = download_etf_data(ticker, start_date, end_date, description)
        
        if df is not None:
            all_data[ticker] = df
            combined_data.append(df)
    
    print("\n" + "=" * 80)
    print(f"DOWNLOAD COMPLETE: {len(all_data)}/{len(ENERGY_ETFS)} ETFs downloaded successfully")
    print("=" * 80)
    
    # Combine all data
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        return all_data, combined_df
    else:
        return all_data, None


def save_etf_data(all_data, combined_df, output_dir='../data/raw/energy_etfs'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING DATA")
    print("=" * 80)
    
    # Save individual ETF files
    for ticker, df in all_data.items():
        filename = f"{output_dir}/{ticker}.csv"
        df.to_csv(filename, index=False)
        print(f"✓ Saved {ticker}: {filename}")
    
    # Save combined file
    if combined_df is not None:
        combined_filename = f"{output_dir}/all_energy_etfs.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"\n✓ Saved combined data: {combined_filename}")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    # Create a summary file
    summary_data = []
    for ticker, description in ENERGY_ETFS.items():
        if ticker in all_data:
            df = all_data[ticker]
            summary_data.append({
                'Ticker': ticker,
                'Description': description,
                'Start_Date': df['Date'].min(),
                'End_Date': df['Date'].max(),
                'Days': len(df),
                'Avg_Volume': df['Volume'].mean(),
                'Latest_Price': df['Close'].iloc[-1] if len(df) > 0 else None,
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{output_dir}/etf_summary.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"\n✓ Saved summary: {summary_filename}")
    
    print("=" * 80)


def get_category_etfs():
    """
    Return ETFs grouped by electricity generation fuel type
    """
    categories = {
        'Natural Gas (~43% of US electricity)': ['UNG'],
        'Coal (~16% of US electricity)': ['KOL'],
        'Nuclear/Uranium (~18% of US electricity)': ['URA'],
        'Renewables (~21% of US electricity)': ['ICLN'],
        'Electric Utilities (Power Companies)': ['XLU'],
        'Oil (Backup/Peak Generation ~1%)': ['USO'],
    }
    return categories


if __name__ == "__main__":
    
    all_data, combined_df = download_all_energy_etfs(
        start_date='2014-01-01',
        end_date=None  # Today
    )
    
    save_etf_data(all_data, combined_df)
