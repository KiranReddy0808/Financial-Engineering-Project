"""
Download ETF prices for energy commodities used in electricity production
Focus on: Natural Gas, Coal, Uranium, and Renewable Energy
These are the primary fuel sources for power generation in the US
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

# Define ETFs for electricity production fuel sources
# US Electricity Generation Mix (2023):
# - Natural Gas: ~43%
# - Coal: ~16%
# - Nuclear: ~18%
# - Renewables: ~21%
# - Other: ~2%

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
    """
    Download historical data for a single ETF
    
    Args:
        ticker: ETF ticker symbol
        start_date: Start date for data download
        end_date: End date for data download
        description: Description of the ETF
    
    Returns:
        DataFrame with OHLCV data
    """
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
    """
    Download data for all energy ETFs
    
    Args:
        start_date: Start date (default: 2014-01-01 to match other data)
        end_date: End date (default: today)
    
    Returns:
        Dictionary of DataFrames, one per ETF
    """
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
    """
    Save ETF data to CSV files
    
    Args:
        all_data: Dictionary of individual ETF DataFrames
        combined_df: Combined DataFrame with all ETFs
        output_dir: Directory to save the data
    """
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


def print_etf_info():
    """
    Print information about available ETFs by category
    """
    print("\n" + "=" * 80)
    print("ELECTRICITY GENERATION FUEL SOURCE ETFs")
    print("=" * 80)
    print("\nUS Electricity Generation Mix (2023):")
    print("  Natural Gas: ~43%")
    print("  Nuclear: ~18%")
    print("  Coal: ~16%")
    print("  Renewables (Solar, Wind, Hydro): ~21%")
    print("  Other: ~2%")
    print("=" * 80)
    
    categories = get_category_etfs()
    
    for category, tickers in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for ticker in tickers:
            description = ENERGY_ETFS.get(ticker, 'Unknown')
            print(f"  {ticker:6s} - {description}")
    
    print("\n" + "=" * 80)
    print("WHY THESE ETFs?")
    print("=" * 80)
    print("These ETFs track the primary fuels and companies involved in electricity")
    print("generation. Their prices correlate with electricity production costs and")
    print("can help predict electricity load patterns and wholesale power prices.")
    print("=" * 80)


if __name__ == "__main__":
    # Print ETF information
    print_etf_info()
    
    # Download data
    # Default: 2014-01-01 to present (to match your load data timeframe)
    all_data, combined_df = download_all_energy_etfs(
        start_date='2014-01-01',
        end_date=None  # Today
    )
    
    # Save data
    if all_data:
        save_etf_data(all_data, combined_df)
        
        print("\n" + "=" * 80)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 80)
        print("\nData saved to: ../data/raw/energy_etfs/")
        print("\nNext steps:")
        print("  1. Check the etf_summary.csv for overview of all ETFs")
        print("  2. Individual ETF data in separate CSV files")
        print("  3. Combined data in all_energy_etfs.csv")
        print("  4. Use this data to correlate with electricity load patterns")
    else:
        print("\n⚠️  No data was downloaded. Check your internet connection.")
