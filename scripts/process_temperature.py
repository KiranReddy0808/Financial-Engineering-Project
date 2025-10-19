"""
Process temperature data for seasonal analysis.

This script processes raw temperature data files to create clean datasets
with average temperatures suitable for seasonal modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def process_temperature_file(input_path, output_path, city_name):
    """
    Process a single temperature file.
    
    Args:
        input_path: Path to raw temperature CSV
        output_path: Path to save processed CSV
        city_name: Name of the city for reporting
    """
    print(f"Processing {city_name}...")
    
    # Read raw data
    df = pd.read_csv(input_path)
    
    # Rename columns to lowercase for consistency
    df.columns = df.columns.str.lower()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate average temperature from max and min
    if 'daily_max_temp' in df.columns and 'daily_min_temp' in df.columns:
        df['tavg'] = (df['daily_max_temp'] + df['daily_min_temp']) / 2
        df['tmax'] = df['daily_max_temp']
        df['tmin'] = df['daily_min_temp']
    elif 'tmax' in df.columns and 'tmin' in df.columns:
        df['tavg'] = (df['tmax'] + df['tmin']) / 2
    else:
        raise ValueError(f"Could not find temperature columns in {city_name} data")
    
    # Select relevant columns
    result = df[['date', 'tmin', 'tmax', 'tavg']].copy()
    
    # Remove any duplicates (keep first)
    result = result.drop_duplicates(subset='date', keep='first')
    
    # Sort by date
    result = result.sort_values('date')
    
    # Remove any rows with missing temperature data
    result = result.dropna(subset=['tmin', 'tmax', 'tavg'])
    
    # Filter to analysis period (2014-2022)
    result = result[(result['date'] >= '2014-01-01') & (result['date'] <= '2022-12-31')]
    
    # Save processed data
    result.to_csv(output_path, index=False)
    
    print(f"  ✓ Processed {len(result)} days")
    print(f"  ✓ Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"  ✓ Avg temp: {result['tavg'].mean():.2f}°F (min: {result['tmin'].min():.1f}°F, max: {result['tmax'].max():.1f}°F)")
    print(f"  ✓ Saved to: {output_path}")
    print()
    
    return result


def process_all_cities():
    """Process temperature data for all cities."""
    
    # Define cities to process
    cities = {
        'Boston': 'Boston.csv',
        'NewYork': 'NewYork.csv',
        'Houston': 'Houston.csv',
        'Chicago': 'Chicago.csv',
        'Dallas': 'Dallas.csv',
        'Minneapolis': 'Minneapolis.csv'
    }
    
    # Setup paths
    raw_dir = Path('../data/raw/temperature')
    processed_dir = Path('../data/processed/temperature')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TEMPERATURE DATA PREPROCESSING")
    print("="*80)
    print()
    
    all_data = {}
    
    for city_name, filename in cities.items():
        input_path = raw_dir / filename
        output_path = processed_dir / filename
        
        if not input_path.exists():
            print(f"⚠ Warning: {input_path} not found, skipping...")
            continue
        
        try:
            data = process_temperature_file(input_path, output_path, city_name)
            all_data[city_name] = data
        except Exception as e:
            print(f"✗ Error processing {city_name}: {e}")
            print()
    
    print("="*80)
    print(f"PROCESSING COMPLETE! Processed {len(all_data)} cities.")
    print("="*80)
    print()
    print("Summary Statistics:")
    print("-"*80)
    
    summary_data = []
    for city, data in all_data.items():
        summary_data.append({
            'City': city,
            'Days': len(data),
            'Start Date': data['date'].min().strftime('%Y-%m-%d'),
            'End Date': data['date'].max().strftime('%Y-%m-%d'),
            'Mean Temp (°F)': data['tavg'].mean(),
            'Std Temp (°F)': data['tavg'].std(),
            'Min Temp (°F)': data['tmin'].min(),
            'Max Temp (°F)': data['tmax'].max()
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()
    
    # Save summary
    summary_path = processed_dir / 'temperature_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    print()
    
    return all_data


if __name__ == '__main__':
    process_all_cities()
