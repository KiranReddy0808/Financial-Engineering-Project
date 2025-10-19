# Electricity Generation Fuel ETF Data

## Overview
This directory contains historical price data for ETFs that track fuel sources used in electricity generation in the United States.

## Data Range
**2014-01-02 to 2025-10-17** (aligned with electricity load data)

## ETFs Included (6 Total)

### 1. Natural Gas - UNG (~43% of US electricity)
- **ETF**: United States Natural Gas Fund
- **Ticker**: UNG
- **Why**: Natural gas is the #1 fuel source for US electricity generation
- **Days of Data**: 2,967
- **Current Price**: $11.87

### 2. Coal - KOL (~16% of US electricity)
- **ETF**: VanEck Coal ETF
- **Ticker**: KOL
- **Why**: Coal is still a significant fuel source, especially for baseload power
- **Days of Data**: 1,764 (Note: Trading ended 2021-01-04)
- **Last Price**: $94.92

### 3. Nuclear/Uranium - URA (~18% of US electricity)
- **ETF**: Global X Uranium ETF
- **Ticker**: URA
- **Why**: Nuclear power provides consistent baseload electricity
- **Days of Data**: 2,967
- **Current Price**: $53.31

### 4. Renewables - ICLN (~21% of US electricity)
- **ETF**: iShares Global Clean Energy ETF
- **Ticker**: ICLN
- **Why**: Solar, wind, and hydro are rapidly growing electricity sources
- **Days of Data**: 2,967
- **Current Price**: $16.53

### 5. Electric Utilities - XLU
- **ETF**: Utilities Select Sector SPDR Fund
- **Ticker**: XLU
- **Why**: Tracks companies that generate and distribute electricity
- **Days of Data**: 2,967
- **Current Price**: $91.57

### 6. Oil - USO (~1% of electricity, but important)
- **ETF**: United States Oil Fund
- **Ticker**: USO
- **Why**: Used for peak generation and price signals
- **Days of Data**: 2,967
- **Current Price**: $67.98

---

## File Structure

```
energy_etfs/
├── README.md                    # This file
├── etf_summary.csv             # Summary statistics for all ETFs
├── all_energy_etfs.csv         # Combined data (16,599 rows)
├── UNG.csv                     # Natural gas prices
├── KOL.csv                     # Coal prices
├── URA.csv                     # Uranium/nuclear prices
├── ICLN.csv                    # Clean energy prices
├── XLU.csv                     # Utilities stock prices
└── USO.csv                     # Oil prices
```

---

## Data Columns

Each CSV file contains:
- **Date**: Trading date (timezone-aware)
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Number of shares traded
- **Dividends**: Dividend payments
- **Stock Splits**: Stock split events
- **Ticker**: ETF ticker symbol
- **Description**: ETF description

---

## Use Cases

### 1. Correlation with Electricity Load
Analyze how fuel prices correlate with electricity demand:
```python
import pandas as pd

# Load electricity load data
boston_load = pd.read_csv('../processed/Boston/Boston.csv')

# Load natural gas prices
gas_prices = pd.read_csv('UNG.csv')

# Merge and analyze correlation
merged = pd.merge(boston_load, gas_prices[['Date', 'Close']], 
                  left_on='date', right_on='Date')
correlation = merged[['avg_load', 'Close']].corr()
```

### 2. Electricity Production Cost Forecasting
Use fuel prices to predict wholesale electricity costs:
- Higher natural gas prices → Higher electricity generation costs
- Coal prices affect baseload electricity costs
- Renewable ETF trends show investment in clean energy

### 3. Seasonal Pattern Analysis
Compare seasonal patterns in fuel prices vs electricity demand:
- Summer: Higher natural gas demand for cooling → higher prices
- Winter: Higher heating demand → potential fuel switching
- Spring/Fall: Lower overall demand → lower fuel prices

### 4. Regional Analysis
Different regions use different fuel mixes:
- **New England (Boston)**: More natural gas, less coal
- **Midwest (Chicago)**: Mix of coal and natural gas
- **Texas (Houston/Dallas)**: Heavy natural gas, wind growing
- **Minneapolis**: Coal, nuclear, wind

---

## Data Quality Notes

⚠️ **KOL (Coal ETF)** - Trading ended on 2021-01-04
- Only has data through early 2021
- Coal industry consolidation led to ETF closure
- Use with caution for recent years

✅ **All other ETFs** - Complete data through 2025-10-17

---

## Next Steps

1. **Process the data**: Create daily/monthly aggregations
2. **Merge with load data**: Combine fuel prices with electricity load patterns
3. **Correlation analysis**: Study relationship between fuel prices and load
4. **Forecasting**: Use fuel price trends to predict electricity costs
5. **Regional comparison**: Analyze how different regions respond to fuel price changes

---

## Data Source
All data downloaded from Yahoo Finance using the `yfinance` Python library.

**Last Updated**: 2025-10-19
