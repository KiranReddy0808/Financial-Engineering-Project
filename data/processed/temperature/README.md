# Temperature Seasonal Analysis

## Overview

This directory contains processed temperature data and seasonal analysis results for 6 US cities that align with our electricity load regions.

## Processing Pipeline

### 1. Data Preprocessing
**Script**: `/workspaces/Financial-Engineering-Project/scripts/process_temperature.py`

- **Input**: Raw temperature data from `/data/raw/temperature/`
- **Output**: Processed CSV files with standardized format
- **Processing Steps**:
  1. Load raw temperature data (min/max daily temps)
  2. Calculate average temperature: `tavg = (tmin + tmax) / 2`
  3. Filter to analysis period: 2014-01-01 to 2022-12-31
  4. Remove duplicates and missing values
  5. Sort by date and save clean CSV

### 2. Seasonal Modeling
**Notebook**: `/workspaces/Financial-Engineering-Project/notebooks/Temperature_Seasonal_Analytics.ipynb`

Applies the same modeling framework used for electricity load analysis:

#### Model Structure
**Seasonal Model**: `T(t) = a₀ + a₁·t + Σₙ[Aₙ·sin(2πn·doy/365) + Bₙ·cos(2πn·doy/365)] + ε(t)`

Where:
- `a₀` = intercept (baseline temperature)
- `a₁` = linear trend coefficient (°F/day)
- `n` = harmonic number (1, 2, 3 for annual, semi-annual, tertiary cycles)
- `doy` = day of year (1-365)
- `ε(t)` = residual error

#### GARCH Extension
For temperature volatility modeling:
- **GARCH(1,1)** captures conditional heteroskedasticity
- Useful for understanding temperature variability patterns
- Helps model extreme weather events

## Data Files

### Processed Temperature Data
- `Boston.csv` - Boston temperature (2014-2022)
- `NewYork.csv` - New York City temperature
- `Houston.csv` - Houston temperature  
- `Chicago.csv` - Chicago temperature
- `Dallas.csv` - Dallas temperature
- `Minneapolis.csv` - Minneapolis temperature

**Columns**:
- `date` - Date (YYYY-MM-DD)
- `tmin` - Daily minimum temperature (°F)
- `tmax` - Daily maximum temperature (°F)
- `tavg` - Daily average temperature (°F)

### Analysis Results

#### Model Outputs
- `temperature_seasonal_models.pkl` - Saved model objects (seasonal & GARCH)
- `temperature_model_parameters.csv` - Model coefficients table
- `temperature_model_summary.csv` - Performance metrics by city

#### Visualizations
- `temperature_timeseries_all_cities.png` - Time series plots
- `temperature_seasonal_patterns.png` - Day-of-year patterns
- `temperature_trends.png` - Long-term trend analysis
- `temperature_regional_comparison.png` - Regional characteristics
- `temperature_residual_distributions.png` - Residual analysis
- `temperature_aic_bic_comparison.png` - Model comparison

#### City-Specific Diagnostics
- `Boston/temperature_diagnostics.png`
- `NewYork/temperature_diagnostics.png`
- `Houston/temperature_diagnostics.png`
- `Chicago/temperature_diagnostics.png`
- `Dallas/temperature_diagnostics.png`
- `Minneapolis/temperature_diagnostics.png`

Each diagnostic plot includes:
1. Fitted model vs actual data
2. Residual time series
3. Residual distribution (Q-Q plot)
4. Autocorrelation (ACF)
5. Partial autocorrelation (PACF)
6. Volatility clustering analysis

## Summary Statistics (2014-2022)

| City | Days | Mean Temp | Std Dev | Min Temp | Max Temp |
|------|------|-----------|---------|----------|----------|
| Boston | 3287 | 53.03°F | 17.52°F | -9.0°F | 100.0°F |
| New York | 3287 | 57.26°F | 17.62°F | 1.0°F | 101.0°F |
| Houston | 3287 | 70.98°F | 13.27°F | 13.0°F | 106.0°F |
| Chicago | 3287 | 51.32°F | 20.25°F | -23.0°F | 99.0°F |
| Dallas | 3287 | 67.70°F | 16.07°F | -2.0°F | 109.0°F |
| Minneapolis | 3287 | 47.31°F | 23.04°F | -28.0°F | 101.0°F |

## Key Findings

### Regional Characteristics
- **Warmest City**: Houston (70.98°F average)
- **Coldest City**: Minneapolis (47.31°F average)
- **Most Volatile**: Minneapolis (σ = 23.04°F)
- **Least Volatile**: Houston (σ = 13.27°F)

### Climate Zones
1. **Northern Continental** (Minneapolis, Chicago, Boston)
   - High seasonal variation
   - Extreme winter temperatures
   - Strong annual cycles

2. **Mid-Atlantic** (New York)
   - Moderate seasonal variation
   - Maritime influence
   - Less extreme temperatures

3. **Southern/Subtropical** (Houston, Dallas)
   - Lower seasonal variation
   - Milder winters
   - Hot summers

### Model Performance
The seasonal model (trend + 3 harmonics) typically explains **>95%** of temperature variance, indicating:
- Strong predictable seasonal patterns
- Relatively stable year-to-year climate
- Well-defined annual temperature cycles

GARCH models provide additional value for:
- Capturing weather volatility clustering
- Modeling extreme temperature events
- Understanding short-term temperature variability

## Usage Examples

### Load Processed Data
```python
import pandas as pd

# Load temperature data
df = pd.read_csv('/workspaces/Financial-Engineering-Project/data/processed/temperature/Boston.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print(df.head())
```

### Load Model Results
```python
import pickle

# Load all model results
with open('/workspaces/Financial-Engineering-Project/data/processed/temperature_seasonal_models.pkl', 'rb') as f:
    results = pickle.load(f)

# Access seasonal model for Boston
boston_model = results['seasonal_models']['Boston']
print(f"R² = {boston_model['r_squared']:.4f}")
print(f"AIC = {boston_model['aic']:.2f}")

# Get fitted values
fitted_temps = boston_model['fitted']
residuals = boston_model['residuals']
```

### Analyze Temperature-Load Correlation
```python
import numpy as np
import pandas as pd

# Load temperature and load data
temp = pd.read_csv('data/processed/temperature/Boston.csv', index_col='date', parse_dates=True)
load = pd.read_csv('data/processed/Boston/Boston.csv', index_col='date', parse_dates=True)

# Merge on date
combined = pd.merge(temp, load, left_index=True, right_index=True, how='inner')

# Calculate correlation
corr = combined[['tavg', 'avg_load']].corr()
print("Temperature-Load Correlation:")
print(corr)

# Heating vs Cooling Degree Days
base_temp = 65  # °F
combined['HDD'] = np.maximum(base_temp - combined['tavg'], 0)
combined['CDD'] = np.maximum(combined['tavg'] - base_temp, 0)

print("\nHeating Degree Days (HDD) correlation with load:", 
      combined[['HDD', 'avg_load']].corr().iloc[0, 1])
print("Cooling Degree Days (CDD) correlation with load:", 
      combined[['CDD', 'avg_load']].corr().iloc[0, 1])
```

## Next Steps

1. **Temperature-Load Correlation Analysis**
   - Merge temperature and electricity load data
   - Calculate heating/cooling degree days
   - Analyze regional differences in temperature sensitivity

2. **Fuel Price Integration**
   - Combine with ETF price data
   - Study how temperature affects energy commodity prices
   - Seasonal patterns in natural gas/heating oil demand

3. **Forecasting Applications**
   - Use temperature forecasts to predict electricity demand
   - Weather derivatives pricing
   - Energy trading strategies

4. **Extreme Event Analysis**
   - Identify heat waves and cold snaps
   - Impact on grid reliability
   - Peak demand forecasting

## Related Files

- **Load Analysis**: `/workspaces/Financial-Engineering-Project/notebooks/Load_Analytics.ipynb`
- **Seasonal Models**: `/workspaces/Financial-Engineering-Project/notebooks/seasonal_models.py`
- **ETF Data**: `/workspaces/Financial-Engineering-Project/data/raw/energy_etfs/`

## References

- Temperature data source: NOAA/NCDC weather stations
- Analysis period: 2014-01-01 to 2022-12-31 (3,287 days)
- Model methodology: Seasonal decomposition with GARCH volatility
- Aligned with electricity load data for integrated analysis
