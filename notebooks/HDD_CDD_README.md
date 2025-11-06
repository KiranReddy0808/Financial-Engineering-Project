# HDD/CDD Hedge Analysis Notebooks

## Two Versions Available

### `HDD_CDD_Hedge_Analysis.ipynb` (Version 1)
**Perfect Foresight Model**
- XGBoost uses actual contemporaneous weather
- Represents upper bound on hedge effectiveness
- Good for understanding relationships

### `HDD_CDD_Hedge_Analysis2.ipynb` (Version 2) ⭐ **RECOMMENDED**
**Realistic Forecasting Model**
- Adds 2°F Gaussian noise to temperature forecasts
- XGBoost uses forecasted weather (actual + noise)
- Hedging uses actual weather for settlement
- More realistic operational scenario

## Key Implementation Details (Version 2)

### Feature Creation Strategy

```python
# 1. Store actual weather for hedging
df['HDD_actual'] = df['HDD'].copy()
df['CDD_actual'] = df['CDD'].copy()

# 2. Add forecast error
temp_error = np.random.normal(0, 2.0, len(df))  # 2°F std
tavg_forecast = df['tavg'] + temp_error

# 3. Recalculate HDD/CDD from forecast
HDD_forecast = np.maximum(0, 65.0 - tavg_forecast)
CDD_forecast = np.maximum(0, tavg_forecast - 65.0)

# 4. XGBoost sees FORECASTED weather
df['HDD'] = HDD_forecast
df['CDD'] = CDD_forecast

# 5. Lagged features use ACTUAL past values
df['HDD_lag1'] = df['HDD_actual'].shift(1)
df['CDD_lag1'] = df['CDD_actual'].shift(1)

# 6. Interactions use FORECASTED weather
df['HDD_winter'] = df['HDD'] * df['is_winter']
df['CDD_summer'] = df['CDD'] * df['is_summer']
```

### Hedging Strategy

```python
# Hedge multiplier regression uses ACTUAL weather
hdd_slope, _, hdd_r, hdd_p, _ = linregress(
    winter_data['HDD_actual'], winter_data['error']
)

# Hedge settlement uses ACTUAL weather
test_df['HDD_payout'] = test_df['HDD_actual'] * 20  # $20 per degree-day
test_df['HDD_hedge_MW'] = test_df['HDD_payout'] * hdd_multiplier
```

## Usage

```bash
# For realistic analysis (recommended)
jupyter notebook HDD_CDD_Hedge_Analysis2.ipynb

# For theoretical upper bound
jupyter notebook HDD_CDD_Hedge_Analysis.ipynb
```

## Results Interpretation

**Version 2 will show:**
- Higher RMSE (forecast error includes weather uncertainty)
- Slightly lower hedge effectiveness (more realistic)
- Better representation of operational performance

**Version 1 will show:**
- Lower RMSE (perfect weather foresight)
- Higher hedge effectiveness (optimistic)
- Upper bound on theoretical performance
