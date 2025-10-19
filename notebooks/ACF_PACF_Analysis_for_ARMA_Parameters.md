# ACF and PACF Analysis for ARMA Parameter Selection

## Overview
This document provides guidance on selecting AR(p) and MA(q) parameters for the SARIMAX model used in `fit_seasonal_garch_model()` based on ACF and PACF patterns.

## Theory: Interpreting ACF and PACF

### Autocorrelation Function (ACF)
- **Shows**: Correlation between the series and its lagged values
- **Significance**: Bars outside the confidence bands (blue shaded region) indicate significant correlation
- **Use**: Helps identify the MA(q) order and overall dependence structure

### Partial Autocorrelation Function (PACF)
- **Shows**: Correlation between the series and its lagged values, after removing the effect of intermediate lags
- **Significance**: Bars outside the confidence bands indicate significant partial correlation
- **Use**: Helps identify the AR(p) order

## Pattern Recognition Guide

### Pure AR(p) Process
- **ACF**: Decays exponentially or with damped sinusoidal pattern
- **PACF**: Cuts off sharply after lag p
- **Action**: Set `ar_order=p`, `ma_order=0`

### Pure MA(q) Process
- **ACF**: Cuts off sharply after lag q
- **PACF**: Decays exponentially or with damped sinusoidal pattern
- **Action**: Set `ar_order=0`, `ma_order=q`

### ARMA(p,q) Process
- **ACF**: Decays gradually (exponentially or sinusoidally)
- **PACF**: Decays gradually (exponentially or sinusoidally)
- **Action**: Both `ar_order=p` and `ma_order=q` needed
- **Common patterns**:
  - ARMA(1,1): Most common, handles both short-term autocorrelation and moving average effects
  - ARMA(2,1) or ARMA(1,2): For more complex patterns

### White Noise (Well-Fitted Model)
- **ACF**: All lags within confidence bands (except lag 0 = 1)
- **PACF**: All lags within confidence bands
- **Action**: No ARMA terms needed (seasonal model sufficient)

## Electricity Load Characteristics

For electricity load data (after seasonal detrending), typical patterns include:

1. **Weekly Cycles**: Day-of-week effects (handled by exogenous variables in SARIMAX)
2. **Short-term Autocorrelation**: Previous day(s) affect current day → AR component
3. **Shock Effects**: Unusual events affect multiple subsequent days → MA component
4. **Volatility Clustering**: Periods of high/low variance → GARCH component (separate stage)

## Recommended Parameter Selection Strategy

### Step 1: Visual Inspection
1. Look at ACF plot saved in `data/images/autocorrelation_{region}.png`
2. Count significant lags (outside confidence bands) in:
   - ACF: First few lags (1-7) for MA order
   - PACF: First few lags (1-7) for AR order

### Step 2: Initial Parameter Guess
Based on electricity load patterns, start with:

#### Conservative (Default):
```python
fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=1)
```
- **Rationale**: ARMA(1,1) captures most common patterns without overfitting
- **Good for**: When ACF and PACF both show gradual decay after lag 1-2

#### No Autocorrelation:
```python
fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=0, ma_order=0)
```
- **Rationale**: Seasonal harmonics + day-of-week explain all patterns
- **Good for**: When ACF and PACF show minimal significant lags (white noise residuals)

#### Stronger AR Component:
```python
fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=2, ma_order=1)
```
- **Rationale**: Yesterday and day-before-yesterday strongly predict today
- **Good for**: When PACF shows 2-3 significant lags, ACF decays gradually

#### Stronger MA Component:
```python
fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=2)
```
- **Rationale**: Shock effects propagate through multiple days
- **Good for**: When ACF shows 2-3 significant lags, PACF decays gradually

### Step 3: Model Comparison
Use AIC/BIC to compare models:

```python
# Test multiple specifications
models_to_test = [
    {'ar': 0, 'ma': 0},
    {'ar': 1, 'ma': 0},
    {'ar': 0, 'ma': 1},
    {'ar': 1, 'ma': 1},
    {'ar': 2, 'ma': 1},
    {'ar': 1, 'ma': 2},
]

results = []
for spec in models_to_test:
    fit = fit_seasonal_garch_model(
        srs, 
        n_harmonics=3, 
        ar_order=spec['ar'], 
        ma_order=spec['ma']
    )
    results.append({
        'AR': spec['ar'],
        'MA': spec['ma'],
        'AIC': fit['aic'],
        'BIC': fit['bic'],
        'R²': fit['r_squared']
    })

comparison_df = pd.DataFrame(results)
print(comparison_df.sort_values('AIC'))
```

### Step 4: Residual Diagnostics
After fitting, verify the model with:

1. **Ljung-Box Test**: Check if residuals are white noise
   - Run `run_full_diagnostics(fit, region_name, save_dir)`
   - Look at ACF of residuals: should be within confidence bands

2. **ARCH LM Test**: Check if volatility clustering remains
   - If significant, GARCH component will handle it
   - If not significant after ARMA, may reduce GARCH importance

## Common Patterns in Load Data

### Pattern 1: Strong Day-of-Week + Minimal Autocorrelation
- **ACF/PACF**: Mostly within confidence bands after lag 7
- **Recommendation**: `ar_order=0, ma_order=0` or `ar_order=1, ma_order=0`
- **Reason**: Day-of-week dummies capture weekly patterns

### Pattern 2: Persistent Autocorrelation
- **ACF**: Decays slowly, significant up to lag 5-10
- **PACF**: Significant at lags 1-2
- **Recommendation**: `ar_order=2, ma_order=1`
- **Reason**: Need stronger AR to capture persistence

### Pattern 3: Shock Effects
- **ACF**: Significant at lags 1-3
- **PACF**: Decays after lag 1
- **Recommendation**: `ar_order=1, ma_order=2`
- **Reason**: MA component captures lagged effects of shocks

## Practical Guidelines

### Starting Point
For electricity load data, the default `ar_order=1, ma_order=1` is a solid starting point because:
1. Previous day load predicts today (AR term)
2. Unexpected shocks propagate (MA term)
3. Balances model complexity vs. goodness of fit

### Warning Signs
- **Overfitting**: If `ar_order + ma_order > 4`, you're likely overfitting
  - Solution: Reduce parameters, ensure seasonal harmonics are adequate
  
- **Non-convergence**: If SARIMAX doesn't converge
  - Solution: Reduce parameters or check for data issues
  
- **High AIC/BIC**: If adding parameters increases AIC/BIC
  - Solution: Use simpler model

### Model Selection Priority
1. **Parsimony**: Prefer simpler models (fewer parameters)
2. **AIC**: Lower is better for predictive performance
3. **BIC**: Lower is better, penalizes complexity more than AIC
4. **Residual Diagnostics**: Ensure residuals are white noise
5. **Domain Knowledge**: Electricity load has strong daily patterns

## Example Workflow

```python
import pandas as pd
from seasonal_models import fit_seasonal_garch_model, run_full_diagnostics

# Load data
df = pd.read_csv('../data/processed/Boston/Boston.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
srs = df['avg_load']

# Try different ARMA specifications
print("Testing ARMA(1,1) - Default")
fit_11 = fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=1)
print(f"  AIC: {fit_11['aic']:.2f}, BIC: {fit_11['bic']:.2f}, R²: {fit_11['r_squared']:.4f}")

print("\nTesting ARMA(2,1) - Stronger AR")
fit_21 = fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=2, ma_order=1)
print(f"  AIC: {fit_21['aic']:.2f}, BIC: {fit_21['bic']:.2f}, R²: {fit_21['r_squared']:.4f}")

print("\nTesting ARMA(1,2) - Stronger MA")
fit_12 = fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=2)
print(f"  AIC: {fit_12['aic']:.2f}, BIC: {fit_12['bic']:.2f}, R²: {fit_12['r_squared']:.4f}")

# Select best model based on AIC
best_fit = min([fit_11, fit_21, fit_12], key=lambda x: x['aic'])
print(f"\n✓ Best model selected with AIC: {best_fit['aic']:.2f}")

# Run full diagnostics on best model
run_full_diagnostics(best_fit, 'Boston', save_dir='../data/images')
```

## References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts.
- Project notebook: `notebooks/Load_Analytics.ipynb`
- Modeling functions: `notebooks/seasonal_models.py`
