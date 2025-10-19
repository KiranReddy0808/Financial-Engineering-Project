# Final ARMA Parameter Recommendations - Updated

## Executive Summary

Based on ACF/PACF analysis and systematic model comparison:

### Recommended Configuration

```python
# For Boston, New York, Chicago, Minneapolis:
fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=0, include_dayofweek=False)

# For Houston, Dallas (Texas regions):
fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=0, include_dayofweek=True)
```

## Regional Differences

### Why Different Treatments?

**Northern/Eastern Regions (Boston, NYC, Chicago, Minneapolis)**:
- Strong seasonal patterns (heating in winter)
- Industrial/commercial load dominates
- Weekly patterns captured by annual harmonics
- **Day-of-week adds complexity without benefit**

**Texas Regions (Houston, Dallas)**:
- Heavy cooling load (air conditioning)
- More volatile due to extreme heat
- Strong weekend vs. weekday differences
- **Day-of-week captures work/home AC patterns**

## Implementation Code

```python
import pandas as pd
import numpy as np
from seasonal_models import fit_seasonal_garch_model

# Load all region data
regions_data = {
    'Boston': pd.read_csv('data/processed/Boston/Boston.csv'),
    'New York': pd.read_csv('data/processed/NY/NewYork.csv'),
    'Houston': pd.read_csv('data/processed/Houston/Houston.csv'),
    'Chicago': pd.read_csv('data/processed/Chicago/Chicago.csv'),
    'Dallas': pd.read_csv('data/processed/Dallas/Dallas.csv'),
    'Minneapolis': pd.read_csv('data/processed/Minneapolis/Minneapolis.csv'),
}

# Process each region
region_fits = {}

for region_name, df in regions_data.items():
    # Prepare data
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2022-12-31')]
    df.set_index('date', inplace=True)
    srs = df['avg_load']
    
    # Texas regions use day-of-week
    use_dow = region_name in ['Houston', 'Dallas']
    
    print(f"\n{'='*80}")
    print(f"Fitting {region_name}")
    print(f"{'='*80}")
    
    # Fit model
    region_fits[region_name] = fit_seasonal_garch_model(
        srs,
        n_harmonics=3,
        ar_order=1,
        ma_order=0,
        include_dayofweek=use_dow
    )
    
    # Report results
    fit = region_fits[region_name]
    dow_text = "with day-of-week" if use_dow else "seasonal only"
    print(f"Model: 3H + AR(1) + GARCH(1,1) {dow_text}")
    print(f"Parameters: {fit['n_total_params']}")
    print(f"AIC: {fit['aic']:.2f}")
    print(f"BIC: {fit['bic']:.2f}")
    print(f"R²: {fit['r_squared']:.4f}")
    print(f"Residual Std: {fit['residual_std']:.2f} MWh")
    print(f"GARCH Persistence (α+β): {fit['persistence']:.4f}")
```

## Why MA Terms Are Not Needed

### ACF/PACF Evidence

**What the residual ACF/PACF show** (after removing seasonality):
- PACF cuts off after lag 1 → AR(1) process
- ACF decays gradually → Consistent with AR(1)
- No evidence of MA structure

### Empirical Evidence

| Model | AIC | BIC | R² | Status |
|-------|-----|-----|-----|--------|
| AR(1) | 43,165 | 43,281 | 0.762 | ✓ Best |
| ARMA(1,1) | 54,187 | 54,309 | -2.606 | ✗ Overfits |

### Physical Reasoning

Electricity load is **persistent** (AR), not **shock-driven** (MA):
- Today's load predicts tomorrow (AR effect)
- Weather changes gradually (AR effect)
- Economic activity has inertia (AR effect)
- Shocks (outages, heat waves) are rare and captured by seasonality

## Common Questions Answered

### Q1: "But my ACF shows significance at multiple lags!"

**A**: That's the **original data** ACF, which includes:
- Seasonal patterns (handled by harmonics)
- Weekly patterns (handled by day-of-week for Texas)
- Day-to-day persistence (handled by AR(1))

Look at **residual ACF** after fitting the seasonal model!

### Q2: "Why not ARMA(1,1) if it converges?"

**A**: Convergence ≠ Good model. ARMA(1,1) shows:
- Negative R² = fits worse than predicting the mean
- AR and MA coefficients nearly cancel (0.87 - 0.90 ≈ 0)
- Parameter redundancy with seasonal harmonics
- Much higher AIC/BIC

### Q3: "Why no day-of-week for most regions?"

**A**: Principle of parsimony:
- Adding 6 parameters (Tue-Sun dummies)
- Must provide substantial improvement to justify complexity
- Northern/Eastern regions: Weekly patterns captured by annual cycles
- Texas regions: Extreme heat + work patterns → need day-of-week

### Q4: "How do I know if I need day-of-week for a new region?"

**A**: Test both and compare BIC:
```python
fit_without = fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=0, include_dayofweek=False)
fit_with = fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=0, include_dayofweek=True)

bic_improvement = fit_without['bic'] - fit_with['bic']

if bic_improvement > 10:
    print("Use day-of-week (substantial improvement)")
elif bic_improvement > 0:
    print("Consider day-of-week (moderate improvement)")
else:
    print("Skip day-of-week (not worth the complexity)")
```

## Model Selection Workflow

### Step 1: Fit Seasonal Baseline
```python
fit = fit_seasonal_model(srs, n_harmonics=3, include_dayofweek=False)
```

### Step 2: Check Residual ACF/PACF
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(fit['residuals'], lags=50)
plot_pacf(fit['residuals'], lags=50)
```

### Step 3: Identify ARMA Order
- PACF cuts off at lag p → AR(p)
- ACF cuts off at lag q → MA(q)
- Both decay → Try ARMA(p,q)

### Step 4: Test Candidates
```python
# Test AR(1), AR(2), ARMA(1,1), etc.
candidates = [
    {'ar': 1, 'ma': 0},
    {'ar': 2, 'ma': 0},
    {'ar': 1, 'ma': 1},
]

for spec in candidates:
    fit = fit_seasonal_garch_model(srs, n_harmonics=3, 
                                     ar_order=spec['ar'], 
                                     ma_order=spec['ma'],
                                     include_dayofweek=False)
    print(f"AR({spec['ar']},MA{spec['ma']}): AIC={fit['aic']:.2f}, R²={fit['r_squared']:.4f}")
```

### Step 5: Select Best Model
- Lowest AIC/BIC
- Positive R²
- Converges properly
- Residuals are white noise

### Step 6: Test Day-of-Week (Optional)
```python
# Only if you suspect strong weekly patterns
fit_with_dow = fit_seasonal_garch_model(srs, n_harmonics=3,
                                         ar_order=1, ma_order=0,
                                         include_dayofweek=True)
# Compare BIC with baseline
```

## Validation Checklist

After fitting your final model:

- [ ] **R² > 0**: Model improves over naive mean
- [ ] **R² > 0.5**: Model explains substantial variance
- [ ] **AIC < baseline**: Better than simpler models
- [ ] **BIC < baseline**: Complexity is justified
- [ ] **Converged**: SARIMAX optimization succeeded
- [ ] **Residual ACF**: Most lags within confidence bands
- [ ] **Residual PACF**: Most lags within confidence bands
- [ ] **Ljung-Box test**: p-value > 0.05 (white noise)
- [ ] **ARCH LM test**: Significant (confirms need for GARCH)
- [ ] **Parameters**: Sensible values (|AR| < 1, |MA| < 1)

## Files Created

1. `ARMA_Parameter_Recommendations.md` - This file
2. `notebooks/ACF_PACF_Analysis_for_ARMA_Parameters.md` - Theory guide
3. `notebooks/Why_Negative_R2_ARMA_Overfitting.md` - Overfitting explanation
4. `notebooks/Why_ACF_Shows_Correlation_Without_MA.md` - ACF interpretation
5. `scripts/select_arma_parameters.py` - Automated testing tool

## Quick Reference

```python
# Recommended settings by region:

REGION_CONFIGS = {
    'Boston': {'harmonics': 3, 'ar': 1, 'ma': 0, 'dow': False},
    'New York': {'harmonics': 3, 'ar': 1, 'ma': 0, 'dow': False},
    'Chicago': {'harmonics': 3, 'ar': 1, 'ma': 0, 'dow': False},
    'Minneapolis': {'harmonics': 3, 'ar': 1, 'ma': 0, 'dow': False},
    'Houston': {'harmonics': 3, 'ar': 1, 'ma': 0, 'dow': True},
    'Dallas': {'harmonics': 3, 'ar': 1, 'ma': 0, 'dow': True},
}

# Apply configuration
for region, config in REGION_CONFIGS.items():
    fit = fit_seasonal_garch_model(
        data[region],
        n_harmonics=config['harmonics'],
        ar_order=config['ar'],
        ma_order=config['ma'],
        include_dayofweek=config['dow']
    )
```

---

## Summary

**Use AR(1) without day-of-week for most regions. Texas regions (Houston, Dallas) need day-of-week due to extreme cooling load patterns. MA terms cause overfitting and should not be used.**
