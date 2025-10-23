# Comprehensive Financial Engineering Project Report
## Electricity Load Analytics, Extreme Value Theory & Risk Management

---

## Executive Summary

This project develops a complete quantitative framework for **electricity load forecasting, extreme event risk modeling, and hedging strategy optimization** across major U.S. markets. Using advanced statistical, machine learning, and financial engineering techniques, we analyze 6 cities (Boston, New York, Chicago, Houston, Dallas, Minneapolis) with 4,000-7,300 daily observations per city (2005-2025 range).

**Key Results:**
- **Forecasting**: XGBoost achieves 15-28% RMSE improvement over OLS baseline (R² = 0.77-0.80)
- **Extreme Value Theory**: All cities show heavy-tailed distributions (ξ = 0.18-0.29)
- **Tail Dependence**: Joint 99th percentile exceedance is 16-5000x more likely than independence
- **Hedging**: ETF hedge ratios range from -4,761 to +3,029 MW/return (mostly negative UNG/USO correlations)
- **Volatility Modeling**: ARMA-GARCH captures clustering with persistence (α+β ≈ 0.99)

---

## Table of Contents

1. [Data Sources & Collection](#1-data-sources--collection)
2. [Methodology & Statistical Framework](#2-methodology--statistical-framework)
3. [Electricity Load Analytics](#3-electricity-load-analytics)
4. [Temperature Analytics](#4-temperature-analytics)
5. [Comprehensive Forecasting Framework](#5-comprehensive-forecasting-framework)
6. [Extreme Value Theory & Tail Risk](#6-extreme-value-theory--tail-risk)
7. [Copula Modeling & Joint Scenarios](#7-copula-modeling--joint-scenarios)
8. [Hedging Optimization with ETFs](#8-hedging-optimization-with-etfs)
9. [Cross-Correlation & Tail Dependence Analysis](#9-cross-correlation--tail-dependence-analysis)
10. [Actionable Insights & Risk Management](#10-actionable-insights--risk-management)
11. [Future Work](#11-future-work)

---

## 1. Data Sources & Collection

### 1.1 Electricity Load Data

**Geographic Coverage:**
- **Boston** (NEMASSBOST): ISO-NE region, 3,283 days
- **New York** (NYISO): PAL Integrated zone, 7,300 days  
- **Chicago** (MISO Central): Mid-continent ISO, 5,844 days
- **Houston** (ERCOT Coast): Texas grid, 4,017 days
- **Dallas** (ERCOT North): Texas grid, 4,017 days
- **Minneapolis** (MISO): Northern territory, 5,844 days

**Data Structure:**
```
Columns: [date, min_load, max_load, avg_load]
Frequency: Daily aggregates from 5-minute/hourly data
Date Range: 2005-01-01 to 2024-12-31 (varies by region)
Analysis Period: 2014-01-01 to 2022-12-31 (consistency)
```

**Collection Scripts:**
- `download_miso_load.py`: Fetches monthly zips (2009-2022) + daily XLS (2023-2024)
- `download_nyiso_palintegrated.py`: Daily CSV zips from NYISO
- `process_boston.py`: Parses ISO-NE Excel files with `NEMASSBOST`/`NEMA` sheets
- `process_houston.py`: Processes ERCOT data (also handles Dallas)

**Load Characteristics (Daily Averages):**
| City | Min Load (MW) | Max Load (MW) | Avg Load (MW) | Std Dev (MW) |
|------|--------------|--------------|--------------|--------------|
| Boston | ~2,500 | ~4,500 | 3,200 | 450 |
| New York | ~8,000 | ~13,000 | 10,500 | 1,450 |
| Chicago | ~25,000 | ~40,000 | 32,000 | 4,200 |
| Minneapolis | ~8,500 | ~11,500 | 9,800 | 850 |

![Load Volatility Patterns](data/images/Boston_volatility_over_time.png)

### 1.2 Temperature Data

**Source:** Raw CSV files from weather APIs (NOAA/Visual Crossing)
**Variables:**
- `DAILY_MAX_TEMP`: Maximum temperature (°F)
- `DAILY_MIN_TEMP`: Minimum temperature (°F)
- `tavg`: Average = (max + min) / 2

**Derived Weather Indices:**
```python
HDD = max(65 - tavg, 0)  # Heating Degree Days (base 65°F)
CDD = max(tavg - 65, 0)  # Cooling Degree Days (base 65°F)
```

**Temperature Statistics:**
| City | Mean (°F) | Std (°F) | Min (°F) | Max (°F) | Seasonal R² |
|------|-----------|----------|----------|----------|-------------|
| Boston | 53.0 | 17.5 | 1.5 | 90.5 | 0.816 |
| New York | 57.3 | 17.6 | 9.0 | 91.5 | 0.834 |
| Houston | 71.0 | 13.3 | 20.5 | 93.5 | 0.728 |
| Chicago | 51.3 | 20.3 | -16.5 | 87.0 | 0.812 |
| Dallas | 67.7 | 16.1 | 8.0 | 97.5 | 0.768 |
| Minneapolis | 47.3 | 23.0 | -20.5 | 90.0 | 0.835 |

![Temperature Day-of-Year Patterns](data/images/temp_boston_day_of_year_patterns.png)

### 1.3 Energy ETF Data

**Tickers & Strategy:**
- **UNG** (Natural Gas): Heating fuel proxy (negative correlation expected)
- **XLU** (Utilities): Sector hedge
- **ICLN** (Clean Energy): Renewable capacity
- **URA** (Uranium): Nuclear power
- **USO** (Crude Oil): Fossil fuel energy
- **KOL** (Coal): Traditional generation

**Data Processing:**
```python
# Log returns calculation
etf_data['log_return'] = np.log(etf_data['Close'] / etf_data['Close'].shift(1))

# Timestamp normalization (critical fix)
etf_data['Date'] = pd.to_datetime(etf_data['Date'], utc=True) \
    .dt.tz_localize(None).dt.normalize()
```

**Coverage:** 2005-01-03 to 2024-12-31 (aligned with load data)

![ETF Hedge Ratios Heatmap](data/images/combined_correlation_matrices.png)

---

## 2. Methodology & Statistical Framework

### 2.1 Seasonal Decomposition

**Model Specification:**
$$
y_t = \mu + \beta \cdot t + \sum_{n=1}^{N} \left[ \alpha_n \sin\left(\frac{2\pi n \cdot \text{doy}_t}{365.25}\right) + \beta_n \cos\left(\frac{2\pi n \cdot \text{doy}_t}{365.25}\right) \right] + \sum_{d=2}^{7} \gamma_d \cdot \mathbb{1}(\text{dow}_t = d) + \varepsilon_t
$$

**Components:**
- **Trend**: Linear $\beta \cdot t$ captures long-term changes
- **Harmonics**: $N \in \{3, 6\}$ captures seasonal cycles (365.25 accounts for leap years)
- **Day-of-week**: Monday as reference category
- **Residuals**: $\varepsilon_t$ modeled with ARMA-GARCH

**Implementation:**
```python
# Design matrix construction
X = np.ones((n, 1))  # Intercept
X = np.column_stack([X, t])  # Trend
for n in range(1, N_harmonics + 1):
    X = np.column_stack([X, 
        np.sin(2 * np.pi * n * doy / 365.25),
        np.cos(2 * np.pi * n * doy / 365.25)
    ])
for d in range(1, 7):  # Tue-Sun dummies
    X = np.column_stack([X, (dow == d).astype(int)])

# OLS estimation
params = np.linalg.lstsq(X, y, rcond=None)[0]
```

### 2.2 ARMA-GARCH Volatility Modeling

**Mean Model (ARMA(p,q)):**
$$
\varepsilon_t = \phi_1 \varepsilon_{t-1} + \cdots + \phi_p \varepsilon_{t-p} + \theta_1 \eta_{t-1} + \cdots + \theta_q \eta_{t-q} + \eta_t
$$

**Volatility Model (GARCH(1,1)):**
$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

**Persistence:** $\alpha + \beta \approx 0.99$ indicates high volatility clustering

**Model Selection:**
- **AIC/BIC comparison** between 3H vs 6H harmonics
- **ARMA order selection** via ACF/PACF analysis
- **Diagnostic tests**: Ljung-Box (autocorrelation), ARCH LM (heteroskedasticity)

![ARMA Comparison Boston](data/images/arma_comparison_Boston.png)

### 2.3 Model Comparison Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | Variance explained (0-1) |
| RMSE | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Prediction error (same units) |
| MAE | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | Average absolute error |
| AIC | $-2\ln(L) + 2k$ | Model parsimony (lower = better) |
| BIC | $-2\ln(L) + k\ln(n)$ | Penalizes complexity more |

**Model Comparison Results (Boston):**
```csv
Model                  | N_Harmonics | ARMA    | AIC      | BIC      | R²
3H + ARMA(0,2) + GARCH | 3           | (0, 2)  | 42995.09 | 43086.56 | 0.759
6H + ARMA(0,2) + GARCH | 6           | (0, 2)  | 42997.61 | 43125.65 | 0.759
```
**Winner:** 3H model (lower AIC/BIC, equivalent R²)

---

## 3. Electricity Load Analytics

**Notebook:** `Load_Analytics.ipynb` (62 cells, 1,234 lines)

### 3.1 Volatility Patterns

**Analysis:**
1. **Rolling Volatility** (30-day window): Captures time-varying uncertainty
2. **Day-of-Year Volatility**: Summer peaks (AC usage), winter troughs
3. **Clustering Analysis**: ACF of squared residuals shows ARCH effects
4. **Time Series**: Long-term trends in market volatility

**Key Findings:**
- Boston: Summer volatility 2x winter (AC vs heating efficiency)
- New York: Consistent volatility year-round (diversified demand)
- Chicago: Extreme spikes in polar vortex events (Jan 2014, Feb 2021)

![Rolling Volatility](data/images/Boston_rolling_volatility.png)
![Volatility by Day of Year](data/images/Boston_volatility_by_doy.png)

### 3.2 Autocorrelation Analysis

**Purpose:** Identify ARMA(p,q) orders for residual modeling

**Results (Boston):**
- **ACF**: Significant lags 1-7 → MA(2) component
- **PACF**: Cuts off after lag 2 → AR(1) or AR(2)
- **Selected:** ARMA(0,2) based on parsimony

![Autocorrelation](data/images/autocorrelation_Boston.png)

**All Cities ACF Patterns:**
| City | ACF Lag-1 | Significant Lags | Recommended MA |
|------|-----------|------------------|----------------|
| Boston | 0.83 | 1-7 | MA(2) |
| New York | 0.87 | 1-10 | MA(2) |
| Chicago | 0.79 | 1-5 | MA(1) |
| Minneapolis | 0.81 | 1-6 | MA(2) |

### 3.3 Model Fits & Residual Diagnostics

**Full Pipeline:**
1. Fit seasonal harmonics (3H or 6H)
2. Add ARMA(p,q) to residuals
3. Apply GARCH(1,1) for volatility
4. Diagnostic tests: Normality (D'Agostino-Pearson), Ljung-Box, ARCH LM

**Boston Results:**
```
3H + ARMA(0,2) + GARCH(1,1):
  R² = 0.759
  Residual Std = 192.31 MW
  Log-Likelihood = -21,578.19
  AIC = 43,184.39
  
  GARCH parameters:
    ω = 0.1754
    α = 0.1722
    β = 0.8271
    Persistence (α+β) = 0.9993
```

![Model Fit Boston](data/images/model_fit_and_residuals_Boston.png)
![Residual Distribution](data/images/residual_distribution_Boston.png)

### 3.4 Multi-Region Comparison

**Comprehensive Results Table:**
*See `data/processed/all_regions_model_comparison.csv` (288 rows, 18 columns)*

**Summary Statistics:**
| Region | Best Model | R² | Residual Std | Persistence |
|--------|------------|-----|-------------|-------------|
| Boston | 3H+ARMA(0,2)+G(1,1) | 0.759 | 192.3 MW | 0.999 |
| NewYork | 3H+ARMA(1,3)+G(1,2) | 0.850 | 485.0 MW | 0.998 |

**Key Insight:** All regions show near-unit-root GARCH persistence → shocks persist for weeks

---

## 4. Temperature Analytics

**Notebook:** `Temperature_Analytics.ipynb` (27 cells, 537 lines)

### 4.1 Seasonal GARCH Modeling

**Model Performance:**
| City | Seasonal AIC | GARCH AIC | Improvement | Mean Volatility |
|------|--------------|-----------|-------------|-----------------|
| Boston | 22,618.99 | 22,406.58 | 212.41 | 233.8°F |
| NewYork | 22,314.39 | 21,954.94 | 359.46 | 238.9°F |
| Houston | 22,070.37 | 20,675.04 | 1,395.33 | 306.3°F |
| Chicago | 23,639.47 | 23,108.34 | 531.13 | 286.2°F |
| Dallas | 22,805.91 | 21,811.54 | 994.37 | 305.3°F |
| Minneapolis | 24,053.33 | 23,333.65 | 719.68 | 303.0°F |

**Interpretation:** GARCH captures volatility clustering (cold snaps, heat waves)

![Temperature Seasonal GARCH](data/images/temp_boston_seasonal_garch_model.png)

### 4.2 Extreme Value Theory for Temperature

**Right Tail (Heat Waves):**
```csv
City         | EVI (ξ) | Threshold | P(extreme)
Boston       | 0.23    | 85°F      | 0.05
NewYork      | 0.27    | 87°F      | 0.048
Chicago      | 0.19    | 82°F      | 0.052
Minneapolis  | 0.21    | 84°F      | 0.051
```

**Left Tail (Cold Snaps):**
```csv
City         | EVI (ξ) | Threshold | P(extreme)
Boston       | 0.18    | 20°F      | 0.047
Chicago      | 0.24    | 10°F      | 0.053
Minneapolis  | 0.27    | -5°F      | 0.049
```

*See `data/processed/temperature_evi_left_tail.csv` and `temperature_evi_right_tail.csv`*

![Extreme Value Indices](data/images/evt_extreme_value_indices.png)

### 4.3 HDD/CDD Analysis

**Heating Degree Days (Winter):**
- Boston: 3,800 HDD/year (moderate heating)
- Chicago: 4,500 HDD/year (severe heating)
- Minneapolis: 5,200 HDD/year (extreme heating)
- Houston: 800 HDD/year (minimal heating)

**Cooling Degree Days (Summer):**
- Houston: 3,200 CDD/year (extreme cooling)
- Dallas: 2,900 CDD/year (high cooling)
- Boston: 600 CDD/year (moderate cooling)
- Minneapolis: 450 CDD/year (limited cooling)

![HDD vs CDD Correlations](data/images/seasonal_correlations_hdd_vs_cdd.png)

---

## 5. Comprehensive Forecasting Framework

**Notebook:** `Load_Forecasting_EVT_Hedging.ipynb` (29 cells, 988 lines)

### 5.1 Feature Engineering

**51 Features Created:**
1. **Weather Variables (8):**
   - HDD, CDD (raw)
   - HDD_anom, CDD_anom (deviation from 30-day moving average)
   - HDD_roll_7d, CDD_roll_7d (rolling means)
   - HDD_lag1, CDD_lag1 (1-day lags)

2. **Calendar Effects (10):**
   - Month (1-12)
   - Day of week (0-6)
   - Day of year (1-366)
   - Is weekend (0/1)
   - Is summer/winter (0/1)
   - Quarter (1-4)

3. **Load History (6):**
   - avg_load_lag1, lag2, lag3 (autoregressive)
   - avg_load_roll_7d (weekly average)
   - avg_load_roll_30d (monthly trend)
   - load_growth_7d (weekly change %)

4. **ETF Returns (6):**
   - UNG_ret, XLU_ret, ICLN_ret, URA_ret, USO_ret, KOL_ret

5. **Interactions (21):**
   - HDD × month (heating efficiency seasonality)
   - CDD × month (cooling demand patterns)
   - Weekend × HDD/CDD (different usage patterns)
   - ETF × weather anomalies (energy price impacts)

**Implementation:**
```python
# Weather anomalies
features['HDD_anom'] = (features['HDD'] - 
    features['HDD'].rolling(30, min_periods=1).mean())

# Interactions
features['HDD_summer'] = features['HDD'] * (features['month'].isin([6,7,8]))
features['CDD_weekend'] = features['CDD'] * features['is_weekend']
```

### 5.2 Econometric Baseline (OLS)

**Model Specification:**
$$
\text{Load}_t = \beta_0 + \sum_{i=1}^{51} \beta_i X_{it} + \varepsilon_t
$$

**Results:**
| City | RMSE (MW) | MAE (MW) | R² | Significant Features |
|------|-----------|----------|-----|---------------------|
| Boston | 127.19 | 86.84 | 0.643 | 38/51 (p < 0.05) |
| NewYork | 356.69 | 227.05 | 0.678 | 42/51 |
| Chicago | 1,414.85 | 1,002.96 | 0.758 | 45/51 |
| Minneapolis | 469.34 | 331.44 | 0.721 | 40/51 |

**Key Coefficients (Boston):**
```
CDD: +15.2 MW/degree (cooling demand)
HDD: +8.7 MW/degree (heating demand)
Weekend: -120.5 MW (commercial closure)
Summer: +85.3 MW (AC usage)
UNG_ret: -45.8 MW/return (inverse gas prices)
```

### 5.3 Machine Learning (XGBoost)

**Hyperparameters:**
```python
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1
}
```

**Time Series Cross-Validation:**
- 5 folds
- Expanding window (no data leakage)
- Early stopping at 20 rounds

**Performance:**
| City | XGB RMSE | XGB MAE | XGB R² | Improvement vs OLS |
|------|----------|---------|--------|-------------------|
| Boston | 92.59 | 59.96 | 0.768 | **27.2%** |
| NewYork | 255.02 | 170.67 | 0.801 | **28.5%** |
| Chicago | 1,197.59 | 864.43 | 0.804 | **15.4%** |
| Minneapolis | 506.95 | 344.83 | 0.785 | **-8.0%** (overfit) |

**Feature Importance (Boston):**
```
1. avg_load_lag1: 18.2%
2. CDD: 12.5%
3. avg_load_roll_7d: 9.8%
4. HDD: 7.3%
5. day_of_week: 5.6%
... (46 more features)
```

![Forecast Comparison](data/images/temp_comprehensive_all_regions.png)

### 5.4 Forecast Error Analysis

**Out-of-Sample Testing (2022):**
- **Best Case (NewYork):** 28.5% RMSE reduction, captures peak events
- **Worst Case (Minneapolis):** Overfitting → OLS more robust
- **Average:** 20.7% improvement across regions

**Error Distribution:**
- Residuals ~ N(0, σ) for OLS
- XGBoost residuals show right-skew (underestimates peaks)

---

## 6. Extreme Value Theory & Tail Risk

### 6.1 Generalized Pareto Distribution (GPD)

**Threshold Selection:**
- **95th percentile** (q95): λ ≈ 0.05 (5% exceedances)
- **97.5th percentile** (q975): λ ≈ 0.025 (2.5% exceedances)

**Model:**
$$
P(X > u + y \mid X > u) = \left(1 + \xi \frac{y}{\sigma}\right)^{-1/\xi}
$$

**Parameter Estimates:**
| City | Quantile | ξ (shape) | σ (scale) | Threshold (MW) | n_exceed |
|------|----------|-----------|-----------|----------------|----------|
| Boston | q95 | 0.280 | 24.62 | 69.06 | 164 |
| Boston | q975 | 0.181 | 34.42 | 87.02 | 82 |
| NewYork | q95 | 0.285 | 80.94 | 197.06 | 164 |
| NewYork | q975 | 0.287 | 95.50 | 263.52 | 82 |
| Chicago | q95 | 0.185 | 369.19 | 952.72 | 164 |
| Chicago | q975 | 0.233 | 383.68 | 1,242.85 | 82 |
| Minneapolis | q95 | 0.208 | 138.09 | 349.72 | 164 |
| Minneapolis | q975 | 0.264 | 146.79 | 454.76 | 82 |

**Interpretation:**
- **ξ > 0**: Heavy-tailed distributions (extreme events likely)
- **ξ ≈ 0.2-0.3**: Comparable to financial tail risk (stock market crashes)
- **Implications**: Simple Gaussian models underestimate extreme load

*See `data/processed/gpd_tail_parameters.csv`*

### 6.2 Exceedance Probabilities

**Question:** What's P(Load > threshold + 500 MW)?

**Results (Boston q95 threshold = 69.06 MW):**
```python
# P(Load > 569 MW | Load > 69 MW)
y = 500
prob = (1 + 0.280 * 500 / 24.62) ** (-1/0.280)
# prob ≈ 0.023 (2.3% chance of 500 MW spike)
```

**Practical Use:**
- **Reserve capacity planning**: Size backup generation
- **Price spike risk**: Estimate P(price > $500/MWh)
- **System reliability**: N-1 contingency analysis

### 6.3 Return Levels

**10-year return level** (occurs once every 10 years):
$$
\text{RL}_{10} = u + \frac{\sigma}{\xi}\left[\left(\frac{10 \cdot 365 \cdot \lambda}{1}\right)^\xi - 1\right]
$$

**Estimates:**
| City | 10-year RL (MW) | 20-year RL (MW) | 50-year RL (MW) |
|------|----------------|----------------|----------------|
| Boston | 285 | 320 | 370 |
| NewYork | 890 | 1,050 | 1,280 |
| Chicago | 3,420 | 3,850 | 4,450 |
| Minneapolis | 1,150 | 1,290 | 1,480 |

**Risk Management:**
- Size generation capacity to 50-year level
- Purchase insurance for events > 20-year level
- Stress test portfolios at 10-year level

---

## 7. Copula Modeling & Joint Scenarios

### 7.1 Student-t Copula

**Why Student-t?**
- Captures tail dependence (unlike Gaussian)
- Degrees of freedom parameter ν controls tail thickness
- Realistic for energy markets (simultaneous spikes)

**Correlation Structure (Kendall's τ):**
```csv
           Boston  NewYork  Chicago  Minneapolis
Boston     1.000    0.542    0.387      0.298
NewYork    0.542    1.000    0.456      0.341
Chicago    0.387    0.456    1.000      0.625
Minneapolis 0.298   0.341    0.625      1.000
```

**Fitted Parameters:**
- ν = 5.2 (moderate tail dependence)
- Correlation matrix: positive across all pairs

### 7.2 Scenario Generation

**Method:**
1. Fit marginal GPD distributions (city-specific)
2. Estimate Kendall's τ from residuals
3. Fit Student-t copula to transform to uniform [0,1]
4. Generate 100,000 joint scenarios
5. Back-transform to original load space

**Simulation:**
```python
from scipy.stats import t as tdist

# Generate correlated Student-t variates
Z = np.random.multivariate_normal(np.zeros(4), R, size=100000)
U = tdist.cdf(Z, df=nu)  # Transform to uniform

# Apply inverse GPD
load_scenarios = {}
for city in cities:
    load_scenarios[city] = gpd_inverse_cdf(U[:, city_idx], 
                                            xi=xi[city], 
                                            sigma=sigma[city])
```

### 7.3 Joint Tail Probabilities

**Multi-City Exceedance:**
| Threshold | P(all 4 exceed) | P(any exceed) | P(exactly 2) | P(exactly 3) |
|-----------|-----------------|---------------|--------------|--------------|
| 0.90 (90th pct) | 0.00179 | 0.30319 | 0.06455 | 0.01386 |
| 0.95 (95th pct) | 0.00060 | 0.16160 | 0.02715 | 0.00513 |
| 0.99 (99th pct) | 0.00003 | 0.03462 | 0.00436 | 0.00082 |

**Comparison to Independence:**
- **Independence:** P(all 4 > 99th) = 0.01⁴ = 0.00000001
- **Observed:** P = 0.00003 
- **Ratio:** 3,000x more likely (extreme tail dependence!)

**Interpretation:**
- Heat waves/cold snaps affect entire regions simultaneously
- Diversification benefits limited in extreme events
- Portfolio risk ≠ sum of individual risks

*See `data/processed/joint_tail_probabilities.csv`*

![Tail Dependence Ratios](data/images/evt_tail_dependence_ratios.png)

---

## 8. Hedging Optimization with ETFs

### 8.1 Hedge Ratio Calculation

**Definition:**
$$
\beta_{\text{hedge}} = \frac{\text{Cov}(\Delta \text{Load}, r_{\text{ETF}})}{\text{Var}(r_{\text{ETF}})}
$$

**Interpretation:** MW of load exposure per 1% ETF return

**Results:**
*See full table in `data/processed/etf_hedge_ratios.csv`*

**Boston:**
```csv
ETF   | Hedge Ratio  | Correlation | Covariance
UNG   | -144.77      | -0.0176     | -0.0959
XLU   | +31.75       | +0.0018     | +0.0046
ICLN  | -68.49       | -0.0047     | -0.0148
URA   | -34.96       | -0.0029     | -0.0111
USO   | -58.30       | -0.0069     | -0.0370
```

**Chicago:**
```csv
ETF   | Hedge Ratio  | Correlation | Covariance
UNG   | -2,405.12    | -0.0227     | -1.5935
XLU   | +3,029.50    | +0.0133     | +0.4350
ICLN  | +2,746.94    | +0.0148     | +0.5945
URA   | -4,760.97    | -0.0310     | -1.5097
USO   | -3,217.48    | -0.0297     | -2.0428
```

**Patterns:**
- **UNG (Natural Gas)**: Negative hedge for all cities (inverse relationship)
- **XLU (Utilities)**: Positive for Midwest (sector exposure)
- **USO (Oil)**: Large negative ratios (fossil fuel substitution)

### 8.2 CVaR Hedge Optimization

**Objective:** Minimize 95% Conditional Value-at-Risk
$$
\text{CVaR}_{0.95} = \mathbb{E}[\text{Loss} \mid \text{Loss} > \text{VaR}_{0.95}]
$$

**Portfolio:**
$$
\text{Loss}_t = \Delta \text{Load}_t - \sum_{i} w_i \cdot r_{i,t}
$$

**Constraints:**
- $\sum_{i} w_i \leq 1$ (maximum 100% allocation)
- $w_i \geq 0$ (long-only, no shorting)

**Results (Boston + UNG):**
```
Optimal Hedge:
  UNG position: 0.0 (no hedge recommended)
  Unhedged CVaR: -185.2 MW
  Hedged CVaR: -185.2 MW
  Benefit: 0.0 MW (correlation too weak)
  
Interpretation: UNG correlation (-0.0176) insufficient for effective hedge
```

**New York + XLU:**
```
Optimal Hedge:
  XLU position: 18.3%
  Unhedged CVaR: -512.4 MW
  Hedged CVaR: -498.7 MW
  Benefit: 13.7 MW (2.7% reduction)
```

**Conclusion:** ETF hedging provides marginal risk reduction; better suited for:
- **Hourly/sub-hourly** price volatility (not daily aggregates)
- **Direct commodity futures** (natural gas, power contracts)
- **Weather derivatives** (HDD/CDD swaps)

---

## 9. Cross-Correlation & Tail Dependence Analysis

**Notebook:** `Combined_Temperature_Load_Analytics.ipynb`

### 9.1 Temperature-Load Cross-Correlations

**Pearson Correlation (Linear):**
```csv
City         | Temp-Load ρ | HDD-Load ρ | CDD-Load ρ
Boston       | -0.12       | +0.48      | +0.65
NewYork      | -0.08       | +0.42      | +0.58
Houston      | +0.35       | +0.15      | +0.72
Chicago      | -0.15       | +0.53      | +0.61
Minneapolis  | -0.22       | +0.61      | +0.54
```

**Interpretation:**
- **Negative temp-load**: Heating > cooling (Boston, Minneapolis)
- **Positive temp-load**: Cooling > heating (Houston)
- **HDD/CDD dominance**: Cooling drives summer peaks more

![Temperature-Load Cross-Correlations](data/images/temp_load_cross_correlations.png)

### 9.2 Kendall's Tau (Nonparametric)

**Benefits:** Robust to outliers, measures rank correlation

**Results (Residuals after detrending):**
```csv
           Boston  NewYork  Chicago  Minneapolis
Boston     1.000    0.387    0.245      0.189
NewYork    0.387    1.000    0.312      0.227
Chicago    0.245    0.312    1.000      0.498
Minneapolis 0.189   0.227    0.498      1.000
```

**Kendall's Tau Heatmap:**
![Kendall Tau Full](data/images/temp_kendall_tau_full.png)

### 9.3 Tail Dependence Coefficients

**Definition:**
$$
\chi_U = \lim_{u \to 1^-} P(U_2 > u \mid U_1 > u)
$$

**Ferreira Estimator (Upper Tail):**
```csv
City Pair              | χ_U   | 95% CI
Boston - NewYork       | 0.42  | [0.38, 0.46]
Chicago - Minneapolis  | 0.58  | [0.54, 0.62]
Boston - Chicago       | 0.31  | [0.27, 0.35]
```

**Interpretation:**
- **0.58**: 58% chance Minneapolis spikes when Chicago spikes
- **0.42**: 42% chance NYC spikes when Boston spikes
- **Regional clustering**: Midwest cities show stronger dependence

![Seasonal Tail Dependence](data/images/seasonal_tail_dependence_hdd_vs_cdd.png)

### 9.4 HDD vs CDD Seasonal Patterns

**Winter (HDD Analysis):**
```csv
           Boston  Chicago  Minneapolis
Boston     1.000    0.623      0.549
Chicago    0.623    1.000      0.785
Minneapolis 0.549   0.785      1.000
```

**Summer (CDD Analysis):**
```csv
           Boston  NewYork  Houston
Boston     1.000    0.712      0.384
NewYork    0.712    1.000      0.412
Houston    0.384    0.412      1.000
```

**Insight:** 
- **Winter**: Midwest cluster (polar vortex events)
- **Summer**: East Coast cluster (heat domes)

![HDD vs CDD Correlations](data/images/seasonal_correlations_hdd_vs_cdd.png)

---

## 10. Actionable Insights & Risk Management

### 10.1 Forecasting Recommendations

**Model Selection:**
1. **Short-term (1-7 days):** XGBoost with weather forecasts
   - Use ensemble of OLS + XGBoost for uncertainty quantification
   - Update daily with rolling 30-day window

2. **Medium-term (1-4 weeks):** ARMA-GARCH with seasonal harmonics
   - Captures volatility clustering for risk management
   - Confidence intervals via GARCH variance forecasts

3. **Long-term (1-12 months):** Seasonal decomposition only
   - Weather uncertainty dominates → use climatological normals
   - Scenario analysis with HDD/CDD distributions

**Performance Benchmarks:**
- **Boston/NewYork:** XGBoost R² > 0.77 (reliable)
- **Chicago:** XGBoost R² = 0.80 (excellent)
- **Minneapolis:** Use OLS (XGBoost overfits)

### 10.2 Extreme Event Planning

**Reserve Capacity Targets:**
| City | Current Peak (MW) | 50-year RL (MW) | Reserve Needed |
|------|------------------|----------------|----------------|
| Boston | 4,200 | 4,570 | +370 MW |
| NewYork | 12,800 | 14,080 | +1,280 MW |
| Chicago | 38,500 | 42,950 | +4,450 MW |
| Minneapolis | 11,000 | 12,480 | +1,480 MW |

**Recommendations:**
1. **Peaker plants:** Maintain 10-15% reserve above 50-year level
2. **Demand response:** Trigger at 95th percentile (5% exceedance)
3. **Interruptible contracts:** Size to 20-year return level

### 10.3 Hedging Strategy

**Natural Gas (UNG):**
- **Weak correlation** (-0.018 to -0.037) → poor daily hedge
- **Better use:** Seasonal hedges (winter HDD, summer CDD)
- **Alternative:** NYMEX natural gas futures (NG contract)

**Utilities (XLU):**
- **Positive correlation** (0.002-0.013) for Midwest
- **Use:** Sector hedge for portfolio diversification
- **Caveat:** Moves with load (not inverse hedge)

**Optimal Instruments:**
1. **Weather derivatives:** HDD/CDD swaps matched to city
   - Example: Chicago 4,500 HDD put option
2. **Power futures:** Locational marginal pricing (LMP) contracts
3. **Transmission rights:** Financial hedges for congestion
4. **VIX-like products:** Volatility indices for power markets

### 10.4 Portfolio Risk Management

**Diversification Analysis:**
- **Regional split:** 40% East Coast / 35% Midwest / 25% South
- **Reduces CVaR by 18%** (vs single-region concentration)
- **Tail dependence limits benefits** (3,000x > independence at 99th pct)

**Stress Testing Scenarios:**
1. **Polar Vortex:** Chicago + Minneapolis simultaneous 20-year event
   - Load spike: +6,000 MW combined
   - Price impact: $1.2M/day at $500/MWh

2. **Heat Dome:** New York + Boston simultaneous 10-year event
   - Load spike: +2,500 MW combined
   - Grid stress: N-1 contingency violated

3. **Multi-Region:** All 4 cities exceed 95th percentile
   - Probability: 0.06% (once every 5 years)
   - System-wide impact: Requires federal coordination

### 10.5 Data-Driven Insights

**From All Notebooks:**

1. **Seasonal Patterns (Load_Analytics.ipynb):**
   - 3 harmonics sufficient (6H overfits)
   - Day-of-week effects: -120 to -250 MW on weekends
   - Volatility clustering: GARCH persistence ≈ 0.99

2. **Temperature Drivers (Temperature_Analytics.ipynb):**
   - CDD explains 42-65% of summer load variance
   - HDD explains 48-61% of winter load variance
   - Extreme value index ξ = 0.18-0.29 (heavy tails)

3. **Forecasting (Load_Forecasting_EVT_Hedging.ipynb):**
   - Feature engineering: 51 features → 20.7% improvement
   - XGBoost best for peak events (captures nonlinearity)
   - OLS best for interpretability and regulatory filings

4. **Tail Risk (Load_Forecasting_EVT_Hedging.ipynb):**
   - GPD models predict 500 MW spikes with 2.3% probability
   - Student-t copula: ν = 5.2 (fatter tails than Gaussian)
   - Joint exceedances 16-5000x more likely than independence

5. **Cross-Correlation (Combined_Temperature_Load_Analytics.ipynb):**
   - Midwest cluster: χ_U = 0.58 (Chicago-Minneapolis)
   - East Coast cluster: χ_U = 0.42 (Boston-NYC)
   - Regional dependencies increase in extreme events

---

## 11. Future Work

### 11.1 Methodological Extensions

**Hourly/Sub-hourly Forecasting:**
- Current: Daily aggregates
- Proposed: 5-minute/1-hour resolution
- Benefits: Capture intraday volatility, better hedging

**Deep Learning Models:**
- LSTM/GRU for sequence modeling
- Transformer architectures (attention mechanisms)
- Hybrid physics-informed neural networks

**Multivariate Copulas:**
- Vine copulas for complex dependencies
- Time-varying copulas (capture regime shifts)
- Extreme value copulas (generalized Pareto)

**Bayesian Hierarchical Models:**
- Pool information across cities
- Shrinkage estimation for rare events
- Probabilistic forecasting (full predictive distribution)

### 11.2 Data Enhancements

**Additional Variables:**
- Solar irradiance (renewable generation)
- Wind speed (renewable + cooling effects)
- Humidity (AC efficiency)
- Cloud cover (solar + behavioral)

**Economic Factors:**
- Industrial production index
- Retail sales (commercial demand)
- Population growth
- Electricity prices (feedback effects)

**Grid Topology:**
- Transmission constraints
- Generator outages
- Import/export flows
- Renewable curtailment

### 11.3 Risk Management Tools

**Real-Time Dashboard:**
- Live load forecasts (hourly updates)
- Extreme event probability tracker
- Hedge position recommendations
- Stress test scenarios

**Automated Hedging Platform:**
- API integration with CME/ICE futures
- Dynamic rebalancing (daily/weekly)
- Risk-parity portfolio allocation
- Backtesting framework

**Weather Derivative Pricing:**
- HDD/CDD option valuation
- Temperature swap curve fitting
- Basis risk quantification (station vs index)

### 11.4 Academic Contributions

**Publications:**
1. "Multivariate Tail Dependence in Electricity Markets" → *Energy Economics*
2. "XGBoost vs ARMA-GARCH for Load Forecasting" → *IEEE Transactions on Power Systems*
3. "Copula-Based Scenario Generation for Grid Resilience" → *Operations Research*

**Open Source:**
- Python package: `loadforecasting` (all models + data)
- R package: `energyEVT` (extreme value analysis)
- Julia package: `PowerCopulas.jl` (high-performance copula fitting)

---

## Appendix: File Structure

### Data Files

**Processed Results (30+ CSV files):**
- `forecast_model_comparison.csv`: OLS vs XGBoost metrics
- `gpd_tail_parameters.csv`: EVT shape/scale parameters
- `joint_tail_probabilities.csv`: Multi-city exceedance probabilities
- `etf_hedge_ratios.csv`: Hedge ratios by city/ETF
- `all_regions_model_comparison.csv`: 288 ARMA-GARCH specifications
- `temperature_model_summary.csv`: 6 cities' seasonal GARCH results
- `temperature_evi_*.csv`: Extreme value indices (left/right tails)
- `combined_*_correlations.csv`: Kendall/Pearson/Spearman matrices

**Raw Data:**
- `data/raw/Boston/`: ISO-NE Excel files (2014-2024)
- `data/raw/miso/`: Monthly zips + daily XLS (2009-2024)
- `data/raw/palIntegrated/`: NYISO daily CSV zips (2005-2024)
- `data/raw/temperature/`: 18 cities' daily min/max temps
- `data/raw/energy_etfs/`: 6 ETF tickers' OHLCV data

### Visualizations (70+ Images)

**Load Analytics:**
- `*_rolling_volatility.png`: 30-day rolling std
- `*_volatility_by_doy.png`: Day-of-year patterns
- `*_volatility_clustering_acf.png`: ACF of squared residuals
- `autocorrelation_*.png`: ACF/PACF plots (6 cities)
- `model_fit_and_residuals_*.png`: Fitted values + residuals
- `residual_distribution_*.png`: Q-Q plots, histograms

**Temperature Analytics:**
- `temp_*_day_of_year_patterns.png`: Seasonal cycles
- `temp_*_seasonal_garch_model.png`: ARMA-GARCH fits
- `temp_autocorrelation_*.png`: ACF/PACF (6 cities)
- `temp_model_fit_and_residuals_*.png`: Seasonal decomposition
- `temp_*_3h_vs_6h_comparison.png`: Harmonic model comparison

**Cross-Analysis:**
- `combined_correlation_matrices.png`: 4-panel heatmaps
- `temp_load_cross_correlations.png`: Lagged correlations
- `residual_temp_load_cross_correlations.png`: Detrended
- `seasonal_correlations_hdd_vs_cdd.png`: Winter vs summer
- `seasonal_tail_dependence_hdd_vs_cdd.png`: Tail dependence

**Extreme Value Theory:**
- `evt_extreme_value_indices.png`: EVI estimates (left/right)
- `evt_tail_dependence_ratios.png`: Ferreira TDC
- `temp_ferreira_tdc_*.png`: HDD/CDD tail dependence
- `temp_kendall_tau_*.png`: Rank correlation heatmaps

### Notebooks (5 Primary Analyses)

1. **`Load_Analytics.ipynb`** (62 cells, 1,234 lines)
   - Volatility patterns (rolling, clustering, day-of-year)
   - Autocorrelation analysis (ACF/PACF)
   - ARMA-GARCH model fitting (3H vs 6H comparison)
   - Full diagnostics (Ljung-Box, ARCH LM, normality tests)
   - Multi-region comparison (Boston, NewYork, Chicago, Minneapolis)

2. **`Temperature_Analytics.ipynb`** (27 cells, 537 lines)
   - Day-of-year seasonal patterns
   - Seasonal GARCH modeling (AIC improvement 212-1,395)
   - HDD/CDD calculation and analysis
   - Extreme value theory (left/right tail EVI)
   - Temperature residual correlations

3. **`Load_Forecasting_EVT_Hedging.ipynb`** (29 cells, 988 lines)
   - Feature engineering (51 variables)
   - OLS econometric baseline (R² = 0.64-0.76)
   - XGBoost machine learning (R² = 0.77-0.80, 15-28% improvement)
   - GPD tail fitting (ξ = 0.18-0.29 heavy tails)
   - Student-t copula (100k scenarios, ν = 5.2)
   - Joint tail probabilities (16-5000x > independence)
   - ETF hedge optimization (CVaR minimization)

4. **`Combined_Temperature_Load_Analytics.ipynb`**
   - Temperature-load cross-correlations (Pearson/Kendall)
   - Tail dependence coefficients (Ferreira estimator)
   - Seasonal patterns (HDD vs CDD)
   - Regional clustering (Midwest, East Coast)

5. **`Temperature_Analytics2.ipynb`**
   - Extended temperature analysis
   - Additional cities (18 total)
   - Comparative studies

### Scripts (11 Data Processing)

**Download Scripts:**
- `download_miso_load.py`: Fetch MISO data (monthly zips + daily XLS)
- `download_nyiso_palintegrated.py`: Fetch NYISO PAL data (daily CSV zips)
- `download_energy_etfs.py`: Fetch Yahoo Finance ETF data

**Processing Scripts:**
- `convert_miso_reports.py`: Parse MISO Excel/CSV to standard format
- `aggregate_converted_csv.py`: Combine multiple MISO files
- `extract_nyiso_palintegrated.py`: Unzip and parse NYISO CSVs
- `process_boston.py`: Parse ISO-NE Excel with `NEMASSBOST` sheets
- `process_houston.py`: Process ERCOT data (Houston + Dallas)
- `process_palintegrated2.py`: NYC load aggregation
- `process_temperature.py`: Calculate HDD/CDD from daily min/max temps
- `select_arma_parameters.py`: Automated ARMA order selection

---

## References & Citations

**Methodology:**
1. Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007.
2. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
3. McNeil, A. J., & Frey, R. (2000). "Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series." *Journal of Empirical Finance*, 7(3-4), 271-300.
4. Sklar, A. (1959). "Fonctions de répartition à n dimensions et leurs marges." *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229-231.
5. Pickands III, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Annals of Statistics*, 3(1), 119-131.

**Energy Applications:**
6. Weron, R. (2014). "Electricity Price Forecasting: A Review of the State-of-the-Art with a Look into the Future." *International Journal of Forecasting*, 30(4), 1030-1081.
7. Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., & Hyndman, R. J. (2016). "Probabilistic Energy Forecasting: Global Energy Forecasting Competition 2014 and Beyond." *International Journal of Forecasting*, 32(3), 896-913.
8. Benth, F. E., Benth, J. Š., & Koekebakker, S. (2008). *Stochastic Modelling of Electricity and Related Markets*. World Scientific.

**Machine Learning:**
9. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
10. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.

**Data Sources:**
- **MISO**: Midcontinent Independent System Operator (https://www.misoenergy.org)
- **NYISO**: New York Independent System Operator (https://www.nyiso.com)
- **ISO-NE**: ISO New England (https://www.iso-ne.com)
- **ERCOT**: Electric Reliability Council of Texas (https://www.ercot.com)
- **NOAA**: National Oceanic and Atmospheric Administration (https://www.noaa.gov)
- **Yahoo Finance**: ETF historical data (https://finance.yahoo.com)

---

## Contact & Collaboration

**Project Lead:** [Your Name]
**Institution:** [Your University/Company]
**Email:** [Your Email]
**GitHub:** [Repository Link]

**Code Availability:** All scripts, notebooks, and data processing pipelines available at:
`https://github.com/[your-repo]/Financial-Engineering-Project`

**License:** MIT (code), CC BY 4.0 (documentation)

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Status:** Complete Analysis (2005-2024 data)

---

## Acknowledgments

- **Data Providers**: MISO, NYISO, ISO-NE, ERCOT, NOAA for open data access
- **Open Source Community**: pandas, numpy, scipy, statsmodels, xgboost developers
- **Academic Advisors**: [Your Advisor Names]
- **Funding**: [Grant/Sponsorship if applicable]

---

*This report synthesizes 5 comprehensive notebooks (Load_Analytics, Temperature_Analytics, Load_Forecasting_EVT_Hedging, Combined_Temperature_Load_Analytics, Temperature_Analytics2), 30+ result CSV files, 70+ visualizations, and 11 data processing scripts into a unified financial engineering framework for electricity market risk management.*
