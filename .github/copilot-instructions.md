# Financial Engineering Project - AI Coding Agent Instructions

## Project Overview
This is a financial engineering project focused on **electricity load analytics** and **weather derivatives pricing**. The codebase analyzes seasonal patterns, volatility clustering, and extreme value behavior in electricity demand and temperature data across multiple US regions (Boston, New York, Houston, Chicago, Dallas, Minneapolis).

## Core Architecture

### Data Pipeline (3 Stages)
1. **Download** (`scripts/download_*.py`): Fetch raw data from MISO, NYISO, and weather APIs
2. **Process** (`scripts/process_*.py`): Convert Excel/CSV files to standardized daily summaries with `[date, min_load, max_load, avg_load]` columns
3. **Analyze** (`notebooks/*.ipynb`): Statistical modeling using seasonal harmonics, ARMA-GARCH models, and extreme value theory

### Key Data Sources
- **MISO**: Monthly zips (2009-2022) + daily XLS (2023-2024) → `data/raw/miso/`
- **NYISO**: Daily `palIntegrated` CSV zips → `data/raw/palIntegrated/`
- **Boston**: ISO-NE Excel files with `NEMASSBOST` or `NEMA` sheets → `data/raw/Boston/`
- **Temperature**: Daily min/max temps from raw CSV files → `data/raw/temperature/`

## Statistical Modeling Framework

### Seasonal Models (`notebooks/seasonal_models.py`)
The project uses a **two-stage modeling approach**:

1. **Mean Model**: `fit_seasonal_garch_model()` uses SARIMAX with:
   - Harmonic terms: `sin(2πn·doy/365.25)` and `cos(2πn·doy/365.25)` for n=1..N
   - Day-of-week dummies (Tue-Sun, Monday as reference)
   - ARMA(p,q) for autocorrelation
   
2. **Volatility Model**: GARCH(1,1) on residuals to capture time-varying volatility
   - `σ²[t] = ω + α·ε²[t-1] + β·σ²[t-1]`

**Model comparison uses AIC/BIC** to select optimal harmonic counts (typically 3H vs 6H).

### Critical Modeling Conventions
- **Day of year**: Use `.day_of_year` attribute (1-366) for seasonal patterns
- **Day of week**: `.dayofweek` (0=Monday, 6=Sunday) with Monday as reference category
- **Harmonics**: Always use `365.25` (not 365) to account for leap years
- **Design matrix**: Build with `np.ones()` for intercept, then append features column-wise
- **Least squares**: Use `np.linalg.lstsq(X, y, rcond=None)[0]` for parameter estimation

### Diagnostic Workflow
```python
# Standard analysis pattern from Load_Analytics.ipynb:
1. fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=1)
2. run_full_diagnostics(fit, region_name, save_dir)
   - Residual normality tests (D'Agostino-Pearson)
   - ACF/PACF for autocorrelation
   - Ljung-Box test on squared residuals (ARCH effects)
   - ARCH LM test for volatility clustering
3. compare_models() to evaluate different harmonic specifications
4. plot_aic_bic_comparison() for visual model selection
```

## Development Patterns

### File Organization
- **Processed data**: Always `[date, min_load, max_load, avg_load]` format (date is index)
- **Time range**: Filter to `2014-01-01` to `2022-12-31` in notebooks for consistency
- **Output paths**: Raw → `data/raw/`, Processed → `data/processed/`, Figures → `data/images/`

### Data Processing Scripts
- Use `Path` from `pathlib` for all file operations
- Implement `--skip-existing` flag to avoid re-downloading
- Add logging with `logging.info()` for progress tracking
- Handle multiple sheet names/formats (e.g., `NEMASSBOST` vs `NEMA` in Boston files)
- Use `tqdm` for progress bars on batch operations

### Notebook Conventions
- Import `seasonal_models` functions directly (it's a module, not a notebook)
- Use `sns.set_style('whitegrid')` for consistent plotting
- Store region data in dicts like `regions_dict = {'Boston': srs_boston, 'NewYork': srs_ny, ...}`
- Date filtering: `df = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2022-12-31')]`

## External Dependencies

### Key Libraries
- **statsmodels**: SARIMAX for seasonal ARMA models, diagnostic tests (Ljung-Box, ARCH LM)
- **scipy**: Optimization (`minimize`), distributions (GEV, Pareto), statistical tests
- **pandas**: All time series manipulation with `.day_of_year`, `.dayofweek` attributes
- **matplotlib/seaborn**: Plotting with consistent figure sizes `(14, 6)` or `(16, 12)` for multi-panel

### Weather Derivatives Module
The `notebooks/wxderivs/` package contains specialized tools:
- `fit_mean_std()`: Seasonal mean/std fitting with GARCH volatility
- `make_cdf_ppf()`: Empirical CDF/PPF for residual distributions
- `EllipticalCop`: Copula modeling for multi-variate temperature analysis
- `unibm.benchmark`: Benchmarking functions (imported separately)

## Running the Project

### Data Acquisition
```bash
# Download data (run in order)
python scripts/download_miso_load.py --start 2009-07-01 --end 2024-12-31 --skip-existing
python scripts/download_nyiso_palintegrated.py --start 2005-01-01 --end 2024-12-31 --skip-existing
```

### Data Processing
```bash
# Convert and aggregate MISO reports
python scripts/convert_miso_reports.py --format csv
python scripts/aggregate_converted_csv.py

# Process region-specific data
python scripts/process_boston.py
python scripts/process_houston.py  # Also processes Dallas
python scripts/process_palintegrated2.py  # NYC load
python scripts/process_temperature.py
```

### Analysis
Open `notebooks/Load_Analytics.ipynb` or `notebooks/Temperature_Analytics.ipynb` and run cells sequentially. Models are defined in `notebooks/seasonal_models.py`.

## Common Tasks

### Adding a New Region
1. Create download script following `download_*.py` pattern
2. Add processor to `scripts/process_<region>.py` outputting `[date, min_load, max_load, avg_load]` CSV
3. Load in notebook: `df = pd.read_csv('../data/processed/<Region>/<Region>.csv')`
4. Add to `regions_dict` and run `fit_all_regions(regions_dict)`

### Modifying Seasonal Models
- When changing `fit_seasonal_model()`, ensure consistency with `fit_seasonal_garch_model()`
- Both functions should accept `include_dayofweek` parameter
- Return dict must include: `params`, `y_pred`, `residuals`, `index`, `y`, diagnostics (AIC, BIC, R²)
- Use same design matrix construction pattern (intercept, trend, harmonics, day-of-week)

### Debugging Data Issues
- Check `column_names_log.csv` in MISO data for column mapping issues
- Verify date ranges with `.index.min()` and `.index.max()`
- Use `.dropna()` before fitting models to handle missing data
- Inspect sheet names with `pd.read_excel(file, sheet_name=None).keys()`
