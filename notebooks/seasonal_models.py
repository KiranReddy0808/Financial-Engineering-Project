"""
Seasonal modeling functions for electricity load analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def fit_seasonal_model(srs, n_harmonics=3):
    """
    Fit a seasonal model with trend and harmonics.
    
    Model: y = a0 + a1*t + sum(An*sin(2*pi*n*doy/365) + Bn*cos(2*pi*n*doy/365))
    
    Args:
        srs: pandas Series with datetime index
        n_harmonics: number of harmonic terms to include (default: 3)
    
    Returns:
        dict with fitted values, residuals, parameters, and model diagnostics
    """
    # Remove NaN values
    srs_clean = srs.dropna()
    
    # Time variables
    t = np.arange(len(srs_clean))  # Time index for trend
    doy = srs_clean.index.day_of_year.values  # Day of year for seasonality
    y = srs_clean.values
    
    # Build design matrix
    n_params = 2 + 2 * n_harmonics
    X = np.ones((len(srs_clean), n_params))
    X[:, 0] = 1  # Intercept
    X[:, 1] = t  # Linear trend
    
    # Add harmonics
    for n in range(1, n_harmonics + 1):
        X[:, 2 + 2*(n-1)] = np.sin(2 * np.pi * n * doy / 365.25)
        X[:, 2 + 2*(n-1) + 1] = np.cos(2 * np.pi * n * doy / 365.25)
    
    # Fit using least squares
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Predictions and residuals
    y_pred = X @ params
    residuals = y - y_pred
    
    # Calculate log-likelihood (assuming normal errors)
    n = len(residuals)
    sigma2 = np.var(residuals, ddof=n_params)
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    
    # Calculate AIC and BIC
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood
    
    # Calculate R²
    r_squared = 1 - np.var(residuals) / np.var(y)
    
    return {
        'params': params,
        'y_pred': y_pred,
        'residuals': residuals,
        'index': srs_clean.index,
        'y': y,
        't': t,
        'doy': doy,
        'n_harmonics': n_harmonics,
        'n_params': n_params,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'r_squared': r_squared,
        'residual_std': np.std(residuals),
    }


def fit_seasonal_garch_model(srs, n_harmonics=3):
    """
    Fit a two-stage model:
    1. Seasonal model (trend + harmonics) for the mean
    2. GARCH(1,1) model for the volatility of residuals
    
    Args:
        srs: pandas Series with datetime index
        n_harmonics: number of harmonic terms to include (default: 3)
    
    Returns:
        dict with fitted mean, residuals, time-varying volatility, and diagnostics
    """
    # Stage 1: Fit seasonal model
    srs_clean = srs.dropna()
    
    # Time variables
    t = np.arange(len(srs_clean))
    doy = srs_clean.index.day_of_year.values
    y = srs_clean.values
    
    # Build design matrix for mean
    n_mean_params = 2 + 2 * n_harmonics
    X = np.ones((len(srs_clean), n_mean_params))
    X[:, 0] = 1  # Intercept
    X[:, 1] = t  # Linear trend
    
    # Add harmonics
    for n in range(1, n_harmonics + 1):
        X[:, 2 + 2*(n-1)] = np.sin(2 * np.pi * n * doy / 365.25)
        X[:, 2 + 2*(n-1) + 1] = np.cos(2 * np.pi * n * doy / 365.25)
    
    # Fit mean model
    mean_params = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ mean_params
    residuals = y - y_pred
    
    # Stage 2: Fit GARCH(1,1) to residuals
    # GARCH model: σ²[t] = ω + α*ε²[t-1] + β*σ²[t-1]
    
    def garch_likelihood(params):
        """Negative log-likelihood for GARCH(1,1)"""
        omega, alpha, beta = params
        
        # Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        n = len(residuals)
        variance = np.zeros(n)
        variance[0] = np.var(residuals)
        
        # Compute variance recursion
        for i in range(1, n):
            variance[i] = omega + alpha * residuals[i-1]**2 + beta * variance[i-1]
        
        # Negative log-likelihood (assuming normal errors)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + residuals**2 / variance)
        
        return -log_likelihood
    
    # Initial parameter guesses
    initial_params = [0.1, 0.1, 0.8]  # [ω, α, β]
    
    # Bounds: ω > 0, 0 ≤ α, β ≤ 1, α + β < 1
    bounds = [(1e-6, 1.0), (0, 0.99), (0, 0.99)]
    
    # Optimize
    result = minimize(
        garch_likelihood,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    # Extract fitted parameters
    omega, alpha, beta = result.x
    n_garch_params = 3
    
    # Compute fitted volatilities
    fitted_volatility = np.zeros(len(residuals))
    fitted_volatility[0] = np.sqrt(np.var(residuals))
    
    for i in range(1, len(residuals)):
        variance = omega + alpha * residuals[i-1]**2 + beta * fitted_volatility[i-1]**2
        fitted_volatility[i] = np.sqrt(variance)
    
    # Calculate log-likelihood for full model
    variance_series = fitted_volatility ** 2
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance_series) + residuals**2 / variance_series)
    
    # Total number of parameters
    n_total_params = n_mean_params + n_garch_params
    n = len(residuals)
    
    # Calculate AIC and BIC
    aic = 2 * n_total_params - 2 * log_likelihood
    bic = n_total_params * np.log(n) - 2 * log_likelihood
    
    # Calculate R²
    r_squared = 1 - np.var(residuals) / np.var(y)
    
    return {
        'mean_params': mean_params,
        'garch_params': {'omega': omega, 'alpha': alpha, 'beta': beta},
        'y_pred': y_pred,
        'residuals': residuals,
        'volatility': fitted_volatility,
        'index': srs_clean.index,
        'y': y,
        't': t,
        'doy': doy,
        'n_harmonics': n_harmonics,
        'n_mean_params': n_mean_params,
        'n_garch_params': n_garch_params,
        'n_total_params': n_total_params,
        'persistence': alpha + beta,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'r_squared': r_squared,
        'residual_std': np.std(residuals),
    }


def compare_models(region_name, srs, harmonics_list=[3, 6]):
    """
    Compare seasonal models with different numbers of harmonics.
    
    Args:
        region_name: name of the region for display
        srs: pandas Series with datetime index
        harmonics_list: list of harmonic counts to test (default: [3, 6])
    
    Returns:
        pandas DataFrame with model comparison results
    """
    results = []
    
    for n_harmonics in harmonics_list:
        # Fit model
        fit = fit_seasonal_garch_model(srs, n_harmonics=n_harmonics)
        
        # Store results
        results.append({
            'Region': region_name,
            'Model': f'{n_harmonics}H + GARCH(1,1)',
            'N_Harmonics': n_harmonics,
            'N_Params': fit['n_total_params'],
            'R²': fit['r_squared'],
            'Residual_Std': fit['residual_std'],
            'Log_Likelihood': fit['log_likelihood'],
            'AIC': fit['aic'],
            'BIC': fit['bic'],
            'GARCH_ω': fit['garch_params']['omega'],
            'GARCH_α': fit['garch_params']['alpha'],
            'GARCH_β': fit['garch_params']['beta'],
            'Persistence': fit['persistence'],
            'Avg_Volatility': fit['volatility'].mean(),
        })
    
    return pd.DataFrame(results)


def fit_all_regions(regions_dict, harmonics_list=[3, 6]):
    """
    Fit seasonal + GARCH models to all regions and compare.
    
    Args:
        regions_dict: dict of {region_name: pandas Series}
        harmonics_list: list of harmonic counts to test (default: [3, 6])
    
    Returns:
        pandas DataFrame with comparison results for all regions
    """
    all_results = []
    
    for region_name, srs in regions_dict.items():
        region_results = compare_models(region_name, srs, harmonics_list)
        all_results.append(region_results)
    
    return pd.concat(all_results, ignore_index=True)


def plot_model_comparison(fit_3h, fit_6h, region_name, save_path=None):
    """
    Create side-by-side comparison visualization of 3H vs 6H models.
    
    Args:
        fit_3h: fitted model with 3 harmonics
        fit_6h: fitted model with 6 harmonics
        region_name: name of the region for plot titles
        save_path: optional path to save the figure
    
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
    
    models = [
        ('3H + GARCH(1,1)', fit_3h),
        ('6H + GARCH(1,1)', fit_6h),
    ]
    
    for col, (model_name, fit) in enumerate(models):
        # Row 1: Residuals vs DOY
        axes[0, col].scatter(fit['doy'], fit['residuals'], s=0.5, alpha=0.3, color='red')
        residuals_mean = pd.Series(fit['residuals'], index=fit['index']).groupby(lambda x: x.day_of_year).mean()
        axes[0, col].plot(residuals_mean.index, residuals_mean.values, color='blue', lw=2)
        axes[0, col].axhline(0, color='black', linestyle='--', lw=1)
        axes[0, col].set_ylabel('Residuals (MWh)')
        axes[0, col].set_title(f'{model_name}\nResiduals vs DOY', fontsize=10)
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_xlim(1, 366)
        
        # Row 2: Residuals with volatility bands (time series)
        axes[1, col].scatter(fit['index'], fit['residuals'], s=0.3, alpha=0.3, color='red')
        axes[1, col].plot(fit['index'], fit['volatility'], color='green', lw=0.8, linestyle='--')
        axes[1, col].plot(fit['index'], -fit['volatility'], color='green', lw=0.8, linestyle='--')
        axes[1, col].axhline(0, color='black', linestyle='-', lw=0.8, alpha=0.5)
        axes[1, col].set_ylabel('Residuals (MWh)')
        axes[1, col].set_title('Residuals with Volatility Bands', fontsize=10)
        axes[1, col].grid(True, alpha=0.3)
        
        # Row 3: Volatility by DOY
        vol_by_doy = pd.Series(fit['volatility'], index=fit['index']).groupby(lambda x: x.day_of_year).mean()
        axes[2, col].plot(vol_by_doy.index, vol_by_doy.values, color='darkgreen', lw=2)
        axes[2, col].fill_between(vol_by_doy.index, 0, vol_by_doy.values, alpha=0.3, color='green')
        axes[2, col].set_xlabel('Day of Year')
        axes[2, col].set_ylabel('Avg Volatility (MWh)')
        axes[2, col].set_title('Average Volatility by DOY', fontsize=10)
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].set_xlim(1, 366)
        
        # Add stats text box
        stats_text = (f"R² = {fit['r_squared']:.4f}\n"
                      f"Resid Std = {fit['residual_std']:.2f}\n"
                      f"AIC = {fit['aic']:.1f}\n"
                      f"BIC = {fit['bic']:.1f}\n"
                      f"Avg Vol = {fit['volatility'].mean():.2f}\n"
                      f"α+β = {fit['persistence']:.4f}")
        axes[2, col].text(0.02, 0.98, stats_text, transform=axes[2, col].transAxes, 
                         fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', 
                         facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{region_name} Model Comparison: 3H vs 6H + GARCH(1,1)', fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_aic_bic_comparison(all_regions_comparison, regions_dict, save_path=None):
    """
    Create bar charts comparing AIC and BIC across all regions.
    
    Args:
        all_regions_comparison: DataFrame with model comparison results
        regions_dict: dict of {region_name: pandas Series}
        save_path: optional path to save the figure
    
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
    
    regions_list = list(regions_dict.keys())
    x_pos = np.arange(len(regions_list))
    width = 0.35
    
    # Prepare data for plotting
    aic_3h = []
    aic_6h = []
    bic_3h = []
    bic_6h = []
    
    for region in regions_list:
        region_data = all_regions_comparison[all_regions_comparison['Region'] == region]
        aic_3h.append(region_data[region_data['N_Harmonics'] == 3]['AIC'].values[0])
        aic_6h.append(region_data[region_data['N_Harmonics'] == 6]['AIC'].values[0])
        bic_3h.append(region_data[region_data['N_Harmonics'] == 3]['BIC'].values[0])
        bic_6h.append(region_data[region_data['N_Harmonics'] == 6]['BIC'].values[0])
    
    # Plot AIC
    axes[0].bar(x_pos - width/2, aic_3h, width, label='3H + GARCH', color='steelblue', alpha=0.8)
    axes[0].bar(x_pos + width/2, aic_6h, width, label='6H + GARCH', color='coral', alpha=0.8)
    axes[0].set_xlabel('Region')
    axes[0].set_ylabel('AIC')
    axes[0].set_title('AIC Comparison Across Regions (Lower is Better)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(regions_list, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot BIC
    axes[1].bar(x_pos - width/2, bic_3h, width, label='3H + GARCH', color='steelblue', alpha=0.8)
    axes[1].bar(x_pos + width/2, bic_6h, width, label='6H + GARCH', color='coral', alpha=0.8)
    axes[1].set_xlabel('Region')
    axes[1].set_ylabel('BIC')
    axes[1].set_title('BIC Comparison Across Regions (Lower is Better)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(regions_list, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Calculate percentage improvement from 3H to 6H
    print("\n" + "=" * 80)
    print("IMPROVEMENT FROM 3H TO 6H (Negative = Better)")
    print("=" * 80)
    for i, region in enumerate(regions_list):
        aic_improvement = ((aic_6h[i] - aic_3h[i]) / aic_3h[i]) * 100
        bic_improvement = ((bic_6h[i] - bic_3h[i]) / bic_3h[i]) * 100
        print(f"{region:<15} | AIC: {aic_improvement:+6.2f}% | BIC: {bic_improvement:+6.2f}%")
    print("=" * 80)
    
    return fig


# ===========================================================================================
# DIAGNOSTIC AND ANALYSIS FUNCTIONS
# ===========================================================================================

def plot_fitted_model_and_residuals(fit, region_name, save_path=None):
    """
    Visualize fitted model and residuals in a 3-panel plot.
    
    Args:
        fit: fitted model dictionary (from fit_seasonal_model or fit_seasonal_garch_model)
        region_name: name of the region for plot titles
        save_path: optional path to save the figure
    
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12))
    
    # Top panel: Observed and fitted
    axes[0].scatter(fit['index'], fit['y'], s=1, alpha=0.5, color='blue', label='Observed')
    axes[0].plot(fit['index'], fit['y_pred'], color='red', lw=1.5, label='Fitted', alpha=0.8)
    axes[0].set_ylabel('Load (MWh)')
    axes[0].set_title(f'{region_name} Electricity Load - Observed vs Fitted (Seasonal + Trend Model)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Middle panel: Residuals over time
    axes[1].scatter(fit['index'], fit['residuals'], s=1, alpha=0.5, color='red')
    axes[1].axhline(0, color='black', linestyle='--', lw=1)
    axes[1].axhline(fit['residuals'].std(), color='gray', linestyle=':', lw=1, label='+/- 1 std')
    axes[1].axhline(-fit['residuals'].std(), color='gray', linestyle=':', lw=1)
    axes[1].set_ylabel('Residuals (MWh)')
    axes[1].set_title('Model Residuals Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Bottom panel: Residuals vs day of year
    axes[2].scatter(fit['doy'], fit['residuals'], s=1, alpha=0.5, color='green')
    axes[2].axhline(0, color='black', linestyle='--', lw=1)
    axes[2].set_xlabel('Day of Year')
    axes[2].set_ylabel('Residuals (MWh)')
    axes[2].set_title('Model Residuals vs Day of Year')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 366)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_residual_distribution(fit, region_name, save_path=None):
    """
    Analyze and visualize residual distribution with histogram and Q-Q plot.
    
    Args:
        fit: fitted model dictionary
        region_name: name of the region for plot titles
        save_path: optional path to save the figure
    
    Returns:
        matplotlib figure object and statistical test results
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(fit['residuals'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Residuals (MWh)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{region_name} - Residuals Histogram')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(fit['residuals'], dist="norm", plot=axes[1])
    axes[1].set_title(f'{region_name} - Q-Q Plot (Normal Distribution)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Statistical tests
    from scipy.stats import normaltest
    stat_normaltest, p_normaltest = normaltest(fit['residuals'])
    
    print(f"\n{'='*60}")
    print(f"{region_name} - RESIDUAL NORMALITY TEST")
    print(f"{'='*60}")
    print(f"D'Agostino-Pearson test: statistic={stat_normaltest:.4f}, p-value={p_normaltest:.4e}")
    print(f"Reject normality (p < 0.05): {p_normaltest < 0.05}")
    print(f"{'='*60}\n")
    
    return fig, {'statistic': stat_normaltest, 'p_value': p_normaltest}


def plot_autocorrelation(fit, region_name, lags=50, save_path=None):
    """
    Plot ACF and PACF of residuals to check for autocorrelation.
    
    Args:
        fit: fitted model dictionary
        region_name: name of the region for plot titles
        lags: number of lags to display (default: 50)
        save_path: optional path to save the figure
    
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    
    plot_acf(fit['residuals'], lags=lags, ax=axes[0])
    axes[0].set_title(f'{region_name} - Autocorrelation Function (ACF) of Residuals')
    
    plot_pacf(fit['residuals'], lags=lags, ax=axes[1])
    axes[1].set_title(f'{region_name} - Partial Autocorrelation Function (PACF) of Residuals')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def analyze_volatility_clustering(fit, region_name, save_dir=None):
    """
    Comprehensive volatility clustering analysis with multiple visualizations and tests.
    
    Args:
        fit: fitted model dictionary
        region_name: name of the region for plot titles
        save_dir: optional directory to save figures (will create multiple files)
    
    Returns:
        dict with test results and figures
    """
    residuals = fit['residuals']
    index = fit['index']
    
    results = {}
    
    # ===========================
    # 1. Visual Inspection: Absolute and Squared Residuals
    # ===========================
    fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12))
    
    # Original residuals
    axes[0].plot(index, residuals, lw=0.5, alpha=0.7)
    axes[0].set_ylabel('Residuals (MWh)')
    axes[0].set_title(f'{region_name} - Residuals Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='red', linestyle='--', lw=1)
    
    # Absolute residuals (proxy for volatility)
    abs_residuals = np.abs(residuals)
    axes[1].plot(index, abs_residuals, lw=0.5, alpha=0.7, color='orange')
    axes[1].set_ylabel('|Residuals| (MWh)')
    axes[1].set_title('Absolute Residuals Over Time (Volatility Proxy)')
    axes[1].grid(True, alpha=0.3)
    
    # Squared residuals (variance proxy)
    squared_residuals = residuals ** 2
    axes[2].plot(index, squared_residuals, lw=0.5, alpha=0.7, color='red')
    axes[2].set_ylabel('Residuals² (MWh²)')
    axes[2].set_title('Squared Residuals Over Time (Variance Proxy)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Date')
    
    plt.tight_layout()
    
    if save_dir:
        fig1.savefig(f'{save_dir}/{region_name}_volatility_over_time.png', dpi=150, bbox_inches='tight')
    
    results['fig_volatility_time'] = fig1
    
    # ===========================
    # 2. ACF/PACF of Squared Residuals
    # ===========================
    fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    
    plot_acf(squared_residuals, lags=50, ax=axes[0])
    axes[0].set_title(f'{region_name} - ACF of Squared Residuals')
    axes[0].set_ylabel('Autocorrelation')
    
    plot_pacf(squared_residuals, lags=50, ax=axes[1])
    axes[1].set_title(f'{region_name} - PACF of Squared Residuals')
    axes[1].set_ylabel('Partial Autocorrelation')
    
    plt.tight_layout()
    
    if save_dir:
        fig2.savefig(f'{save_dir}/{region_name}_volatility_clustering_acf.png', dpi=150, bbox_inches='tight')
    
    results['fig_acf_squared'] = fig2
    
    # ===========================
    # 3. Ljung-Box Test on Squared Residuals
    # ===========================
    lb_result_squared = acorr_ljungbox(squared_residuals, lags=20, return_df=True)
    
    print(f"\n{'='*80}")
    print(f"{region_name} - LJUNG-BOX TEST (Test for ARCH Effects)")
    print(f"{'='*80}")
    print(lb_result_squared.head(10))
    print("\nIf p-values < 0.05, reject null hypothesis of no autocorrelation")
    print("This indicates presence of ARCH effects (volatility clustering)")
    
    # Count significant lags
    significant_lags = (lb_result_squared['lb_pvalue'] < 0.05).sum()
    print(f"\nNumber of significant lags (out of 20): {significant_lags}")
    if significant_lags > 5:
        print("⚠️  Strong evidence of volatility clustering!")
    elif significant_lags > 2:
        print("⚠️  Moderate evidence of volatility clustering")
    else:
        print("✓ Little evidence of volatility clustering")
    print(f"{'='*80}\n")
    
    results['ljungbox_test'] = lb_result_squared
    results['significant_lags'] = significant_lags
    
    # ===========================
    # 4. ARCH LM Test
    # ===========================
    print(f"{'='*80}")
    print(f"{region_name} - ARCH LM TEST (Engle's Test)")
    print(f"{'='*80}")
    
    arch_results = []
    for lag in [1, 5, 10, 20]:
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=lag)
        arch_results.append({
            'lag': lag,
            'lm_stat': lm_stat,
            'lm_pvalue': lm_pvalue,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue
        })
        print(f"\nLag {lag}:")
        print(f"  LM Statistic: {lm_stat:.4f}, p-value: {lm_pvalue:.4e}")
        print(f"  F Statistic:  {f_stat:.4f}, p-value: {f_pvalue:.4e}")
        if lm_pvalue < 0.05:
            print(f"  ⚠️  Reject null hypothesis - ARCH effects present at lag {lag}")
        else:
            print(f"  ✓ Cannot reject null hypothesis - No ARCH effects at lag {lag}")
    
    print(f"\n{'='*80}")
    print("Null Hypothesis: No ARCH effects (no volatility clustering)")
    print("If p-value < 0.05: Reject null → ARCH effects present → Volatility clustering exists")
    print(f"{'='*80}\n")
    
    results['arch_test'] = arch_results
    
    # ===========================
    # 5. Rolling Standard Deviation
    # ===========================
    window = 30  # 30-day rolling window
    
    rolling_std = pd.Series(residuals, index=index).rolling(window=window).std()
    
    fig3, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(rolling_std.index, rolling_std.values, lw=1.5, color='darkred')
    ax.fill_between(rolling_std.index, 0, rolling_std.values, alpha=0.3, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Std Dev (MWh)')
    ax.set_title(f'{region_name} - Rolling {window}-Day Standard Deviation of Residuals\n(Evidence of Time-Varying Volatility)')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line for mean volatility
    mean_vol = rolling_std.mean()
    ax.axhline(mean_vol, color='blue', linestyle='--', lw=2, label=f'Mean: {mean_vol:.2f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_dir:
        fig3.savefig(f'{save_dir}/{region_name}_rolling_volatility.png', dpi=150, bbox_inches='tight')
    
    results['fig_rolling_vol'] = fig3
    
    print(f"Rolling standard deviation ranges from {rolling_std.min():.2f} to {rolling_std.max():.2f} MWh")
    print(f"Coefficient of variation: {rolling_std.std() / rolling_std.mean():.2%}")
    print("\nLarge variations in rolling std indicate volatility clustering\n")
    
    # ===========================
    # 6. Volatility by Day of Year
    # ===========================
    abs_residuals_series = pd.Series(abs_residuals, index=index)
    doy_volatility = abs_residuals_series.groupby(abs_residuals_series.index.day_of_year).mean()
    doy_volatility_std = abs_residuals_series.groupby(abs_residuals_series.index.day_of_year).std()
    
    fig4, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(doy_volatility.index, doy_volatility.values, lw=2, color='darkblue', label='Mean |Residual|')
    ax.fill_between(
        doy_volatility.index,
        doy_volatility.values - doy_volatility_std.values,
        doy_volatility.values + doy_volatility_std.values,
        alpha=0.3,
        color='blue'
    )
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Average |Residual| (MWh)')
    ax.set_title(f'{region_name} - Average Volatility by Day of Year')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 366)
    ax.legend()
    
    plt.tight_layout()
    
    if save_dir:
        fig4.savefig(f'{save_dir}/{region_name}_volatility_by_doy.png', dpi=150, bbox_inches='tight')
    
    results['fig_volatility_doy'] = fig4
    
    # Identify most volatile periods
    top_volatile_days = doy_volatility.nlargest(10)
    print(f"{region_name} - Most volatile days of year (top 10):")
    print(top_volatile_days)
    print(f"\nLeast volatile days of year (bottom 10):")
    print(doy_volatility.nsmallest(10))
    print("\n")
    
    results['volatility_by_doy'] = doy_volatility
    
    return results


def run_full_diagnostics(fit, region_name, save_dir=None):
    """
    Run all diagnostic analyses for a fitted model.
    
    Args:
        fit: fitted model dictionary
        region_name: name of the region
        save_dir: optional directory to save all figures
    
    Returns:
        dict with all analysis results and figures
    """
    print("\n" + "="*80)
    print(f"RUNNING FULL DIAGNOSTICS FOR {region_name}")
    print("="*80 + "\n")
    
    results = {}
    
    # 1. Fitted Model and Residuals
    print("1. Plotting fitted model and residuals...")
    save_path_1 = f'{save_dir}/{region_name}_model_fit_and_residuals.png' if save_dir else None
    results['fig_fitted'] = plot_fitted_model_and_residuals(fit, region_name, save_path=save_path_1)
    
    # 2. Residual Distribution
    print("2. Analyzing residual distribution...")
    save_path_2 = f'{save_dir}/{region_name}_residual_distribution.png' if save_dir else None
    results['fig_distribution'], results['normality_test'] = plot_residual_distribution(
        fit, region_name, save_path=save_path_2
    )
    
    # 3. Autocorrelation
    print("3. Plotting autocorrelation functions...")
    save_path_3 = f'{save_dir}/{region_name}_autocorrelation.png' if save_dir else None
    results['fig_acf'] = plot_autocorrelation(fit, region_name, lags=50, save_path=save_path_3)
    
    # 4. Volatility Clustering
    print("4. Analyzing volatility clustering...")
    results['volatility_analysis'] = analyze_volatility_clustering(fit, region_name, save_dir)
    
    print("\n" + "="*80)
    print(f"DIAGNOSTICS COMPLETE FOR {region_name}")
    print("="*80 + "\n")
    
    return results
