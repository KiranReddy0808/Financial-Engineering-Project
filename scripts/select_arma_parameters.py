#!/usr/bin/env python3import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add notebooks directory to path
sys.path.append('notebooks')
from seasonal_models import fit_seasonal_garch_model


def load_region_data(region_name):
    """Load processed data for a given region."""
    region_map = {
        'Boston': 'data/processed/Boston/Boston.csv',
        'New York': 'data/processed/NY/NewYork.csv',
        'Houston': 'data/processed/Houston/Houston.csv',
        'Chicago': 'data/processed/Chicago/Chicago.csv',
        'Dallas': 'data/processed/Dallas/Dallas.csv',
        'Minneapolis': 'data/processed/Minneapolis/Minneapolis.csv',
    }
    
    if region_name not in region_map:
        raise ValueError(f"Unknown region: {region_name}. Choose from {list(region_map.keys())}")
    
    file_path = Path(region_map[region_name])
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2022-12-31')]
    df.set_index('date', inplace=True)
    
    return df['avg_load']


def test_arma_parameters(srs, max_p=3, max_q=3, n_harmonics=3):

    results = []
    
    print(f"\nTesting ARMA combinations (p=0..{max_p}, q=0..{max_q})")
    print("=" * 80)
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                model_name = "Seasonal Only"
            else:
                model_name = f"ARMA({p},{q})"
            
            print(f"  Testing {model_name}...", end=" ")
            
            try:
                fit = fit_seasonal_garch_model(
                    srs,
                    n_harmonics=n_harmonics,
                    ar_order=p,
                    ma_order=q,
                    include_dayofweek=True
                )
                
                # Check if ARMA converged (if present)
                converged = True
                if fit['arma_params'] is not None:
                    converged = fit['arma_params'].get('converged', True)
                
                results.append({
                    'AR(p)': p,
                    'MA(q)': q,
                    'Model': model_name,
                    'N_Params': fit['n_total_params'],
                    'AIC': fit['aic'],
                    'BIC': fit['bic'],
                    'R²': fit['r_squared'],
                    'Residual_Std': fit['residual_std'],
                    'GARCH_α': fit['garch_params']['alpha'],
                    'GARCH_β': fit['garch_params']['beta'],
                    'Persistence': fit['persistence'],
                    'Converged': converged
                })
                
                status = "✓" if converged else "⚠ (did not converge)"
                print(f"{status} AIC={fit['aic']:.1f}, BIC={fit['bic']:.1f}")
                
            except Exception as e:
                print(f"✗ Failed: {str(e)[:50]}")
                results.append({
                    'AR(p)': p,
                    'MA(q)': q,
                    'Model': model_name,
                    'N_Params': np.nan,
                    'AIC': np.nan,
                    'BIC': np.nan,
                    'R²': np.nan,
                    'Residual_Std': np.nan,
                    'GARCH_α': np.nan,
                    'GARCH_β': np.nan,
                    'Persistence': np.nan,
                    'Converged': False
                })
    
    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze and print recommendations based on test results."""
    # Filter out non-converged models
    valid_results = results_df[results_df['Converged']].copy()
    
    if len(valid_results) == 0:
        print("\n⚠️  No models converged successfully!")
        return
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(valid_results.sort_values('AIC').to_string(index=False))
    
    # Find best models
    best_aic_idx = valid_results['AIC'].idxmin()
    best_bic_idx = valid_results['BIC'].idxmin()
    best_aic_model = valid_results.loc[best_aic_idx]
    best_bic_model = valid_results.loc[best_bic_idx]
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n🏆 BEST BY AIC (Predictive Performance):")
    print(f"   Model: {best_aic_model['Model']}")
    print(f"   Parameters: ar_order={int(best_aic_model['AR(p)'])}, ma_order={int(best_aic_model['MA(q)'])}")
    print(f"   AIC: {best_aic_model['AIC']:.2f}")
    print(f"   BIC: {best_aic_model['BIC']:.2f}")
    print(f"   R²: {best_aic_model['R²']:.4f}")
    print(f"   Code: fit_seasonal_garch_model(srs, n_harmonics=3, ar_order={int(best_aic_model['AR(p)'])}, ma_order={int(best_aic_model['MA(q)'])})")
    
    print("\n🏆 BEST BY BIC (Parsimonious Model):")
    print(f"   Model: {best_bic_model['Model']}")
    print(f"   Parameters: ar_order={int(best_bic_model['AR(p)'])}, ma_order={int(best_bic_model['MA(q)'])}")
    print(f"   AIC: {best_bic_model['AIC']:.2f}")
    print(f"   BIC: {best_bic_model['BIC']:.2f}")
    print(f"   R²: {best_bic_model['R²']:.4f}")
    print(f"   Code: fit_seasonal_garch_model(srs, n_harmonics=3, ar_order={int(best_bic_model['AR(p)'])}, ma_order={int(best_bic_model['MA(q)'])})")
    
    # Calculate improvements over baseline (0,0)
    baseline = valid_results[(valid_results['AR(p)'] == 0) & (valid_results['MA(q)'] == 0)]
    if len(baseline) > 0:
        baseline_aic = baseline['AIC'].values[0]
        baseline_bic = baseline['BIC'].values[0]
        aic_improvement = baseline_aic - best_aic_model['AIC']
        bic_improvement = baseline_bic - best_bic_model['BIC']
        
        print(f"\n📊 IMPROVEMENT OVER SEASONAL-ONLY MODEL:")
        print(f"   AIC improvement: {aic_improvement:.2f} points")
        print(f"   BIC improvement: {bic_improvement:.2f} points")
        
        if aic_improvement < 2:
            print("   ⚠️  Minimal improvement - seasonal model may be sufficient!")
        elif aic_improvement < 10:
            print("   ✓ Moderate improvement - ARMA terms are helpful")
        else:
            print("   ✓✓ Substantial improvement - ARMA terms are important")
    
    print("\n" + "=" * 80)


def plot_comparison(results_df, region_name, save_path=None):
    """Create visualization comparing different ARMA specifications."""
    valid_results = results_df[results_df['Converged']].copy()
    
    if len(valid_results) == 0:
        print("No valid results to plot.")
        return
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # AIC heatmap
    aic_pivot = valid_results.pivot(index='AR(p)', columns='MA(q)', values='AIC')
    sns.heatmap(aic_pivot, annot=True, fmt='.1f', cmap='YlOrRd_r', ax=axes[0, 0])
    axes[0, 0].set_title(f'{region_name} - AIC (Lower is Better)')
    axes[0, 0].set_xlabel('MA Order (q)')
    axes[0, 0].set_ylabel('AR Order (p)')
    
    # BIC heatmap
    bic_pivot = valid_results.pivot(index='AR(p)', columns='MA(q)', values='BIC')
    sns.heatmap(bic_pivot, annot=True, fmt='.1f', cmap='YlOrRd_r', ax=axes[0, 1])
    axes[0, 1].set_title(f'{region_name} - BIC (Lower is Better)')
    axes[0, 1].set_xlabel('MA Order (q)')
    axes[0, 1].set_ylabel('AR Order (p)')
    
    # R² heatmap
    r2_pivot = valid_results.pivot(index='AR(p)', columns='MA(q)', values='R²')
    sns.heatmap(r2_pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[1, 0])
    axes[1, 0].set_title(f'{region_name} - R² (Higher is Better)')
    axes[1, 0].set_xlabel('MA Order (q)')
    axes[1, 0].set_ylabel('AR Order (p)')
    
    # AIC vs BIC comparison
    axes[1, 1].scatter(valid_results['AIC'], valid_results['BIC'], s=100, alpha=0.6)
    for idx, row in valid_results.iterrows():
        axes[1, 1].annotate(row['Model'], (row['AIC'], row['BIC']), 
                          fontsize=8, ha='right', alpha=0.7)
    axes[1, 1].set_xlabel('AIC')
    axes[1, 1].set_ylabel('BIC')
    axes[1, 1].set_title('AIC vs BIC Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{region_name} - ARMA Parameter Comparison', fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
    
    plt.show()


def main():
    
    parser = argparse.ArgumentParser(
        description='Test and select optimal ARMA parameters for SARIMAX model'
    )
    parser.add_argument(
        '--region',
        type=str,
        required=True,
        choices=['Boston', 'New York', 'Houston', 'Chicago', 'Dallas', 'Minneapolis'],
        help='Region to analyze'
    )
    parser.add_argument(
        '--max-p',
        type=int,
        default=3,
        help='Maximum AR order to test (default: 3)'
    )
    parser.add_argument(
        '--max-q',
        type=int,
        default=3,
        help='Maximum MA order to test (default: 3)'
    )
    parser.add_argument(
        '--harmonics',
        type=int,
        default=3,
        help='Number of seasonal harmonics (default: 3)'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Save comparison plot to this path'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"ARMA PARAMETER SELECTION FOR {args.region.upper()}")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data for {args.region}...")
    srs = load_region_data(args.region)
    print(f"✓ Loaded {len(srs)} observations from {srs.index.min().date()} to {srs.index.max().date()}")
    
    # Test different parameters
    results_df = test_arma_parameters(
        srs,
        max_p=args.max_p,
        max_q=args.max_q,
        n_harmonics=args.harmonics
    )
    
    # Analyze and print recommendations
    analyze_results(results_df)
    
    # Create visualization
    save_path = args.save if args.save else f'data/images/arma_comparison_{args.region.replace(" ", "_")}.png'
    plot_comparison(results_df, args.region, save_path=save_path)
    
    # Save results to CSV
    csv_path = f'data/processed/{args.region.replace(" ", "_")}/arma_comparison.csv'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
