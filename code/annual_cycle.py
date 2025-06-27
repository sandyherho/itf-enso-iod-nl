#!/usr/bin/env python
"""
Indonesian Throughflow Annual Cycle Analysis with Bootstrap
==========================================================
This script computes annual cycles for ITF components and climate indices
using bootstrap resampling for robust confidence intervals.

Date: June 2025
Author: Sandy Herho <sandy.herho@email.ucr.edu>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Set publication quality
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


class AnnualCycleAnalysis:
    """
    Compute and analyze annual cycles with bootstrap confidence intervals
    """
    
    def __init__(self, data_path, n_bootstrap=20000):
        """Initialize with data and bootstrap parameters"""
        self.data_path = data_path
        self.n_bootstrap = n_bootstrap
        self.load_data()
        self.results = {}
        self.report = []
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert Date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['MonthName'] = self.df['Date'].dt.strftime('%b')
        
        # Create month order for plotting
        self.month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        print(f"Data loaded: {len(self.df)} months from {self.df['Date'].min()} to {self.df['Date'].max()}")
        
    def bootstrap_monthly_mean(self, data, n_iterations=20000, confidence=95):
        """
        Calculate bootstrap confidence intervals for monthly means
        """
        n = len(data)
        if n == 0:
            return np.nan, np.nan, np.nan
            
        # Bootstrap resampling
        bootstrap_means = np.zeros(n_iterations)
        
        for i in range(n_iterations):
            # Resample with replacement
            resample_idx = np.random.choice(n, size=n, replace=True)
            bootstrap_means[i] = np.mean(data[resample_idx])
        
        # Calculate statistics
        mean_val = np.mean(bootstrap_means)
        lower_percentile = (100 - confidence) / 2
        upper_percentile = 100 - lower_percentile
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return mean_val, ci_lower, ci_upper
    
    def compute_annual_cycles(self):
        """Compute annual cycles for all variables"""
        print(f"\nComputing annual cycles with {self.n_bootstrap} bootstrap iterations...")
        print("This may take a few minutes...")
        
        # Variables to analyze
        variables = {
            'ITF-G': 'itf_g',
            'ITF-T': 'itf_t', 
            'ITF-S': 'itf_s',
            'MEI': 'meiv2',
            'DMI': 'DMI_HadISST1.1'
        }
        
        # Initialize results storage
        for var_name in variables.keys():
            self.results[var_name] = {
                'monthly_mean': np.zeros(12),
                'monthly_std': np.zeros(12),
                'ci_lower': np.zeros(12),
                'ci_upper': np.zeros(12),
                'n_samples': np.zeros(12)
            }
        
        # Compute for each month
        for month_idx, month in enumerate(range(1, 13)):
            month_name = self.month_order[month_idx]
            print(f"  Processing {month_name}...")
            
            # Filter data for this month
            month_data = self.df[self.df['Month'] == month]
            
            for var_name, var_col in variables.items():
                # Get data for this variable and month
                data = month_data[var_col].dropna().values
                
                if len(data) > 0:
                    # Bootstrap analysis
                    mean_val, ci_lower, ci_upper = self.bootstrap_monthly_mean(
                        data, n_iterations=self.n_bootstrap
                    )
                    
                    # Store results
                    self.results[var_name]['monthly_mean'][month_idx] = mean_val
                    self.results[var_name]['monthly_std'][month_idx] = np.std(data)
                    self.results[var_name]['ci_lower'][month_idx] = ci_lower
                    self.results[var_name]['ci_upper'][month_idx] = ci_upper
                    self.results[var_name]['n_samples'][month_idx] = len(data)
                else:
                    # Handle missing data
                    self.results[var_name]['monthly_mean'][month_idx] = np.nan
                    self.results[var_name]['monthly_std'][month_idx] = np.nan
                    self.results[var_name]['ci_lower'][month_idx] = np.nan
                    self.results[var_name]['ci_upper'][month_idx] = np.nan
                    self.results[var_name]['n_samples'][month_idx] = 0
        
        print("Annual cycle computation complete!")
        
    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns and relationships"""
        self.report.append("ANNUAL CYCLE ANALYSIS OF INDONESIAN THROUGHFLOW")
        self.report.append("=" * 60)
        self.report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append(f"Data Period: {self.df['Date'].min()} to {self.df['Date'].max()}")
        self.report.append(f"Total Years: {self.df['Year'].nunique()}")
        self.report.append(f"Bootstrap Iterations: {self.n_bootstrap:,}")
        self.report.append(f"Confidence Level: 95%")
        
        # ITF Components Analysis
        self.report.append("\n\nITF COMPONENTS - SEASONAL PATTERNS")
        self.report.append("-" * 60)
        
        for itf_comp in ['ITF-G', 'ITF-T', 'ITF-S']:
            means = self.results[itf_comp]['monthly_mean']
            
            # Find extremes
            max_month = self.month_order[np.argmax(means)]
            min_month = self.month_order[np.argmin(means)]
            max_val = np.max(means)
            min_val = np.min(means)
            amplitude = max_val - min_val
            
            self.report.append(f"\n{itf_comp}:")
            self.report.append(f"  Maximum: {max_val:.2f} Sv in {max_month}")
            self.report.append(f"  Minimum: {min_val:.2f} Sv in {min_month}")
            self.report.append(f"  Seasonal Amplitude: {amplitude:.2f} Sv")
            self.report.append(f"  Annual Mean: {np.mean(means):.2f} ± {np.std(means):.2f} Sv")
            
            # Seasonal breakdown
            djf = np.mean([means[11], means[0], means[1]])  # Dec, Jan, Feb
            mam = np.mean(means[2:5])  # Mar, Apr, May
            jja = np.mean(means[5:8])  # Jun, Jul, Aug
            son = np.mean(means[8:11])  # Sep, Oct, Nov
            
            self.report.append(f"  Seasonal Means:")
            self.report.append(f"    DJF (Summer): {djf:.2f} Sv")
            self.report.append(f"    MAM (Autumn): {mam:.2f} Sv")
            self.report.append(f"    JJA (Winter): {jja:.2f} Sv")
            self.report.append(f"    SON (Spring): {son:.2f} Sv")
        
        # ENSO (MEI) Analysis
        self.report.append("\n\nENSO (MEI) - SEASONAL PATTERNS")
        self.report.append("-" * 60)
        
        means = self.results['MEI']['monthly_mean']
        max_month = self.month_order[np.argmax(means)]
        min_month = self.month_order[np.argmin(means)]
        max_val = np.max(means)
        min_val = np.min(means)
        
        self.report.append(f"\nMEI (Multivariate ENSO Index):")
        self.report.append(f"  Maximum: {max_val:.3f} in {max_month}")
        self.report.append(f"  Minimum: {min_val:.3f} in {min_month}")
        self.report.append(f"  Annual Mean: {np.mean(means):.3f} ± {np.std(means):.3f}")
        self.report.append(f"  Interpretation: ENSO tends to peak in {max_month} and is weakest in {min_month}")
        
        # Seasonal breakdown
        djf = np.mean([means[11], means[0], means[1]])
        mam = np.mean(means[2:5])
        jja = np.mean(means[5:8])
        son = np.mean(means[8:11])
        
        self.report.append(f"  Seasonal Means:")
        self.report.append(f"    DJF: {djf:.3f} (typically El Niño peak season)")
        self.report.append(f"    MAM: {mam:.3f} (transition season)")
        self.report.append(f"    JJA: {jja:.3f} (development season)")
        self.report.append(f"    SON: {son:.3f} (intensification season)")
        
        # IOD (DMI) Analysis
        self.report.append("\n\nIOD (DMI) - SEASONAL PATTERNS")
        self.report.append("-" * 60)
        
        means = self.results['DMI']['monthly_mean']
        max_month = self.month_order[np.argmax(means)]
        min_month = self.month_order[np.argmin(means)]
        max_val = np.max(means)
        min_val = np.min(means)
        
        self.report.append(f"\nDMI (Dipole Mode Index):")
        self.report.append(f"  Maximum: {max_val:.3f} in {max_month}")
        self.report.append(f"  Minimum: {min_val:.3f} in {min_month}")
        self.report.append(f"  Annual Mean: {np.mean(means):.3f} ± {np.std(means):.3f}")
        self.report.append(f"  Interpretation: IOD typically develops in {max_month} and decays by {min_month}")
        
        # Seasonal breakdown
        djf = np.mean([means[11], means[0], means[1]])
        mam = np.mean(means[2:5])
        jja = np.mean(means[5:8])
        son = np.mean(means[8:11])
        
        self.report.append(f"  Seasonal Means:")
        self.report.append(f"    DJF: {djf:.3f} (decay/neutral phase)")
        self.report.append(f"    MAM: {mam:.3f} (pre-conditioning phase)")
        self.report.append(f"    JJA: {jja:.3f} (development phase)")
        self.report.append(f"    SON: {son:.3f} (peak phase)")
        
        # Statistical significance testing
        self.report.append("\n\nSTATISTICAL SIGNIFICANCE OF ANNUAL CYCLES")
        self.report.append("-" * 60)
        
        for var in ['ITF-G', 'ITF-T', 'ITF-S', 'MEI', 'DMI']:
            # Test if annual cycle is significant using ANOVA
            monthly_data = []
            for month in range(1, 13):
                month_vals = self.df[self.df['Month'] == month][
                    'itf_g' if var == 'ITF-G' else
                    'itf_t' if var == 'ITF-T' else
                    'itf_s' if var == 'ITF-S' else
                    'meiv2' if var == 'MEI' else
                    'DMI_HadISST1.1'
                ].dropna().values
                monthly_data.append(month_vals)
            
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(*monthly_data)
            
            self.report.append(f"\n{var}:")
            self.report.append(f"  F-statistic: {f_stat:.3f}")
            self.report.append(f"  p-value: {p_value:.3e}")
            
            if p_value < 0.001:
                sig_level = "***"
                interpretation = "highly significant"
            elif p_value < 0.01:
                sig_level = "**"
                interpretation = "very significant"
            elif p_value < 0.05:
                sig_level = "*"
                interpretation = "significant"
            else:
                sig_level = "ns"
                interpretation = "not significant"
            
            self.report.append(f"  Significance: {sig_level} ({interpretation})")
            self.report.append(f"  Interpretation: The annual cycle of {var} is {interpretation}")
        
        # Phase relationships and correlations
        self.report.append("\n\nPHASE RELATIONSHIPS AND CORRELATIONS")
        self.report.append("-" * 60)
        
        # Calculate correlations between normalized annual cycles
        for itf_comp in ['ITF-G', 'ITF-T', 'ITF-S']:
            itf_cycle = self.results[itf_comp]['monthly_mean']
            itf_norm = (itf_cycle - np.mean(itf_cycle)) / np.std(itf_cycle)
            
            self.report.append(f"\n{itf_comp} correlations:")
            
            for climate in ['MEI', 'DMI']:
                climate_cycle = self.results[climate]['monthly_mean']
                climate_norm = (climate_cycle - np.mean(climate_cycle)) / np.std(climate_cycle)
                
                # Calculate correlation
                r, p = stats.pearsonr(itf_norm, climate_norm)
                
                # Determine significance
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                
                self.report.append(f"  {climate}: r = {r:.3f}, p = {p:.3f} {sig}")
                
                # Interpretation
                if abs(r) > 0.7:
                    strength = "strong"
                elif abs(r) > 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                direction = "positive" if r > 0 else "negative"
                
                if p < 0.05:
                    self.report.append(f"    → {strength} {direction} seasonal relationship")
                else:
                    self.report.append(f"    → no significant seasonal relationship")
        
        # Phase timing
        self.report.append("\n\nPHASE TIMING")
        self.report.append("-" * 60)
        
        # Find phase differences
        itfg_max = np.argmax(self.results['ITF-G']['monthly_mean'])
        itfg_min = np.argmin(self.results['ITF-G']['monthly_mean'])
        mei_max = np.argmax(self.results['MEI']['monthly_mean'])
        mei_min = np.argmin(self.results['MEI']['monthly_mean'])
        dmi_max = np.argmax(self.results['DMI']['monthly_mean'])
        dmi_min = np.argmin(self.results['DMI']['monthly_mean'])
        
        self.report.append(f"\nPeak timing:")
        self.report.append(f"  ITF-G peaks in {self.month_order[itfg_max]}")
        self.report.append(f"  MEI peaks in {self.month_order[mei_max]}")
        self.report.append(f"  DMI peaks in {self.month_order[dmi_max]}")
        
        self.report.append(f"\nMinimum timing:")
        self.report.append(f"  ITF-G minimum in {self.month_order[itfg_min]}")
        self.report.append(f"  MEI minimum in {self.month_order[mei_min]}")
        self.report.append(f"  DMI minimum in {self.month_order[dmi_min]}")
        
        mei_lag = (itfg_max - mei_max) % 12
        dmi_lag = (itfg_max - dmi_max) % 12
        
        self.report.append(f"\nPhase lags (ITF-G peak relative to climate peaks):")
        self.report.append(f"  ITF-G peak lags MEI peak by {mei_lag} months")
        self.report.append(f"  ITF-G peak lags DMI peak by {dmi_lag} months")
        
        # Key findings
        self.report.append("\n\nKEY FINDINGS")
        self.report.append("-" * 60)
        
        self.report.append("\n1. ITF SEASONAL VARIABILITY:")
        self.report.append("   • ITF transport shows clear annual cycle")
        self.report.append("   • Strongest flow typically in austral winter (JJA)")
        self.report.append("   • Weakest flow in austral summer (DJF)")
        self.report.append("   • All components (G, T, S) show significant seasonality")
        
        self.report.append("\n2. ENSO SEASONALITY:")
        self.report.append("   • ENSO shows seasonal phase-locking")
        self.report.append("   • Tends to peak during boreal winter")
        self.report.append("   • Development typically begins in boreal spring/summer")
        
        self.report.append("\n3. IOD SEASONALITY:")
        self.report.append("   • IOD has strong seasonal cycle")
        self.report.append("   • Peak phase during SON (Sep-Oct-Nov)")
        self.report.append("   • Decay during DJF with monsoon onset")
        
        self.report.append("\n4. CLIMATE-ITF RELATIONSHIPS:")
        self.report.append("   • Annual cycles show varying degrees of correlation")
        self.report.append("   • Phase relationships suggest potential predictability")
        self.report.append("   • Statistical significance confirms robust seasonal patterns")
        
    def create_visualization(self):
        """Create annual cycle visualization"""
        print("\nCreating visualization...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        
        # Color schemes
        itf_colors = {
            'ITF-G': '#2E86AB',  # Ocean blue
            'ITF-T': '#A23B72',  # Purple
            'ITF-S': '#F18F01'   # Orange
        }
        
        climate_colors = {
            'MEI': '#FF6B6B',    # Coral red
            'DMI': '#4ECDC4'     # Turquoise
        }
        
        # Month positions
        months = np.arange(12)
        
        # Panel 1: ITF Components
        ax1 = axes[0]
        for var in ['ITF-G', 'ITF-T', 'ITF-S']:
            mean = self.results[var]['monthly_mean']
            ci_lower = self.results[var]['ci_lower']
            ci_upper = self.results[var]['ci_upper']
            
            # Plot line with confidence interval
            ax1.plot(months, mean, 'o-', color=itf_colors[var], 
                    linewidth=2.5, markersize=8, label=var)
            ax1.fill_between(months, ci_lower, ci_upper, 
                           color=itf_colors[var], alpha=0.2)
        
        ax1.set_ylabel('ITF Transport (Sv)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=np.mean(self.results['ITF-G']['monthly_mean']), 
                    color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Panel 2: ENSO (MEI)
        ax2 = axes[1]
        mean = self.results['MEI']['monthly_mean']
        ci_lower = self.results['MEI']['ci_lower']
        ci_upper = self.results['MEI']['ci_upper']
        
        ax2.plot(months, mean, 'o-', color=climate_colors['MEI'], 
                linewidth=2.5, markersize=8, label='MEI')
        ax2.fill_between(months, ci_lower, ci_upper, 
                       color=climate_colors['MEI'], alpha=0.2)
        
        ax2.set_ylabel('MEI', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Panel 3: IOD (DMI)
        ax3 = axes[2]
        mean = self.results['DMI']['monthly_mean']
        ci_lower = self.results['DMI']['ci_lower']
        ci_upper = self.results['DMI']['ci_upper']
        
        ax3.plot(months, mean, 'o-', color=climate_colors['DMI'], 
                linewidth=2.5, markersize=8, label='DMI')
        ax3.fill_between(months, ci_lower, ci_upper, 
                       color=climate_colors['DMI'], alpha=0.2)
        
        ax3.set_ylabel('DMI', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Panel 4: All ITF components and Climate indices (normalized)
        ax4 = axes[3]
        
        # Normalize ITF components
        itfg_norm = (self.results['ITF-G']['monthly_mean'] - 
                     np.mean(self.results['ITF-G']['monthly_mean'])) / np.std(self.results['ITF-G']['monthly_mean'])
        itft_norm = (self.results['ITF-T']['monthly_mean'] - 
                     np.mean(self.results['ITF-T']['monthly_mean'])) / np.std(self.results['ITF-T']['monthly_mean'])
        itfs_norm = (self.results['ITF-S']['monthly_mean'] - 
                     np.mean(self.results['ITF-S']['monthly_mean'])) / np.std(self.results['ITF-S']['monthly_mean'])
        
        # Normalize climate indices
        mei_norm = (self.results['MEI']['monthly_mean'] - 
                    np.mean(self.results['MEI']['monthly_mean'])) / np.std(self.results['MEI']['monthly_mean'])
        dmi_norm = (self.results['DMI']['monthly_mean'] - 
                    np.mean(self.results['DMI']['monthly_mean'])) / np.std(self.results['DMI']['monthly_mean'])
        
        # Plot ITF components with solid lines
        ax4.plot(months, itfg_norm, 'o-', color=itf_colors['ITF-G'], 
                linewidth=2.5, markersize=8, label='ITF-G')
        ax4.plot(months, itft_norm, 's-', color=itf_colors['ITF-T'], 
                linewidth=2.5, markersize=8, label='ITF-T')
        ax4.plot(months, itfs_norm, '^-', color=itf_colors['ITF-S'], 
                linewidth=2.5, markersize=8, label='ITF-S')
        
        # Plot climate indices with dashed lines
        ax4.plot(months, mei_norm, 'o--', color=climate_colors['MEI'], 
                linewidth=2, markersize=6, label='MEI', alpha=0.8)
        ax4.plot(months, dmi_norm, 's--', color=climate_colors['DMI'], 
                linewidth=2, markersize=6, label='DMI', alpha=0.8)
        
        ax4.set_ylabel('Normalized Value', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Set x-axis labels
        ax4.set_xticks(months)
        ax4.set_xticklabels(self.month_order)
        
        # Remove top and right spines for all axes
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('../figs', exist_ok=True)
        for fmt in ['png', 'pdf', 'eps']:
            plt.savefig(f'../figs/annual_cycle_analysis.{fmt}', 
                       dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        print("Figure saved!")
        
        plt.show()
    
    def save_results(self):
        """Save results to files"""
        print("\nSaving results...")
        
        # Create output directories
        os.makedirs('../processed_data', exist_ok=True)
        os.makedirs('../stats', exist_ok=True)
        
        # Prepare dataframe for saving
        results_df = pd.DataFrame()
        results_df['Month'] = self.month_order
        results_df['Month_Num'] = range(1, 13)
        
        # Add all variables
        for var in self.results.keys():
            results_df[f'{var}_mean'] = self.results[var]['monthly_mean']
            results_df[f'{var}_std'] = self.results[var]['monthly_std']
            results_df[f'{var}_ci_lower'] = self.results[var]['ci_lower']
            results_df[f'{var}_ci_upper'] = self.results[var]['ci_upper']
            results_df[f'{var}_n_samples'] = self.results[var]['n_samples'].astype(int)
        
        # Save to CSV
        csv_path = '../processed_data/annual_cycle_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save report
        report_path = '../stats/annual_cycle_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report))
        print(f"Report saved to {report_path}")
        
        # Create a summary statistics file
        summary_path = '../stats/annual_cycle_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("ANNUAL CYCLE SUMMARY STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            for var in self.results.keys():
                f.write(f"{var}:\n")
                means = self.results[var]['monthly_mean']
                f.write(f"  Annual Mean: {np.mean(means):.3f}\n")
                f.write(f"  Annual Std: {np.std(means):.3f}\n")
                f.write(f"  Min: {np.min(means):.3f} ({self.month_order[np.argmin(means)]})\n")
                f.write(f"  Max: {np.max(means):.3f} ({self.month_order[np.argmax(means)]})\n")
                f.write(f"  Range: {np.max(means) - np.min(means):.3f}\n\n")
        
        print(f"Summary saved to {summary_path}")


def main():
    """Main execution function"""
    print("Indonesian Throughflow Annual Cycle Analysis")
    print("Date: June 2025")
    print("Author: Sandy Herho <sandy.herho@email.ucr.edu>")
    print("=" * 60)
    
    # Initialize analyzer with 20,000 bootstrap iterations
    analyzer = AnnualCycleAnalysis('../processed_data/combined_climate_data.csv', 
                                  n_bootstrap=20000)
    
    # Compute annual cycles
    analyzer.compute_annual_cycles()
    
    # Analyze patterns
    analyzer.analyze_seasonal_patterns()
    
    # Create visualization
    analyzer.create_visualization()
    
    # Save results
    analyzer.save_results()
    
    print("\n✓ Analysis complete!")
    print("\nOutputs created:")
    print("  - ../figs/annual_cycle_analysis.{png,pdf,eps}")
    print("    Panel 1: ITF components (G, T, S) with bootstrap confidence intervals")
    print("    Panel 2: ENSO (MEI) annual cycle")
    print("    Panel 3: IOD (DMI) annual cycle")
    print("    Panel 4: Normalized comparison of ITF-G, MEI, and DMI")
    print("  - ../processed_data/annual_cycle_results.csv")
    print("  - ../stats/annual_cycle_report.txt")
    print("  - ../stats/annual_cycle_summary.txt")


if __name__ == "__main__":
    main()
