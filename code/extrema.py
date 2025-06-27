#!/usr/bin/env python
"""
Extrema Evaluation for Climate Time Series Data
Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: June 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import genextreme, gumbel_r
import warnings
warnings.filterwarnings('ignore')

class ExtremaAnalyzer:
    """Comprehensive extrema analysis for climate time series"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.results = {}
        self.composite_scores = {}
        
    def statistical_threshold_extrema(self, series, percentiles=[5, 95]):
        """Percentile-based threshold method"""
        lower_threshold = np.percentile(series, percentiles[0])
        upper_threshold = np.percentile(series, percentiles[1])
        
        extrema_low = series < lower_threshold
        extrema_high = series > upper_threshold
        
        return {
            'lower_extrema': extrema_low,
            'upper_extrema': extrema_high,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold
        }
    
    def block_maxima_analysis(self, series, block_size=30):
        """Block maxima approach for extreme value analysis"""
        n_blocks = len(series) // block_size
        block_maxima = []
        block_minima = []
        block_indices_max = []
        block_indices_min = []
        
        for i in range(n_blocks):
            block = series[i*block_size:(i+1)*block_size]
            if len(block) > 0:
                max_idx = i*block_size + np.argmax(block.values)
                min_idx = i*block_size + np.argmin(block.values)
                block_maxima.append(block.max())
                block_minima.append(block.min())
                block_indices_max.append(max_idx)
                block_indices_min.append(min_idx)
        
        # Fit GEV distribution
        maxima_params = genextreme.fit(block_maxima)
        minima_params = genextreme.fit(-np.array(block_minima))
        
        return {
            'block_maxima': block_maxima,
            'block_minima': block_minima,
            'maxima_indices': block_indices_max,
            'minima_indices': block_indices_min,
            'maxima_gev_params': maxima_params,
            'minima_gev_params': minima_params
        }
    
    def peak_over_threshold(self, series, threshold_percentile=90):
        """Peak-over-threshold method"""
        threshold_high = np.percentile(series, threshold_percentile)
        threshold_low = np.percentile(series, 100 - threshold_percentile)
        
        exceedances_high = series[series > threshold_high] - threshold_high
        exceedances_low = threshold_low - series[series < threshold_low]
        
        indices_high = np.where(series > threshold_high)[0]
        indices_low = np.where(series < threshold_low)[0]
        
        return {
            'high_exceedances': exceedances_high,
            'low_exceedances': exceedances_low,
            'high_indices': indices_high,
            'low_indices': indices_low,
            'threshold_high': threshold_high,
            'threshold_low': threshold_low
        }
    
    def moving_window_extrema(self, series, window_size=30):
        """Moving window extrema detection"""
        extrema_max = np.zeros(len(series), dtype=bool)
        extrema_min = np.zeros(len(series), dtype=bool)
        
        for i in range(window_size//2, len(series) - window_size//2):
            window = series[i-window_size//2:i+window_size//2+1]
            if series.iloc[i] == window.max():
                extrema_max[i] = True
            if series.iloc[i] == window.min():
                extrema_min[i] = True
        
        return {
            'local_maxima': extrema_max,
            'local_minima': extrema_min
        }
    
    def zscore_extrema(self, series, threshold=2):
        """Z-score based extrema detection"""
        z_scores = np.abs(stats.zscore(series))
        extrema = z_scores > threshold
        
        return {
            'extrema': extrema,
            'z_scores': z_scores,
            'threshold': threshold
        }
    
    def modified_zscore_extrema(self, series, threshold=3.5):
        """Modified Z-score using Median Absolute Deviation"""
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        
        extrema_high = modified_z_scores > threshold
        extrema_low = modified_z_scores < -threshold
        
        return {
            'extrema_high': extrema_high,
            'extrema_low': extrema_low,
            'modified_z_scores': modified_z_scores,
            'threshold': threshold
        }
    
    def isolation_forest_extrema(self, series, contamination=0.1):
        """Isolation Forest for anomaly detection"""
        series_reshaped = series.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(series_reshaped)
        
        extrema = predictions == -1
        scores = iso_forest.score_samples(series_reshaped)
        
        return {
            'extrema': extrema,
            'anomaly_scores': scores
        }
    
    def local_outlier_factor_extrema(self, series, n_neighbors=20, contamination=0.1):
        """Local Outlier Factor for extrema detection"""
        series_reshaped = series.values.reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        predictions = lof.fit_predict(series_reshaped)
        
        extrema = predictions == -1
        scores = lof.negative_outlier_factor_
        
        return {
            'extrema': extrema,
            'outlier_scores': scores
        }
    
    def compute_composite_score(self, var_name, series):
        """Compute composite extrema score from multiple methods"""
        methods_results = {}
        
        # Apply all methods
        methods_results['statistical'] = self.statistical_threshold_extrema(series)
        methods_results['block_maxima'] = self.block_maxima_analysis(series)
        methods_results['pot'] = self.peak_over_threshold(series)
        methods_results['moving_window'] = self.moving_window_extrema(series)
        methods_results['zscore'] = self.zscore_extrema(series)
        methods_results['modified_zscore'] = self.modified_zscore_extrema(series)
        methods_results['isolation_forest'] = self.isolation_forest_extrema(series)
        methods_results['lof'] = self.local_outlier_factor_extrema(series)
        
        # Create composite score
        n = len(series)
        composite_high = np.zeros(n)
        composite_low = np.zeros(n)
        
        # Aggregate scores from different methods
        weights = {
            'statistical': 1.0,
            'zscore': 1.0,
            'modified_zscore': 1.5,
            'isolation_forest': 1.2,
            'lof': 1.2,
            'moving_window': 0.8
        }
        
        # Statistical threshold
        composite_high += weights['statistical'] * methods_results['statistical']['upper_extrema']
        composite_low += weights['statistical'] * methods_results['statistical']['lower_extrema']
        
        # Z-score methods
        composite_high += weights['zscore'] * methods_results['zscore']['extrema']
        composite_high += weights['modified_zscore'] * methods_results['modified_zscore']['extrema_high']
        composite_low += weights['modified_zscore'] * methods_results['modified_zscore']['extrema_low']
        
        # ML methods
        composite_high += weights['isolation_forest'] * methods_results['isolation_forest']['extrema']
        composite_low += weights['isolation_forest'] * methods_results['isolation_forest']['extrema']
        composite_high += weights['lof'] * methods_results['lof']['extrema']
        composite_low += weights['lof'] * methods_results['lof']['extrema']
        
        # Moving window
        composite_high += weights['moving_window'] * methods_results['moving_window']['local_maxima']
        composite_low += weights['moving_window'] * methods_results['moving_window']['local_minima']
        
        # Normalize composite scores
        total_weight = sum(weights.values())
        composite_high /= total_weight
        composite_low /= total_weight
        
        return {
            'composite_high': composite_high,
            'composite_low': composite_low,
            'methods_results': methods_results
        }
    
    def analyze_variable(self, var_name, columns):
        """Analyze extrema for a specific variable"""
        if isinstance(columns, list):
            # For ITF, compute mean of all three components (itf_g, itf_t, itf_s)
            series = self.data[columns].mean(axis=1)
        else:
            series = self.data[columns]
        
        # Remove NaN values
        valid_mask = ~series.isna()
        series_clean = series[valid_mask]
        
        # Compute composite scores and individual methods
        analysis_results = self.compute_composite_score(var_name, series_clean)
        
        # Store results
        self.results[var_name] = analysis_results
        self.composite_scores[var_name] = {
            'composite_high': analysis_results['composite_high'],
            'composite_low': analysis_results['composite_low'],
            'series': series_clean,
            'valid_mask': valid_mask
        }
        
        return analysis_results
    
    def save_extrema_events(self, output_path):
        """Save all extrema events to a detailed text file"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPLETE EXTREMA EVENTS LISTING\n")
            f.write("Author: Sandy Herho <sandy.herho@email.ucr.edu>\n")
            f.write("Date: June 2025\n")
            f.write("="*80 + "\n\n")
            
            for var_name in ['ITF', 'ENSO', 'DMI']:
                f.write(f"\n{var_name} EXTREMA EVENTS\n")
                f.write("="*60 + "\n")
                
                # Get composite scores and thresholds
                composite_high = self.composite_scores[var_name]['composite_high']
                composite_low = self.composite_scores[var_name]['composite_low']
                series = self.composite_scores[var_name]['series']
                valid_mask = self.composite_scores[var_name]['valid_mask']
                
                high_threshold = np.percentile(composite_high, 90)
                low_threshold = np.percentile(composite_low, 90)
                
                # Get indices
                high_extrema_indices = np.where(composite_high > high_threshold)[0]
                low_extrema_indices = np.where(composite_low > low_threshold)[0]
                
                # Get corresponding dates
                valid_dates = self.data['Date'][valid_mask].reset_index(drop=True)
                valid_times = self.data['time'][valid_mask].reset_index(drop=True)
                
                # Write high extrema
                f.write(f"\nHIGH EXTREMA EVENTS (Total: {len(high_extrema_indices)})\n")
                f.write("-"*60 + "\n")
                f.write("Date                Time      Value       Score\n")
                f.write("-"*60 + "\n")
                
                for idx in high_extrema_indices:
                    date_str = valid_dates.iloc[idx]
                    time_val = valid_times.iloc[idx]
                    value = series.iloc[idx]
                    score = composite_high[idx]
                    f.write(f"{date_str:20s} {time_val:8.2f} {value:10.4f} {score:8.3f}\n")
                
                # Write low extrema
                f.write(f"\nLOW EXTREMA EVENTS (Total: {len(low_extrema_indices)})\n")
                f.write("-"*60 + "\n")
                f.write("Date                Time      Value       Score\n")
                f.write("-"*60 + "\n")
                
                for idx in low_extrema_indices:
                    date_str = valid_dates.iloc[idx]
                    time_val = valid_times.iloc[idx]
                    value = series.iloc[idx]
                    score = composite_low[idx]
                    f.write(f"{date_str:20s} {time_val:8.2f} {value:10.4f} {score:8.3f}\n")
                
                # Add summary statistics
                f.write(f"\nSUMMARY STATISTICS FOR {var_name}\n")
                f.write("-"*60 + "\n")
                f.write(f"High extrema threshold (90th percentile): {high_threshold:.3f}\n")
                f.write(f"Low extrema threshold (90th percentile): {low_threshold:.3f}\n")
                f.write(f"Total high extrema events: {len(high_extrema_indices)}\n")
                f.write(f"Total low extrema events: {len(low_extrema_indices)}\n")
                f.write(f"Percentage of high extrema: {len(high_extrema_indices)/len(series)*100:.2f}%\n")
                f.write(f"Percentage of low extrema: {len(low_extrema_indices)/len(series)*100:.2f}%\n")
    
    def generate_statistics_report(self):
        """Generate comprehensive statistics report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("EXTREMA EVALUATION REPORT FOR CLIMATE TIME SERIES")
        report_lines.append("Author: Sandy Herho <sandy.herho@email.ucr.edu>")
        report_lines.append("Date: June 2025")
        report_lines.append("="*80)
        report_lines.append("")
        
        for var_name, results in self.results.items():
            report_lines.append(f"\n{var_name} ANALYSIS")
            report_lines.append("-"*40)
            
            # Basic statistics
            series = self.composite_scores[var_name]['series']
            valid_mask = self.composite_scores[var_name]['valid_mask']
            report_lines.append(f"Total observations: {len(series)}")
            report_lines.append(f"Mean: {series.mean():.4f}")
            report_lines.append(f"Standard deviation: {series.std():.4f}")
            report_lines.append(f"Skewness: {stats.skew(series):.4f}")
            report_lines.append(f"Kurtosis: {stats.kurtosis(series):.4f}")
            
            # Method-specific results
            methods = results['methods_results']
            
            # Statistical threshold
            stat_results = methods['statistical']
            report_lines.append(f"\nStatistical Threshold Method:")
            report_lines.append(f"  Lower threshold (5%): {stat_results['lower_threshold']:.4f}")
            report_lines.append(f"  Upper threshold (95%): {stat_results['upper_threshold']:.4f}")
            report_lines.append(f"  Lower extrema count: {stat_results['lower_extrema'].sum()}")
            report_lines.append(f"  Upper extrema count: {stat_results['upper_extrema'].sum()}")
            
            # Block maxima
            block_results = methods['block_maxima']
            report_lines.append(f"\nBlock Maxima Analysis:")
            report_lines.append(f"  Number of blocks: {len(block_results['block_maxima'])}")
            report_lines.append(f"  GEV shape parameter (maxima): {block_results['maxima_gev_params'][0]:.4f}")
            report_lines.append(f"  GEV location parameter (maxima): {block_results['maxima_gev_params'][1]:.4f}")
            report_lines.append(f"  GEV scale parameter (maxima): {block_results['maxima_gev_params'][2]:.4f}")
            
            # POT
            pot_results = methods['pot']
            report_lines.append(f"\nPeak-Over-Threshold Analysis:")
            report_lines.append(f"  High threshold: {pot_results['threshold_high']:.4f}")
            report_lines.append(f"  Low threshold: {pot_results['threshold_low']:.4f}")
            report_lines.append(f"  High exceedances: {len(pot_results['high_exceedances'])}")
            report_lines.append(f"  Low exceedances: {len(pot_results['low_exceedances'])}")
            
            # Z-score methods
            zscore_results = methods['zscore']
            report_lines.append(f"\nZ-score Method:")
            report_lines.append(f"  Extrema detected: {zscore_results['extrema'].sum()}")
            report_lines.append(f"  Max Z-score: {zscore_results['z_scores'].max():.4f}")
            
            # Modified Z-score
            mod_zscore_results = methods['modified_zscore']
            report_lines.append(f"\nModified Z-score (MAD) Method:")
            report_lines.append(f"  High extrema: {mod_zscore_results['extrema_high'].sum()}")
            report_lines.append(f"  Low extrema: {mod_zscore_results['extrema_low'].sum()}")
            
            # ML methods
            iso_results = methods['isolation_forest']
            report_lines.append(f"\nIsolation Forest:")
            report_lines.append(f"  Anomalies detected: {iso_results['extrema'].sum()}")
            
            lof_results = methods['lof']
            report_lines.append(f"\nLocal Outlier Factor:")
            report_lines.append(f"  Outliers detected: {lof_results['extrema'].sum()}")
            
            # Composite scores
            composite_high = results['composite_high']
            composite_low = results['composite_low']
            high_threshold = np.percentile(composite_high, 90)
            low_threshold = np.percentile(composite_low, 90)
            
            report_lines.append(f"\nComposite Score Analysis:")
            report_lines.append(f"  High extrema (score > {high_threshold:.3f}): {(composite_high > high_threshold).sum()}")
            report_lines.append(f"  Low extrema (score > {low_threshold:.3f}): {(composite_low > low_threshold).sum()}")
            report_lines.append(f"  Max composite score (high): {composite_high.max():.4f}")
            report_lines.append(f"  Max composite score (low): {composite_low.max():.4f}")
            
            # EXTREMA OCCURRENCE DATES
            report_lines.append(f"\n{var_name} EXTREMA OCCURRENCE DATES")
            report_lines.append("-"*40)
            
            # Get dates for high extrema
            high_extrema_indices = np.where(composite_high > high_threshold)[0]
            low_extrema_indices = np.where(composite_low > low_threshold)[0]
            
            # Get corresponding dates
            valid_dates = self.data['Date'][valid_mask].reset_index(drop=True)
            valid_times = self.data['time'][valid_mask].reset_index(drop=True)
            
            report_lines.append("\nHIGH EXTREMA EVENTS:")
            if len(high_extrema_indices) > 0:
                for idx in high_extrema_indices[:20]:  # Show first 20
                    date_str = valid_dates.iloc[idx]
                    time_val = valid_times.iloc[idx]
                    value = series.iloc[idx]
                    score = composite_high[idx]
                    report_lines.append(f"  {date_str} (time={time_val:.2f}): value={value:.4f}, score={score:.3f}")
                if len(high_extrema_indices) > 20:
                    report_lines.append(f"  ... and {len(high_extrema_indices)-20} more events")
                    report_lines.append(f"  (See extrema_all_events.txt for complete listing)")
            else:
                report_lines.append("  No high extrema detected")
            
            report_lines.append("\nLOW EXTREMA EVENTS:")
            if len(low_extrema_indices) > 0:
                for idx in low_extrema_indices[:20]:  # Show first 20
                    date_str = valid_dates.iloc[idx]
                    time_val = valid_times.iloc[idx]
                    value = series.iloc[idx]
                    score = composite_low[idx]
                    report_lines.append(f"  {date_str} (time={time_val:.2f}): value={value:.4f}, score={score:.3f}")
                if len(low_extrema_indices) > 20:
                    report_lines.append(f"  ... and {len(low_extrema_indices)-20} more events")
                    report_lines.append(f"  (See extrema_all_events.txt for complete listing)")
            else:
                report_lines.append("  No low extrema detected")
            
        # Cross-variable analysis
        report_lines.append("\n\nCROSS-VARIABLE EXTREMA ANALYSIS")
        report_lines.append("="*40)
        
        # Find coincident extrema
        itf_high = self.composite_scores['ITF']['composite_high'] > np.percentile(self.composite_scores['ITF']['composite_high'], 90)
        enso_high = self.composite_scores['ENSO']['composite_high'] > np.percentile(self.composite_scores['ENSO']['composite_high'], 90)
        dmi_high = self.composite_scores['DMI']['composite_high'] > np.percentile(self.composite_scores['DMI']['composite_high'], 90)
        
        # Align indices
        min_len = min(len(itf_high), len(enso_high), len(dmi_high))
        itf_high = itf_high[:min_len]
        enso_high = enso_high[:min_len]
        dmi_high = dmi_high[:min_len]
        
        coincident_all = itf_high & enso_high & dmi_high
        coincident_itf_enso = itf_high & enso_high
        coincident_itf_dmi = itf_high & dmi_high
        coincident_enso_dmi = enso_high & dmi_high
        
        report_lines.append(f"Coincident high extrema (all three): {coincident_all.sum()}")
        report_lines.append(f"Coincident high extrema (ITF-ENSO): {coincident_itf_enso.sum()}")
        report_lines.append(f"Coincident high extrema (ITF-DMI): {coincident_itf_dmi.sum()}")
        report_lines.append(f"Coincident high extrema (ENSO-DMI): {coincident_enso_dmi.sum()}")
        
        return "\n".join(report_lines)
    
    def save_results_csv(self, output_path):
        """Save detailed results to CSV"""
        results_df = pd.DataFrame()
        
        # Add time information
        results_df['time'] = self.data['time']
        results_df['Date'] = self.data['Date']
        
        # Add original data
        results_df['ITF_mean'] = self.data[['itf_g', 'itf_t', 'itf_s']].mean(axis=1)
        results_df['ENSO'] = self.data['meiv2']
        results_df['DMI'] = self.data['DMI_HadISST1.1']
        
        # Add composite scores
        for var_name in ['ITF', 'ENSO', 'DMI']:
            valid_mask = self.composite_scores[var_name]['valid_mask']
            
            # Initialize full-length arrays
            composite_high_full = np.full(len(self.data), np.nan)
            composite_low_full = np.full(len(self.data), np.nan)
            
            # Fill in valid values
            composite_high_full[valid_mask] = self.composite_scores[var_name]['composite_high']
            composite_low_full[valid_mask] = self.composite_scores[var_name]['composite_low']
            
            results_df[f'{var_name}_composite_high'] = composite_high_full
            results_df[f'{var_name}_composite_low'] = composite_low_full
            
            # Add individual method results
            methods = self.results[var_name]['methods_results']
            
            # Statistical threshold
            stat_upper = np.full(len(self.data), False)
            stat_lower = np.full(len(self.data), False)
            stat_upper[valid_mask] = methods['statistical']['upper_extrema']
            stat_lower[valid_mask] = methods['statistical']['lower_extrema']
            results_df[f'{var_name}_statistical_upper'] = stat_upper
            results_df[f'{var_name}_statistical_lower'] = stat_lower
            
            # Z-score
            zscore_extrema = np.full(len(self.data), False)
            zscore_extrema[valid_mask] = methods['zscore']['extrema']
            results_df[f'{var_name}_zscore_extrema'] = zscore_extrema
            
            # Modified Z-score
            mod_zscore_high = np.full(len(self.data), False)
            mod_zscore_low = np.full(len(self.data), False)
            mod_zscore_high[valid_mask] = methods['modified_zscore']['extrema_high']
            mod_zscore_low[valid_mask] = methods['modified_zscore']['extrema_low']
            results_df[f'{var_name}_modified_zscore_high'] = mod_zscore_high
            results_df[f'{var_name}_modified_zscore_low'] = mod_zscore_low
            
            # Isolation Forest
            iso_extrema = np.full(len(self.data), False)
            iso_extrema[valid_mask] = methods['isolation_forest']['extrema']
            results_df[f'{var_name}_isolation_forest'] = iso_extrema
            
            # LOF
            lof_extrema = np.full(len(self.data), False)
            lof_extrema[valid_mask] = methods['lof']['extrema']
            results_df[f'{var_name}_lof'] = lof_extrema
        
        results_df.to_csv(output_path, index=False)
    
    def create_visualization(self, output_prefix):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        
        variables = {
            'ITF': self.data[['itf_g', 'itf_t', 'itf_s']].mean(axis=1),
            'ENSO': self.data['meiv2'],
            'DMI': self.data['DMI_HadISST1.1']
        }
        
        colors = {'ITF': '#2E86AB', 'ENSO': '#E63946', 'DMI': '#06D6A0'}
        
        # Plot each variable with extrema
        for col, (var_name, series) in enumerate(variables.items()):
            valid_mask = self.composite_scores[var_name]['valid_mask']
            time_valid = self.data['time'][valid_mask]
            series_valid = series[valid_mask]
            
            # Time series with composite extrema
            ax = axes[0, col]
            ax.plot(time_valid, series_valid, color=colors[var_name], alpha=0.7, linewidth=1)
            
            # Mark extrema
            composite_high = self.composite_scores[var_name]['composite_high']
            composite_low = self.composite_scores[var_name]['composite_low']
            high_threshold = np.percentile(composite_high, 90)
            low_threshold = np.percentile(composite_low, 90)
            
            high_extrema = composite_high > high_threshold
            low_extrema = composite_low > low_threshold
            
            ax.scatter(time_valid[high_extrema], series_valid[high_extrema], 
                      color='red', s=50, marker='^', alpha=0.8, zorder=5)
            ax.scatter(time_valid[low_extrema], series_valid[low_extrema], 
                      color='blue', s=50, marker='v', alpha=0.8, zorder=5)
            
            ax.set_xlim(time_valid.min(), time_valid.max())
            ax.grid(True, alpha=0.3)
            
            # Set appropriate y-axis labels
            if var_name == 'ITF':
                ax.set_ylabel('ITF Transport (Sv)', fontsize=12)
            elif var_name == 'ENSO':
                ax.set_ylabel('MEI', fontsize=12)
            elif var_name == 'DMI':
                ax.set_ylabel('DMI', fontsize=12)
            
            # Composite score time series
            ax = axes[1, col]
            ax.fill_between(time_valid, 0, composite_high, color='red', alpha=0.3)
            ax.fill_between(time_valid, 0, -composite_low, color='blue', alpha=0.3)
            ax.axhline(y=high_threshold, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=-low_threshold, color='blue', linestyle='--', alpha=0.5)
            ax.set_xlim(time_valid.min(), time_valid.max())
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Composite Score', fontsize=12)
            
            # Method comparison heatmap
            ax = axes[2, col]
            methods_names = ['Statistical', 'Z-score', 'Mod Z-score', 'Iso Forest', 'LOF', 'Moving Win']
            methods_data = []
            
            methods_results = self.results[var_name]['methods_results']
            methods_data.append(methods_results['statistical']['upper_extrema'].sum())
            methods_data.append(methods_results['zscore']['extrema'].sum())
            methods_data.append(methods_results['modified_zscore']['extrema_high'].sum())
            methods_data.append(methods_results['isolation_forest']['extrema'].sum())
            methods_data.append(methods_results['lof']['extrema'].sum())
            methods_data.append(methods_results['moving_window']['local_maxima'].sum())
            
            y_pos = np.arange(len(methods_names))
            bars = ax.barh(y_pos, methods_data, color=colors[var_name], alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(methods_names, fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_xlabel('Number of Extrema', fontsize=12)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, methods_data)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'{int(value)}', va='center', fontsize=9)
            
            # Extrema distribution
            ax = axes[3, col]
            hist_vals, bins, _ = ax.hist(series_valid, bins=50, color=colors[var_name], alpha=0.5, density=True)
            
            # Mark threshold regions
            stat_results = methods_results['statistical']
            ax.axvline(stat_results['lower_threshold'], color='blue', linestyle='--', alpha=0.7)
            ax.axvline(stat_results['upper_threshold'], color='red', linestyle='--', alpha=0.7)
            
            # Add normalized KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series_valid)
            x_range = np.linspace(series_valid.min(), series_valid.max(), 200)
            kde_values = kde(x_range)
            # Normalize KDE to [0, 1]
            kde_normalized = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())
            ax.plot(x_range, kde_normalized, color=colors[var_name], linewidth=2)
            
            # Clear the histogram and replot with normalized density
            ax.clear()
            # Normalize histogram to match KDE scale
            hist_vals, bins, patches = ax.hist(series_valid, bins=50, color=colors[var_name], 
                                               alpha=0.5, density=True)
            hist_normalized = hist_vals / hist_vals.max()
            ax.clear()
            ax.bar(bins[:-1], hist_normalized, width=np.diff(bins), 
                   color=colors[var_name], alpha=0.5, align='edge')
            
            # Replot KDE and thresholds
            ax.plot(x_range, kde_normalized, color=colors[var_name], linewidth=2)
            ax.axvline(stat_results['lower_threshold'], color='blue', linestyle='--', alpha=0.7)
            ax.axvline(stat_results['upper_threshold'], color='red', linestyle='--', alpha=0.7)
            
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            # Set appropriate x-axis labels for distribution plots
            if var_name == 'ITF':
                ax.set_xlabel('ITF Transport (Sv)', fontsize=12)
            elif var_name == 'ENSO':
                ax.set_xlabel('MEI', fontsize=12)
            elif var_name == 'DMI':
                ax.set_xlabel('DMI', fontsize=12)
            ax.set_ylabel('Normalized Density', fontsize=12)
        
        # Add common x-axis labels for time series plots
        for col in range(3):
            axes[0, col].set_xlabel('Time', fontsize=12)
            axes[1, col].set_xlabel('Time', fontsize=12)
        
        # Add row labels on the left
        row_labels = ['Time Series\nwith Extrema', 'Composite\nScore', 'Method\nComparison', 'Distribution\nAnalysis']
        for row, label in enumerate(row_labels):
            axes[row, 0].text(-0.35, 0.5, label, transform=axes[row, 0].transAxes,
                             fontsize=13, ha='right', va='center', weight='bold')
        
        # Adjust layout with more space
        plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06, hspace=0.35, wspace=0.25)
        
        # Save in multiple formats
        for fmt in ['eps', 'pdf', 'png']:
            plt.savefig(f'{output_prefix}.{fmt}', dpi=300 if fmt == 'png' else None, 
                       bbox_inches='tight')
        
        plt.close()
    
    def run_complete_analysis(self):
        """Run complete extrema analysis pipeline"""
        # Analyze each variable
        self.analyze_variable('ITF', ['itf_g', 'itf_t', 'itf_s'])
        self.analyze_variable('ENSO', 'meiv2')
        self.analyze_variable('DMI', 'DMI_HadISST1.1')
        
        # Generate outputs
        report = self.generate_statistics_report()
        
        # Save report
        with open('../stats/extrema_evaluation_report.txt', 'w') as f:
            f.write(report)
        
        # Save all extrema events to separate file
        self.save_extrema_events('../stats/extrema_all_events.txt')
        
        # Save CSV results
        self.save_results_csv('../processed_data/extrema_evaluation_results.csv')
        
        # Create visualization
        self.create_visualization('../figs/extrema_evaluation_comprehensive')
        
        print("Extrema evaluation completed successfully!")
        print("Files saved:")
        print("  - Report: ../stats/extrema_evaluation_report.txt")
        print("  - All events: ../stats/extrema_all_events.txt")
        print("  - Data: ../processed_data/extrema_evaluation_results.csv")
        print("  - Figures: ../figs/extrema_evaluation_comprehensive.[eps/pdf/png]")


if __name__ == "__main__":
    analyzer = ExtremaAnalyzer('../processed_data/combined_climate_data.csv')
    analyzer.run_complete_analysis()
