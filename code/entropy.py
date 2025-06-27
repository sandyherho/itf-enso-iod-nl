#!/usr/bin/env python
"""
Indonesian Throughflow Multi-Entropy Analysis
============================================
This script applies multiple entropy-based methods to quantify information flow
and coupling between climate indices (ENSO/IOD) and ITF components.

Date: June 2025
Author: Sandy Herho <sandy.herho@email.ucr.edu>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import os
from itertools import permutations
from collections import Counter

warnings.filterwarnings('ignore')

# Set publication quality
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


class MultiEntropyAnalysis:
    """
    Comprehensive entropy-based analysis for climate-ocean coupling
    """
    
    def __init__(self, data_path):
        """Initialize with data"""
        self.load_data(data_path)
        self.results = {}
        self.report = []
        
    def load_data(self, data_path):
        """Load and prepare data"""
        self.df = pd.read_csv(data_path)
        
        # Extract time series
        self.time = self.df['time'].values
        self.dates = self.df['Date'].values
        
        # Climate indices
        self.mei = self.df['meiv2'].values
        self.dmi = self.df['DMI_HadISST1.1'].values
        
        # ITF components
        self.itf_g = self.df['itf_g'].values
        self.itf_t = self.df['itf_t'].values
        self.itf_s = self.df['itf_s'].values
        
        # Handle NaN values
        self._handle_nans()
        
        # Normalize for entropy calculations
        self.scaler = StandardScaler()
        self.mei_norm = self.scaler.fit_transform(self.mei.reshape(-1, 1)).flatten()
        self.dmi_norm = self.scaler.fit_transform(self.dmi.reshape(-1, 1)).flatten()
        self.itf_g_norm = self.scaler.fit_transform(self.itf_g.reshape(-1, 1)).flatten()
        self.itf_t_norm = self.scaler.fit_transform(self.itf_t.reshape(-1, 1)).flatten()
        self.itf_s_norm = self.scaler.fit_transform(self.itf_s.reshape(-1, 1)).flatten()
        
    def _handle_nans(self):
        """Handle missing values"""
        for attr in ['mei', 'dmi', 'itf_g', 'itf_t', 'itf_s']:
            data = getattr(self, attr)
            if np.any(np.isnan(data)):
                setattr(self, attr, np.nan_to_num(data, nan=np.nanmean(data)))
    
    def shannon_entropy(self, x, bins=20):
        """Calculate Shannon entropy"""
        # Check for valid data
        if len(x) == 0 or np.std(x) < 1e-10:
            return 0.0
            
        hist, _ = np.histogram(x, bins=bins)
        
        # Check if histogram is empty
        if hist.sum() == 0:
            return 0.0
            
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        
        if len(hist) == 0:
            return 0.0
            
        return -np.sum(hist * np.log2(hist))
    
    def mutual_information(self, x, y, bins=20):
        """Calculate mutual information between two variables"""
        c_xy = np.histogram2d(x, y, bins)[0]
        c_x = np.histogram(x, bins)[0]
        c_y = np.histogram(y, bins)[0]
        
        # Normalize to get probabilities
        c_xy = c_xy / c_xy.sum()
        c_x = c_x / c_x.sum()
        c_y = c_y / c_y.sum()
        
        # Calculate MI
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if c_xy[i, j] > 0 and c_x[i] > 0 and c_y[j] > 0:
                    mi += c_xy[i, j] * np.log2(c_xy[i, j] / (c_x[i] * c_y[j]))
        
        return mi
    
    def transfer_entropy(self, x, y, lag=1, bins=10):
        """Calculate transfer entropy from x to y"""
        # Ensure we have enough data
        if len(x) <= lag or len(y) <= lag:
            return 0.0
            
        # Create lagged versions
        x_lag = x[:-lag]
        y_lag = y[:-lag]
        y_future = y[lag:]
        
        # Ensure all arrays have the same length
        min_len = min(len(x_lag), len(y_lag), len(y_future))
        x_lag = x_lag[:min_len]
        y_lag = y_lag[:min_len]
        y_future = y_future[:min_len]
        
        # Joint probabilities
        xyz = np.column_stack([y_future, y_lag, x_lag])
        yz = np.column_stack([y_future, y_lag])
        
        # Estimate entropies
        h_xyz = self._joint_entropy(xyz, bins)
        h_yz = self._joint_entropy(yz, bins)
        h_y = self.shannon_entropy(y_lag, bins)
        h_xy = self._joint_entropy(np.column_stack([x_lag, y_lag]), bins)
        
        # Transfer entropy
        te = h_xy + h_yz - h_xyz - h_y
        
        return max(0, te)  # Ensure non-negative
    
    def _joint_entropy(self, data, bins):
        """Calculate joint entropy for multivariate data"""
        if data.ndim == 1:
            return self.shannon_entropy(data, bins)
        
        # Check for valid data
        if len(data) == 0:
            return 0.0
            
        # Discretize each dimension
        digitized = np.zeros_like(data, dtype=int)
        for i in range(data.shape[1]):
            # Check if all values are the same
            if np.std(data[:, i]) < 1e-10:
                digitized[:, i] = 0
            else:
                _, edges = np.histogram(data[:, i], bins)
                digitized[:, i] = np.digitize(data[:, i], edges[1:-1])
        
        # Count occurrences
        unique, counts = np.unique(digitized, axis=0, return_counts=True)
        
        if len(counts) == 0:
            return 0.0
            
        probs = counts / counts.sum()
        
        # Calculate entropy
        probs_positive = probs[probs > 0]
        if len(probs_positive) == 0:
            return 0.0
            
        return -np.sum(probs_positive * np.log2(probs_positive))
    
    def permutation_entropy(self, x, order=3, delay=1):
        """Calculate permutation entropy"""
        n = len(x)
        
        # Check if series is too short
        if n < order * delay:
            return 0.0
            
        permutations_list = list(permutations(range(order)))
        c = Counter()
        
        for i in range(n - delay * (order - 1)):
            # Get order vector
            indices = [i + j * delay for j in range(order)]
            if indices[-1] < n:  # Ensure we don't go out of bounds
                sorted_idx = np.argsort([x[idx] for idx in indices])
                pattern = tuple(sorted_idx)
                c[pattern] += 1
        
        # Calculate probabilities
        if len(c) == 0:
            return 0.0
            
        probs = np.array(list(c.values())) / sum(c.values())
        
        # Calculate entropy
        pe = -np.sum(probs * np.log2(probs))
        
        # Normalize
        max_entropy = np.log2(len(permutations_list))
        return pe / max_entropy if max_entropy > 0 else 0.0
    
    def sample_entropy(self, x, m=2, r=0.2):
        """Calculate sample entropy"""
        N = len(x)
        
        # Check if series is too short
        if N < m + 1:
            return 0.0
            
        r = r * np.std(x)
        
        def _maxdist(xi, xj, m):
            """Maximum distance between vectors"""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([x[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if i != j and _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1
            
            # Avoid log(0)
            phi = np.sum(np.log(C + 1)) / (N - m + 1)
            return phi
        
        # Check if we can compute both phi(m) and phi(m+1)
        if N < m + 2:
            return 0.0
            
        return _phi(m) - _phi(m + 1)
    
    def approximate_entropy(self, x, m=2, r=0.2):
        """Calculate approximate entropy"""
        N = len(x)
        
        # Check if series is too short
        if N < m + 1:
            return 0.0
            
        r = r * np.std(x)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([x[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1
            
            # Avoid log(0) and division issues
            phi = 0
            for i in range(N - m + 1):
                if C[i] > 0:
                    phi += np.log(C[i] / (N - m + 1))
            phi = phi / (N - m + 1)
            return phi
        
        # Check if we can compute both phi(m) and phi(m+1)
        if N < m + 2:
            return 0.0
            
        return _phi(m) - _phi(m + 1)
    
    def multiscale_entropy(self, x, scales=range(1, 11), m=2, r=0.2):
        """Calculate multiscale entropy"""
        mse = []
        
        for scale in scales:
            # Coarse-grain the time series
            if scale == 1:
                y = x
            else:
                # Check if we have enough data for this scale
                if len(x) < scale:
                    mse.append(np.nan)
                    continue
                    
                # Coarse-grain by averaging
                n_segments = len(x) // scale
                if n_segments < m + 1:  # Not enough segments for entropy calculation
                    mse.append(np.nan)
                    continue
                    
                y = np.array([np.mean(x[i*scale:(i+1)*scale]) for i in range(n_segments)])
            
            # Calculate sample entropy
            if len(y) > m + 1:
                se = self.sample_entropy(y, m, r)
                mse.append(se)
            else:
                mse.append(np.nan)
        
        return np.array(mse)
    
    def conditional_entropy(self, x, y, bins=20):
        """Calculate conditional entropy H(Y|X)"""
        # Check for valid data
        if len(x) == 0 or len(y) == 0 or len(x) != len(y):
            return 0.0
            
        # Joint and marginal entropies
        h_xy = self._joint_entropy(np.column_stack([x, y]), bins)
        h_x = self.shannon_entropy(x, bins)
        
        # Conditional entropy cannot be negative
        return max(0, h_xy - h_x)
    
    def relative_entropy(self, x, y, bins=20):
        """Calculate relative entropy (KL divergence)"""
        # Check for valid data
        if len(x) == 0 or len(y) == 0:
            return 0.0
            
        # Get probability distributions with same bins
        x_min = min(x.min(), y.min())
        x_max = max(x.max(), y.max())
        edges = np.linspace(x_min, x_max, bins + 1)
        
        p, _ = np.histogram(x, bins=edges, density=True)
        q, _ = np.histogram(y, bins=edges, density=True)
        
        # Normalize to probabilities
        if p.sum() > 0:
            p = p / p.sum()
        else:
            return 0.0
            
        if q.sum() > 0:
            q = q / q.sum()
        else:
            return 0.0
        
        # Avoid log(0) and division by zero
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Renormalize
        p = p / p.sum()
        q = q / q.sum()
        
        # KL divergence
        kl = np.sum(p * np.log2(p / q))
        
        return max(0, kl)  # KL divergence is non-negative
    
    def cross_entropy(self, x, y, bins=20):
        """Calculate cross entropy"""
        # Check for valid data
        if len(x) == 0 or len(y) == 0:
            return 0.0
            
        # Get probability distributions with same bins
        x_min = min(x.min(), y.min())
        x_max = max(x.max(), y.max())
        edges = np.linspace(x_min, x_max, bins + 1)
        
        p, _ = np.histogram(x, bins=edges, density=True)
        q, _ = np.histogram(y, bins=edges, density=True)
        
        # Normalize to probabilities
        if p.sum() > 0:
            p = p / p.sum()
        else:
            return 0.0
            
        if q.sum() > 0:
            q = q / q.sum()
        else:
            return 0.0
        
        # Avoid log(0)
        epsilon = 1e-10
        q = q + epsilon
        
        # Renormalize q
        q = q / q.sum()
        
        # Cross entropy
        # Only compute where p > 0 to avoid unnecessary computation
        mask = p > epsilon
        if not np.any(mask):
            return 0.0
            
        ce = -np.sum(p[mask] * np.log2(q[mask]))
        
        return max(0, ce)  # Cross entropy is non-negative
    
    def causality_ratio(self, x, y, lag_range=range(1, 13)):
        """Calculate causality ratio using transfer entropy"""
        te_xy = []
        te_yx = []
        
        for lag in lag_range:
            if lag < len(x) and lag < len(y):  # Ensure we have enough data
                te_xy.append(self.transfer_entropy(x, y, lag))
                te_yx.append(self.transfer_entropy(y, x, lag))
        
        # Average over lags
        if len(te_xy) > 0 and len(te_yx) > 0:
            avg_te_xy = np.mean(te_xy)
            avg_te_yx = np.mean(te_yx)
            
            # Causality ratio
            if avg_te_xy + avg_te_yx > 0:
                ratio = avg_te_xy / (avg_te_xy + avg_te_yx)
            else:
                ratio = 0.5
        else:
            ratio = 0.5
            te_xy = [0] * len(lag_range)
            te_yx = [0] * len(lag_range)
        
        return ratio, te_xy, te_yx
    
    def compute_all_metrics(self):
        """Compute all entropy metrics for all variable pairs"""
        self.report.append("MULTI-ENTROPY ANALYSIS OF CLIMATE-ITF COUPLING")
        self.report.append("=" * 60)
        self.report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append(f"Data Period: {self.dates[0]} to {self.dates[-1]}")
        self.report.append(f"Total Months: {len(self.time)}")
        
        # Define variable pairs
        climate_vars = {'MEI': self.mei_norm, 'DMI': self.dmi_norm}
        itf_vars = {'ITF-G': self.itf_g_norm, 'ITF-T': self.itf_t_norm, 'ITF-S': self.itf_s_norm}
        
        # Initialize results storage
        self.results = {
            'shannon': {},
            'mutual_info': {},
            'transfer_entropy': {},
            'permutation': {},
            'sample': {},
            'approximate': {},
            'conditional': {},
            'relative': {},
            'cross': {},
            'causality_ratio': {},
            'multiscale': {},
            'ensemble_score': {}
        }
        
        # Compute metrics for each pair
        for climate_name, climate_data in climate_vars.items():
            for itf_name, itf_data in itf_vars.items():
                pair_name = f"{climate_name}→{itf_name}"
                
                print(f"\nAnalyzing {pair_name}...")
                
                # Shannon entropy
                print(f"  Computing Shannon entropy...")
                self.results['shannon'][pair_name] = {
                    'climate': self.shannon_entropy(climate_data),
                    'itf': self.shannon_entropy(itf_data)
                }
                
                # Mutual information
                print(f"  Computing mutual information...")
                self.results['mutual_info'][pair_name] = self.mutual_information(climate_data, itf_data)
                
                # Transfer entropy (multiple lags)
                print(f"  Computing transfer entropy...")
                te_values = []
                for lag in range(1, 13):
                    te = self.transfer_entropy(climate_data, itf_data, lag)
                    te_values.append(te)
                self.results['transfer_entropy'][pair_name] = {
                    'values': te_values,
                    'max': max(te_values) if te_values else 0,
                    'mean': np.mean(te_values) if te_values else 0,
                    'optimal_lag': np.argmax(te_values) + 1 if te_values else 1
                }
                
                # Permutation entropy
                print(f"  Computing permutation entropy...")
                self.results['permutation'][pair_name] = {
                    'climate': self.permutation_entropy(climate_data),
                    'itf': self.permutation_entropy(itf_data)
                }
                
                # Sample entropy
                print(f"  Computing sample entropy...")
                self.results['sample'][pair_name] = {
                    'climate': self.sample_entropy(climate_data),
                    'itf': self.sample_entropy(itf_data)
                }
                
                # Approximate entropy
                print(f"  Computing approximate entropy...")
                self.results['approximate'][pair_name] = {
                    'climate': self.approximate_entropy(climate_data),
                    'itf': self.approximate_entropy(itf_data)
                }
                
                # Conditional entropy
                print(f"  Computing conditional entropy...")
                self.results['conditional'][pair_name] = {
                    'itf_given_climate': self.conditional_entropy(climate_data, itf_data),
                    'climate_given_itf': self.conditional_entropy(itf_data, climate_data)
                }
                
                # Relative entropy
                print(f"  Computing relative entropy...")
                self.results['relative'][pair_name] = self.relative_entropy(climate_data, itf_data)
                
                # Cross entropy
                print(f"  Computing cross entropy...")
                self.results['cross'][pair_name] = self.cross_entropy(climate_data, itf_data)
                
                # Causality ratio
                print(f"  Computing causality ratio...")
                ratio, te_xy, te_yx = self.causality_ratio(climate_data, itf_data)
                self.results['causality_ratio'][pair_name] = {
                    'ratio': ratio,
                    'te_climate_to_itf': te_xy,
                    'te_itf_to_climate': te_yx
                }
                
                # Multiscale entropy
                print(f"  Computing multiscale entropy...")
                self.results['multiscale'][pair_name] = {
                    'climate': self.multiscale_entropy(climate_data),
                    'itf': self.multiscale_entropy(itf_data)
                }
                
                print(f"  Completed {pair_name}")
        
        # Compute ensemble scores
        print("\nComputing ensemble scores...")
        self._compute_ensemble_scores()
        
        # Generate report
        print("Generating report...")
        self._generate_report()
    
    def _compute_ensemble_scores(self):
        """Compute ensemble scores combining all metrics"""
        self.report.append("\n\nENSEMBLE SCORING")
        self.report.append("=" * 60)
        
        for pair_name in self.results['mutual_info'].keys():
            # Normalize each metric to [0, 1]
            scores = []
            
            # Mutual information (higher = stronger coupling)
            mi = self.results['mutual_info'][pair_name]
            mi_normalized = min(mi / 2.0, 1.0)  # Theoretical max is log2(bins), cap at 1
            scores.append(('Mutual Information', mi_normalized))
            
            # Transfer entropy (higher = stronger directed influence)
            te = self.results['transfer_entropy'][pair_name]['mean']
            te_normalized = min(te / 0.5, 1.0)  # Cap at 1
            scores.append(('Transfer Entropy', te_normalized))
            
            # Causality ratio (closer to 1 = stronger climate→ITF influence)
            cr = self.results['causality_ratio'][pair_name]['ratio']
            scores.append(('Causality Ratio', cr))
            
            # Conditional entropy (lower = stronger dependence)
            ce = self.results['conditional'][pair_name]['itf_given_climate']
            # Normalize: if ce is high (near max entropy), dependence is low
            max_entropy = np.log2(20)  # Maximum entropy for 20 bins
            ce_normalized = 1.0 - min(ce / max_entropy, 1.0)
            scores.append(('Conditional Entropy', ce_normalized))
            
            # Sample entropy difference (larger difference = stronger influence)
            se_climate = self.results['sample'][pair_name]['climate']
            se_itf = self.results['sample'][pair_name]['itf']
            se_diff = abs(se_climate - se_itf)
            se_normalized = min(se_diff / 2.0, 1.0)
            scores.append(('Sample Entropy Diff', se_normalized))
            
            # Compute weighted ensemble score
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]  # Weights for each metric
            
            # Ensure all scores are valid numbers
            valid_scores = []
            valid_weights = []
            for i, (name, score) in enumerate(scores):
                if not np.isnan(score) and not np.isinf(score):
                    valid_scores.append(score)
                    valid_weights.append(weights[i])
            
            if len(valid_scores) > 0:
                # Normalize weights
                total_weight = sum(valid_weights)
                if total_weight > 0:
                    normalized_weights = [w/total_weight for w in valid_weights]
                    ensemble_score = sum(w * s for w, s in zip(normalized_weights, valid_scores))
                else:
                    ensemble_score = np.mean(valid_scores)
            else:
                ensemble_score = 0.0
            
            self.results['ensemble_score'][pair_name] = {
                'score': ensemble_score,
                'components': scores,
                'interpretation': self._interpret_score(ensemble_score)
            }
    
    def _interpret_score(self, score):
        """Interpret ensemble score"""
        if score >= 0.7:
            return "Very Strong Coupling"
        elif score >= 0.5:
            return "Strong Coupling"
        elif score >= 0.3:
            return "Moderate Coupling"
        elif score >= 0.1:
            return "Weak Coupling"
        else:
            return "Very Weak Coupling"
    
    def _generate_report(self):
        """Generate comprehensive report"""
        self.report.append("\n\nKEY FINDINGS")
        self.report.append("=" * 60)
        
        # Check if we have ensemble scores
        if not self.results['ensemble_score']:
            self.report.append("\nNo ensemble scores computed.")
            return
            
        # Sort by ensemble score
        sorted_pairs = sorted(self.results['ensemble_score'].items(), 
                            key=lambda x: x[1]['score'], reverse=True)
        
        self.report.append("\nRANKING OF CLIMATE-ITF COUPLING STRENGTH:")
        for rank, (pair, data) in enumerate(sorted_pairs, 1):
            self.report.append(f"\n{rank}. {pair}")
            self.report.append(f"   Ensemble Score: {data['score']:.3f} ({data['interpretation']})")
            
            # Check if transfer entropy results exist
            if pair in self.results['transfer_entropy']:
                optimal_lag = self.results['transfer_entropy'][pair].get('optimal_lag', 'N/A')
                self.report.append(f"   Optimal lag: {optimal_lag} months")
            
            # Check if causality ratio exists
            if pair in self.results['causality_ratio']:
                ratio = self.results['causality_ratio'][pair].get('ratio', 0.5)
                self.report.append(f"   Causality ratio: {ratio:.3f}")
        
        # Detailed analysis for each climate index
        for climate in ['MEI', 'DMI']:
            self.report.append(f"\n\n{climate} INFLUENCE ON ITF COMPONENTS:")
            self.report.append("-" * 40)
            
            climate_pairs = [(p, d) for p, d in sorted_pairs if p.startswith(climate)]
            
            if not climate_pairs:
                self.report.append(f"\nNo data available for {climate}")
                continue
                
            for pair, data in climate_pairs:
                itf_component = pair.split('→')[1]
                self.report.append(f"\n{itf_component}:")
                
                # Key metrics with error checking
                mi = self.results['mutual_info'].get(pair, 0)
                te_data = self.results['transfer_entropy'].get(pair, {})
                te = te_data.get('mean', 0)
                cr_data = self.results['causality_ratio'].get(pair, {})
                cr = cr_data.get('ratio', 0.5)
                
                self.report.append(f"  • Information shared: {mi:.3f} bits")
                self.report.append(f"  • Information flow: {te:.3f} bits/time")
                self.report.append(f"  • Directional influence: {cr:.1%} from {climate}")
                
                # Physical interpretation
                if climate == 'MEI':
                    if itf_component == 'ITF-G':
                        self.report.append(f"  • ENSO {'strongly' if data['score'] > 0.5 else 'moderately'} "
                                         f"controls total ITF transport")
                    elif itf_component == 'ITF-T':
                        self.report.append(f"  • ENSO {'significantly' if data['score'] > 0.5 else 'weakly'} "
                                         f"affects ITF temperature transport")
                    else:  # ITF-S
                        self.report.append(f"  • ENSO has {'strong' if data['score'] > 0.5 else 'limited'} "
                                         f"impact on ITF salinity transport")
                else:  # DMI
                    if itf_component == 'ITF-G':
                        self.report.append(f"  • IOD {'substantially' if data['score'] > 0.5 else 'marginally'} "
                                         f"modulates total ITF flow")
                    elif itf_component == 'ITF-T':
                        self.report.append(f"  • IOD {'clearly' if data['score'] > 0.5 else 'slightly'} "
                                         f"influences ITF heat transport")
                    else:  # ITF-S
                        self.report.append(f"  • IOD {'strongly' if data['score'] > 0.5 else 'weakly'} "
                                         f"affects ITF freshwater flux")
        
        # Summary
        self.report.append("\n\nSUMMARY OF INFORMATION FLOW:")
        self.report.append("=" * 60)
        
        # Find strongest connections
        mei_pairs = [(p, d) for p, d in sorted_pairs if p.startswith('MEI')]
        dmi_pairs = [(p, d) for p, d in sorted_pairs if p.startswith('DMI')]
        
        if mei_pairs:
            strongest_mei = max(mei_pairs, key=lambda x: x[1]['score'])
            self.report.append(f"\n• ENSO most strongly affects: {strongest_mei[0].split('→')[1]}")
        
        if dmi_pairs:
            strongest_dmi = max(dmi_pairs, key=lambda x: x[1]['score'])
            self.report.append(f"• IOD most strongly affects: {strongest_dmi[0].split('→')[1]}")
        
        # Average influence
        if mei_pairs:
            mei_avg = np.mean([d['score'] for p, d in mei_pairs])
            self.report.append(f"\n• Average ENSO influence: {mei_avg:.3f} ({self._interpret_score(mei_avg)})")
        else:
            mei_avg = 0
            
        if dmi_pairs:
            dmi_avg = np.mean([d['score'] for p, d in dmi_pairs])
            self.report.append(f"• Average IOD influence: {dmi_avg:.3f} ({self._interpret_score(dmi_avg)})")
        else:
            dmi_avg = 0
        
        if mei_avg > 0 and dmi_avg > 0:
            if mei_avg > dmi_avg:
                self.report.append(f"\n• ENSO is the dominant climate driver ({(mei_avg/dmi_avg-1)*100:.0f}% stronger)")
            else:
                self.report.append(f"\n• IOD is the dominant climate driver ({(dmi_avg/mei_avg-1)*100:.0f}% stronger)")
    
    def create_visualizations(self):
        """Create beautiful entropy-based visualizations"""
        # Set style
        sns.set_style("white")
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # Color palette
        colors = {
            'MEI': '#FF6B6B',  # Coral red
            'DMI': '#4ECDC4',  # Turquoise
            'ITF-G': '#2E86AB', # Ocean blue
            'ITF-T': '#A23B72', # Purple
            'ITF-S': '#F18F01', # Orange
        }
        
        # Create main figures
        print("Creating ensemble heatmap...")
        self._create_ensemble_heatmap(colors)
        print("Creating transfer entropy flow diagram...")
        self._create_transfer_entropy_flow(colors)
        print("Creating multiscale entropy comparison...")
        self._create_multiscale_comparison(colors)
        print("Creating causality network...")
        self._create_causality_network(colors)
    
    def _create_ensemble_heatmap(self, colors):
        """Create heatmap of ensemble scores"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Prepare data
        climate_indices = ['MEI', 'DMI']
        itf_components = ['ITF-G', 'ITF-T', 'ITF-S']
        
        # Create matrix
        score_matrix = np.zeros((len(climate_indices), len(itf_components)))
        for i, climate in enumerate(climate_indices):
            for j, itf in enumerate(itf_components):
                pair = f"{climate}→{itf}"
                score_matrix[i, j] = self.results['ensemble_score'][pair]['score']
        
        # Create custom colormap - use inferno for black-purple-orange-yellow gradient
        cmap = plt.cm.inferno
        # Adjust vmin/vmax based on actual data range for better contrast
        vmin = score_matrix.min() - 0.02
        vmax = score_matrix.max() + 0.02
        im = ax.imshow(score_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set ticks with more spacing
        ax.set_xticks(np.arange(len(itf_components)))
        ax.set_yticks(np.arange(len(climate_indices)))
        ax.set_xticklabels(itf_components, fontsize=14, fontweight='bold')
        ax.set_yticklabels(climate_indices, fontsize=14, fontweight='bold')
        
        # Add values with better formatting
        for i in range(len(climate_indices)):
            for j in range(len(itf_components)):
                value = score_matrix[i, j]
                # For inferno colormap, use white text on dark colors, black on bright yellow
                norm_value = (value - vmin) / (vmax - vmin)
                text_color = 'white' if norm_value < 0.7 else 'black'
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color=text_color, 
                             fontsize=13, fontweight='bold')
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Ensemble Coupling Score', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Labels with better spacing
        ax.set_xlabel('ITF Components', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Climate Indices', fontsize=16, fontweight='bold', labelpad=10)
        
        # NO TITLE AT ALL
        
        # Style - remove all spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add grid for better readability
        ax.set_xticks(np.arange(len(itf_components))-.5, minor=True)
        ax.set_yticks(np.arange(len(climate_indices))-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf', 'eps']:
            plt.savefig(f'../figs/entropy_ensemble_heatmap.{fmt}', 
                       dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        plt.show()
    
    def _create_transfer_entropy_flow(self, colors):
        """Create transfer entropy flow diagram"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for climate in ['MEI', 'DMI']:
            for itf in ['ITF-G', 'ITF-T', 'ITF-S']:
                ax = axes[plot_idx]
                pair = f"{climate}→{itf}"
                
                # Get transfer entropy values
                te_forward = self.results['causality_ratio'][pair]['te_climate_to_itf']
                te_backward = self.results['causality_ratio'][pair]['te_itf_to_climate']
                lags = range(1, len(te_forward) + 1)
                
                # Plot with thicker lines
                ax.plot(lags, te_forward, 'o-', color=colors[climate], 
                       linewidth=3, markersize=10, label=f'{climate}→{itf}', alpha=0.9)
                ax.plot(lags, te_backward, 's--', color=colors[itf], 
                       linewidth=2.5, markersize=8, alpha=0.7, label=f'{itf}→{climate}')
                
                # Optimal lag
                optimal_lag = self.results['transfer_entropy'][pair]['optimal_lag']
                ax.axvline(x=optimal_lag, color='gray', linestyle=':', alpha=0.5, linewidth=2)
                ax.text(optimal_lag, ax.get_ylim()[1] * 0.9, f'Optimal\nlag: {optimal_lag}', 
                       ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor='white', edgecolor='gray', alpha=0.8))
                
                # Style with better spacing
                ax.set_xlabel('Lag (months)', fontsize=14, fontweight='bold', labelpad=8)
                ax.set_ylabel('Transfer Entropy (bits)', fontsize=14, fontweight='bold', labelpad=8)
                
                # NO SUBPLOT TITLES OR LABELS
                
                ax.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plot_idx += 1
        
        # NO MAIN TITLE
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
        
        # Save
        for fmt in ['png', 'pdf', 'eps']:
            plt.savefig(f'../figs/entropy_transfer_flow.{fmt}', 
                       dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        plt.show()
    
    def _create_multiscale_comparison(self, colors):
        """Create multiscale entropy comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        scales = range(1, 11)
        plot_idx = 0
        
        for climate in ['MEI', 'DMI']:
            for itf in ['ITF-G', 'ITF-T', 'ITF-S']:
                ax = axes[plot_idx]
                pair = f"{climate}→{itf}"
                
                # Get multiscale entropy
                mse_climate = self.results['multiscale'][pair]['climate']
                mse_itf = self.results['multiscale'][pair]['itf']
                
                # Remove NaN values for plotting
                valid_scales = []
                valid_mse_climate = []
                valid_mse_itf = []
                
                for i, scale in enumerate(scales):
                    if i < len(mse_climate) and i < len(mse_itf):
                        if not np.isnan(mse_climate[i]) and not np.isnan(mse_itf[i]):
                            valid_scales.append(scale)
                            valid_mse_climate.append(mse_climate[i])
                            valid_mse_itf.append(mse_itf[i])
                
                if len(valid_scales) > 0:
                    # Plot with thicker lines and larger markers
                    ax.plot(valid_scales, valid_mse_climate, 'o-', color=colors[climate], 
                           linewidth=3, markersize=10, label=climate, alpha=0.9)
                    ax.plot(valid_scales, valid_mse_itf, 's-', color=colors[itf], 
                           linewidth=3, markersize=10, label=itf, alpha=0.9)
                    
                    # Fill between with more transparency
                    ax.fill_between(valid_scales, valid_mse_climate, valid_mse_itf, 
                                   alpha=0.15, color='gray')
                
                # Style with better spacing
                ax.set_xlabel('Scale', fontsize=14, fontweight='bold', labelpad=8)
                ax.set_ylabel('Sample Entropy', fontsize=14, fontweight='bold', labelpad=8)
                
                # NO SUBPLOT TITLES
                
                ax.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Set y-axis limits for better visibility
                if len(valid_mse_climate) > 0 and len(valid_mse_itf) > 0:
                    y_min = min(min(valid_mse_climate), min(valid_mse_itf)) * 0.9
                    y_max = max(max(valid_mse_climate), max(valid_mse_itf)) * 1.1
                    ax.set_ylim(y_min, y_max)
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plot_idx += 1
        
        # NO MAIN TITLE
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
        
        # Save
        for fmt in ['png', 'pdf', 'eps']:
            plt.savefig(f'../figs/entropy_multiscale_comparison.{fmt}', 
                       dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        plt.show()
    
    def _create_causality_network(self, colors):
        """Create causality network diagram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # MEI network
        self._draw_network(ax1, 'MEI', colors)
        # NO TITLE
        
        # DMI network
        self._draw_network(ax2, 'DMI', colors)
        # NO TITLE
        
        plt.tight_layout(pad=2.0)
        
        # Save
        for fmt in ['png', 'pdf', 'eps']:
            plt.savefig(f'../figs/entropy_causality_network.{fmt}', 
                       dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        plt.show()
    
    def _draw_network(self, ax, climate, colors):
        """Draw causality network for one climate index"""
        from matplotlib.patches import FancyArrowPatch
        
        # Node positions with better spacing
        climate_pos = (0.15, 0.5)
        itf_positions = {
            'ITF-G': (0.85, 0.75),
            'ITF-T': (0.85, 0.5),
            'ITF-S': (0.85, 0.25)
        }
        
        # Draw nodes with larger size
        ax.scatter(*climate_pos, s=3000, c=colors[climate], 
                  edgecolors='black', linewidth=3, zorder=3)
        ax.text(*climate_pos, climate, ha='center', va='center', 
               fontsize=16, fontweight='bold', color='white')
        
        for itf, pos in itf_positions.items():
            ax.scatter(*pos, s=2500, c=colors[itf], 
                      edgecolors='black', linewidth=3, zorder=3)
            ax.text(*pos, itf, ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        
        # Draw arrows with width proportional to coupling strength
        for itf, pos in itf_positions.items():
            pair = f"{climate}→{itf}"
            score = self.results['ensemble_score'][pair]['score']
            te = self.results['transfer_entropy'][pair]['mean']
            
            # Arrow width based on score
            arrow_width = 0.01 + score * 0.05
            
            # Color intensity based on score
            arrow_color = plt.cm.Greys(0.3 + score * 0.5)
            
            # Create curved arrow
            arrow = FancyArrowPatch(climate_pos, pos,
                                  connectionstyle="arc3,rad=.1",
                                  arrowstyle='->', 
                                  mutation_scale=arrow_width*1000,
                                  linewidth=arrow_width*60,
                                  color=arrow_color, 
                                  alpha=0.8,
                                  zorder=2)
            ax.add_patch(arrow)
            
            # Add score label with better formatting
            mid_x = (climate_pos[0] + pos[0]) / 2
            mid_y = (climate_pos[1] + pos[1]) / 2
            
            # Adjust label position based on ITF component
            if itf == 'ITF-G':
                mid_y += 0.05
            elif itf == 'ITF-S':
                mid_y -= 0.05
                
            ax.text(mid_x, mid_y, f'{score:.2f}', 
                   fontsize=12, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.4", 
                           facecolor='white', edgecolor='gray',
                           alpha=0.9))
        
        # Style
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add background grid for visual appeal
        ax.grid(True, alpha=0.05, color='gray', linestyle='-')
    
    def save_results(self):
        """Save results to files"""
        # Save detailed results
        results_df = pd.DataFrame()
        
        for pair in self.results['ensemble_score'].keys():
            row = {
                'Pair': pair,
                'Ensemble_Score': self.results['ensemble_score'][pair]['score'],
                'Interpretation': self.results['ensemble_score'][pair]['interpretation'],
                'Mutual_Information': self.results['mutual_info'][pair],
                'Transfer_Entropy_Mean': self.results['transfer_entropy'][pair]['mean'],
                'Transfer_Entropy_Max': self.results['transfer_entropy'][pair]['max'],
                'Optimal_Lag': self.results['transfer_entropy'][pair]['optimal_lag'],
                'Causality_Ratio': self.results['causality_ratio'][pair]['ratio'],
                'Conditional_Entropy': self.results['conditional'][pair]['itf_given_climate'],
                'Relative_Entropy': self.results['relative'][pair],
                'Cross_Entropy': self.results['cross'][pair]
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        # Save to CSV
        results_df.to_csv('../processed_data/entropy_analysis_results.csv', index=False)
        
        # Save report
        with open('../stats/entropy_analysis_report.txt', 'w') as f:
            f.write('\n'.join(self.report))
        
        print("\nResults saved to:")
        print("  - ../processed_data/entropy_analysis_results.csv")
        print("  - ../stats/entropy_analysis_report.txt")
        print("  - ../figs/entropy_*.{png,pdf,eps}")


def main():
    """Main execution function"""
    print("Indonesian Throughflow Multi-Entropy Analysis")
    print("Date: June 2025")
    print("Author: Sandy Herho <sandy.herho@email.ucr.edu>")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('../figs', exist_ok=True)
    os.makedirs('../stats', exist_ok=True)
    os.makedirs('../processed_data', exist_ok=True)
    
    # Initialize analyzer
    print("\nLoading data...")
    analyzer = MultiEntropyAnalysis('../processed_data/combined_climate_data.csv')
    
    # Compute all metrics
    print("\nComputing entropy metrics...")
    print("Note: Some entropy calculations (sample/approximate entropy) may take a few moments...")
    analyzer.compute_all_metrics()
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_visualizations()
    
    # Save results
    print("\nSaving results...")
    analyzer.save_results()
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
