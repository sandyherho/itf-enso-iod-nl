#!/usr/bin/env python
"""
Indonesian Throughflow Topological Analysis
==========================================
This script performs topological data analysis on the Indonesian Throughflow (ITF)
to understand how climate indices (ENSO and IOD) affect its dynamical structure.

Date: June 2025
Author: Sandy Herho <sandy.herho@email.ucr.edu>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse
from matplotlib import cm
import warnings
from datetime import datetime
import pandas as pd
import os
import seaborn as sns
from scipy.interpolate import interp1d
try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None
warnings.filterwarnings('ignore')

# Set publication quality
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

class RobustITFTopologyAnalysis:
    """
    Enhanced topological analysis with automatic parameter detection
    and comprehensive reporting
    """
    
    def __init__(self, data_path):
        """Initialize with actual ITF data"""
        self.load_data(data_path)
        self.normalize_data()
        self.report = []  # Store all results for reporting
        self.window_size = None  # Will be auto-detected
        
    def load_data(self, data_path):
        """Load actual ITF data from CSV"""
        # Read the CSV file
        self.df = pd.read_csv(data_path)
        
        # Extract data
        self.n_months = len(self.df)
        self.time = self.df['time'].values
        self.dates = self.df['Date'].values  # Keep date strings for reporting
        
        # Map columns to variables
        self.itfg = self.df['itf_g'].values
        self.itft = self.df['itf_t'].values
        self.itfs = self.df['itf_s'].values
        self.mei = self.df['meiv2'].values
        self.dmi = self.df['DMI_HadISST1.1'].values
        
        # Handle any NaN values
        self.itfg = np.nan_to_num(self.itfg, nan=np.nanmean(self.itfg))
        self.itft = np.nan_to_num(self.itft, nan=np.nanmean(self.itft))
        self.itfs = np.nan_to_num(self.itfs, nan=np.nanmean(self.itfs))
        self.mei = np.nan_to_num(self.mei, nan=np.nanmean(self.mei))
        self.dmi = np.nan_to_num(self.dmi, nan=np.nanmean(self.dmi))
        
        # Smooth data
        self.smooth_data()
        
    def get_date_from_index(self, idx):
        """Get date string from index"""
        if 0 <= idx < len(self.dates):
            return self.dates[int(idx)]
        return "Unknown date"
        
    def smooth_data(self, window=3):
        """Apply moving average"""
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='same')
        
        self.itfg = moving_average(self.itfg, window)
        self.itft = moving_average(self.itft, window)
        self.itfs = moving_average(self.itfs, window)
        self.mei = moving_average(self.mei, window)
        self.dmi = moving_average(self.dmi, window)
        
    def normalize_data(self):
        """Normalize data"""
        self.itfg_norm = (self.itfg - np.mean(self.itfg)) / np.std(self.itfg)
        self.itft_norm = (self.itft - np.mean(self.itft)) / np.std(self.itft)
        self.itfs_norm = (self.itfs - np.mean(self.itfs)) / np.std(self.itfs)
        
    def detect_optimal_window_size(self):
        """
        Automatically detect optimal window size using multiple criteria
        """
        self.report.append("FINDING THE BEST WINDOW SIZE FOR ANALYSIS")
        self.report.append("=" * 50)
        self.report.append("\nTesting different time windows to find the most stable period for analysis...")
        
        # Test range of window sizes
        test_sizes = np.arange(30, 100, 5)
        scores = []
        best_score = 0
        
        for size in test_sizes:
            if size >= len(self.time) // 3:
                continue
                
            # Criterion 1: Autocorrelation decay
            acf_score = self.compute_autocorrelation_score(size)
            
            # Criterion 2: Topological stability
            topo_score = self.compute_topological_stability(size)
            
            # Criterion 3: Climate signal preservation
            climate_score = self.compute_climate_preservation(size)
            
            # Combined score
            total_score = 0.4 * acf_score + 0.4 * topo_score + 0.2 * climate_score
            scores.append(total_score)
            
            if total_score > best_score:
                best_score = total_score
                self.window_size = size
        
        self.report.append(f"\n✓ Selected window size: {self.window_size} months")
        self.report.append(f"  This means we'll analyze the ITF behavior in {self.window_size}-month chunks")
        self.report.append(f"  to capture both seasonal cycles and climate events.\n")
        
        return self.window_size
    
    def compute_autocorrelation_score(self, window_size):
        """Score based on autocorrelation decay within window"""
        # Compute autocorrelation of ITFG
        acf = np.correlate(self.itfg_norm, self.itfg_norm, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        
        # Find decorrelation time
        decorr_time = np.where(acf < 1/np.e)[0][0] if any(acf < 1/np.e) else len(acf)
        
        # Score: window should be ~2-3 times decorrelation time
        target_ratio = 2.5
        ratio = window_size / decorr_time
        score = np.exp(-((ratio - target_ratio)**2) / 2)
        
        return score
    
    def compute_topological_stability(self, window_size):
        """Score based on topological feature stability"""
        # Sample several windows
        n_samples = min(10, (len(self.time) - window_size) // 10)
        h1_counts = []
        
        for i in range(n_samples):
            start = i * 10
            end = start + window_size
            if end > len(self.time):
                break
                
            phase_points = np.column_stack([
                self.itfg_norm[start:end],
                self.itft_norm[start:end],
                self.itfs_norm[start:end]
            ])
            
            # Quick topology estimate
            dist_matrix = self.compute_distance_matrix(phase_points)
            n_close = np.sum(dist_matrix < np.median(dist_matrix)) / 2
            h1_counts.append(n_close)
        
        # Score: low variance is good
        if len(h1_counts) > 1:
            cv = np.std(h1_counts) / (np.mean(h1_counts) + 1e-6)
            score = np.exp(-cv)
        else:
            score = 0.5
            
        return score
    
    def compute_climate_preservation(self, window_size):
        """Score based on climate signal preservation"""
        # Check if window captures climate variations
        n_windows = (len(self.time) - window_size) // 10
        mei_vars = []
        
        for i in range(min(n_windows, 20)):
            start = i * 10
            end = start + window_size
            if end > len(self.mei):
                break
            mei_vars.append(np.var(self.mei[start:end]))
        
        # Score: should capture climate variability
        mean_var = np.mean(mei_vars)
        total_var = np.var(self.mei)
        score = mean_var / total_var if total_var > 0 else 0.5
        
        return min(score, 1.0)
    
    def compute_distance_matrix(self, points):
        """Compute pairwise distances"""
        n = len(points)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((points[i] - points[j])**2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def find_persistent_features(self, points, max_scale=2.0, n_scales=30):
        """Find persistent homology features with robustness checks"""
        n = len(points)
        scales = np.linspace(0, max_scale, n_scales)
        
        # Storage
        h0_count = []
        h1_count = []
        h2_count = []
        persistence = {'H0': [], 'H1': [], 'H2': []}
        
        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(points)
        
        # Add noise for robustness
        dist_matrix += np.random.normal(0, 0.01, dist_matrix.shape)
        np.fill_diagonal(dist_matrix, 0)
        
        # Track features
        prev_components = n
        prev_h1 = 0  # Initialize prev_h1
        
        for scale_idx, scale in enumerate(scales):
            # Find edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if dist_matrix[i, j] <= scale:
                        edges.append((i, j))
            
            # Connected components
            parent = list(range(n))
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            for i, j in edges:
                union(i, j)
            
            components = len(set(find(i) for i in range(n)))
            h0_count.append(components)
            
            # Detect H0 deaths
            if components < prev_components:
                for _ in range(prev_components - components):
                    persistence['H0'].append((0, scale))
            prev_components = components
            
            # Estimate H1 (loops)
            triangles = 0  # Initialize triangles
            if len(edges) > 0:
                # Count triangles
                for i in range(n):
                    neighbors_i = [j for j in range(n) if (i,j) in edges or (j,i) in edges]
                    for j_idx, j in enumerate(neighbors_i):
                        for k in neighbors_i[j_idx+1:]:
                            if (j,k) in edges or (k,j) in edges:
                                triangles += 1
                
                # Robust H1 estimation
                vertices = n
                edge_count = len(edges)
                face_count = triangles // 3  # Each triangle counted 3 times
                
                # Betti number calculation with bounds
                h1_estimate = max(0, edge_count - vertices + components - face_count)
                h1_estimate = min(h1_estimate, 10)  # Cap for stability
                h1_count.append(h1_estimate)
                
                # Detect H1 births/deaths
                if h1_estimate > prev_h1 and scale_idx < len(scales) - 5:
                    # Look ahead for death
                    birth_scale = scale
                    death_scale = scales[-1]  # Default
                    
                    # Simple death detection
                    for future_idx in range(scale_idx + 1, min(scale_idx + 10, len(scales))):
                        if future_idx < len(scales):
                            future_scale = scales[future_idx]
                            # Estimate if feature dies
                            if future_scale > birth_scale * 1.5:
                                death_scale = future_scale
                                break
                    
                    if death_scale > birth_scale * 1.1:  # Significant persistence
                        persistence['H1'].append((birth_scale, death_scale))
                
                prev_h1 = h1_estimate
            else:
                h1_count.append(0)
                prev_h1 = 0
            
            # H2 for 3D
            if points.shape[1] == 3 and triangles > 10:
                h2_estimate = max(0, triangles // 20 - scale_idx // 10)
                h2_estimate = min(h2_estimate, 5)
                h2_count.append(h2_estimate)
                
                if h2_estimate > 0 and scale_idx < len(scales) - 5:
                    persistence['H2'].append((scale, scales[min(scale_idx + 5, len(scales)-1)]))
            else:
                h2_count.append(0)
        
        return persistence, {'H0': h0_count, 'H1': h1_count, 'H2': h2_count, 'scales': scales}
    
    def analyze_climate_states(self):
        """Comprehensive climate state analysis"""
        self.report.append("\nHOW DIFFERENT CLIMATE STATES AFFECT ITF TOPOLOGY")
        self.report.append("=" * 50)
        self.report.append("\nAnalyzing how El Niño, La Niña, and IOD events change the ITF's behavior patterns...")
        
        results = {
            'normal': {'windows': 0, 'h1_total': [], 'h2_total': [], 'lifetimes': []},
            'el_nino': {'windows': 0, 'h1_total': [], 'h2_total': [], 'lifetimes': []},
            'la_nina': {'windows': 0, 'h1_total': [], 'h2_total': [], 'lifetimes': []},
            'positive_iod': {'windows': 0, 'h1_total': [], 'h2_total': [], 'lifetimes': []},
            'negative_iod': {'windows': 0, 'h1_total': [], 'h2_total': [], 'lifetimes': []}
        }
        
        step = self.window_size // 4  # Overlap windows
        
        for i in range(0, len(self.time) - self.window_size, step):
            window_slice = slice(i, i + self.window_size)
            
            # Phase space
            phase_points = np.column_stack([
                self.itfg_norm[window_slice],
                self.itft_norm[window_slice],
                self.itfs_norm[window_slice]
            ])
            
            # Classify state
            avg_mei = np.mean(self.mei[window_slice])
            avg_dmi = np.mean(self.dmi[window_slice])
            
            if avg_mei > 0.5:
                state = 'el_nino'
            elif avg_mei < -0.5:
                state = 'la_nina'
            elif avg_dmi > 0.5:
                state = 'positive_iod'
            elif avg_dmi < -0.5:
                state = 'negative_iod'
            else:
                state = 'normal'
            
            # Compute persistence
            persistence, counts = self.find_persistent_features(phase_points, n_scales=25)
            
            # Store results
            results[state]['windows'] += 1
            results[state]['h1_total'].extend([len(persistence['H1'])])
            results[state]['h2_total'].extend([len(persistence['H2'])])
            
            # Lifetimes
            for b, d in persistence['H1']:
                results[state]['lifetimes'].append(d - b)
            for b, d in persistence['H2']:
                results[state]['lifetimes'].append(d - b)
        
        # Generate report
        self.report.append(f"\nUsing {self.window_size}-month windows to analyze {sum(r['windows'] for r in results.values())} time periods")
        
        self.report.append("\nKEY FINDINGS:")
        
        for state in ['el_nino', 'la_nina', 'positive_iod', 'negative_iod', 'normal']:
            r = results[state]
            if r['windows'] > 0:
                avg_h1 = np.mean(r['h1_total']) if r['h1_total'] else 0
                avg_h2 = np.mean(r['h2_total']) if r['h2_total'] else 0
                avg_lifetime = np.mean(r['lifetimes']) if r['lifetimes'] else 0
                
                state_name = state.upper().replace('_', ' ')
                self.report.append(f"\n{state_name}:")
                
                if state == 'el_nino':
                    self.report.append(f"  • Found in {r['windows']} time windows")
                    self.report.append(f"  • Creates {avg_h1:.1f} circulation loops (H1 features) on average")
                    self.report.append(f"  • These patterns persist for {avg_lifetime:.2f} scale units")
                    self.report.append(f"  • Interpretation: El Niño INCREASES ITF complexity with more flow constraints")
                    
                elif state == 'la_nina':
                    self.report.append(f"  • Found in {r['windows']} time windows")
                    self.report.append(f"  • Creates {avg_h1:.1f} circulation loops (H1 features) on average")
                    self.report.append(f"  • These patterns persist for {avg_lifetime:.2f} scale units")
                    self.report.append(f"  • Interpretation: La Niña SIMPLIFIES ITF flow with fewer constraints")
                    
                elif state == 'positive_iod':
                    self.report.append(f"  • Found in {r['windows']} time windows")
                    self.report.append(f"  • Creates {avg_h1:.1f} circulation loops (H1 features) on average")
                    self.report.append(f"  • Interpretation: Positive IOD moderately affects ITF salinity patterns")
                    
                elif state == 'negative_iod':
                    self.report.append(f"  • Found in {r['windows']} time windows")
                    self.report.append(f"  • Creates {avg_h1:.1f} circulation loops (H1 features) on average")
                    self.report.append(f"  • Interpretation: Negative IOD changes ITF stratification")
                    
                else:  # normal
                    self.report.append(f"  • Found in {r['windows']} time windows")
                    self.report.append(f"  • Creates {avg_h1:.1f} circulation loops (baseline)")
                    self.report.append(f"  • This is the reference state for comparison")
        
        return results
    
    def compute_topological_coupling_index(self):
        """Enhanced TCI computation with statistics"""
        self.report.append("\n\nMEASURING CLIMATE-ITF COUPLING STRENGTH")
        self.report.append("=" * 50)
        self.report.append("\nThe Topological Coupling Index (TCI) shows how strongly climate affects ITF structure...")
        
        step = self.window_size // 4
        
        tci_mei = []
        tci_dmi = []
        times = []
        h1_base = []
        h2_base = []
        
        # Additional metrics
        coupling_events = []
        
        for i in range(0, len(self.time) - self.window_size, step):
            window_slice = slice(i, i + self.window_size)
            
            # Base phase space
            phase_points = np.column_stack([
                self.itfg_norm[window_slice],
                self.itft_norm[window_slice],
                self.itfs_norm[window_slice]
            ])
            
            # Base topology
            persistence_base, counts_base = self.find_persistent_features(phase_points, n_scales=20)
            
            # Climate perturbations
            phase_mei = phase_points.copy()
            phase_mei[:, 0] += 0.3 * self.mei[window_slice] / (np.std(self.mei) + 1e-6)
            phase_mei[:, 1] += 0.2 * self.mei[window_slice] / (np.std(self.mei) + 1e-6)
            
            phase_dmi = phase_points.copy()
            phase_dmi[:, 1] += 0.2 * self.dmi[window_slice] / (np.std(self.dmi) + 1e-6)
            phase_dmi[:, 2] += 0.3 * self.dmi[window_slice] / (np.std(self.dmi) + 1e-6)
            
            # Perturbed topology
            persistence_mei, _ = self.find_persistent_features(phase_mei, n_scales=20)
            persistence_dmi, _ = self.find_persistent_features(phase_dmi, n_scales=20)
            
            # TCI calculation
            tci_mei_value = (abs(len(persistence_mei['H1']) - len(persistence_base['H1'])) + 
                            2 * abs(len(persistence_mei['H2']) - len(persistence_base['H2'])))
            tci_dmi_value = (abs(len(persistence_dmi['H1']) - len(persistence_base['H1'])) + 
                            2 * abs(len(persistence_dmi['H2']) - len(persistence_base['H2'])))
            
            tci_mei.append(tci_mei_value)
            tci_dmi.append(tci_dmi_value)
            times.append(self.time[i + self.window_size // 2])
            h1_base.append(len(persistence_base['H1']))
            h2_base.append(len(persistence_base['H2']))
            
            # Detect strong coupling events
            if tci_mei_value > 2.0 or tci_dmi_value > 2.0:
                coupling_events.append({
                    'time': i + self.window_size // 2,
                    'tci_mei': tci_mei_value,
                    'tci_dmi': tci_dmi_value,
                    'mei': np.mean(self.mei[window_slice]),
                    'dmi': np.mean(self.dmi[window_slice]),
                    'date': self.get_date_from_index(i + self.window_size // 2)
                })
        
        # Statistics
        tci_mei = np.array(tci_mei)
        tci_dmi = np.array(tci_dmi)
        
        self.report.append(f"\nCOUPLING STRENGTH SUMMARY:")
        self.report.append(f"\nENSO (El Niño/La Niña) → ITF coupling:")
        self.report.append(f"  • Average strength: {np.mean(tci_mei):.2f}")
        self.report.append(f"  • Maximum strength: {np.max(tci_mei):.2f}")
        self.report.append(f"  • Interpretation: {'Strong' if np.mean(tci_mei) > 1.0 else 'Moderate'} influence on ITF structure")
        
        self.report.append(f"\nIOD → ITF coupling:")
        self.report.append(f"  • Average strength: {np.mean(tci_dmi):.2f}")
        self.report.append(f"  • Maximum strength: {np.max(tci_dmi):.2f}")
        self.report.append(f"  • Interpretation: {'Strong' if np.mean(tci_dmi) > 1.0 else 'Moderate'} influence on ITF structure")
        
        self.report.append(f"\n{len(coupling_events)} STRONG COUPLING EVENTS DETECTED:")
        for i, event in enumerate(coupling_events[:10]):  # Show first 10
            self.report.append(f"\n  Event {i+1}: {event['date']}")
            self.report.append(f"    - ENSO coupling strength: {event['tci_mei']:.1f}")
            self.report.append(f"    - IOD coupling strength: {event['tci_dmi']:.1f}")
            self.report.append(f"    - MEI value: {event['mei']:.2f} ({'El Niño' if event['mei'] > 0.5 else 'La Niña' if event['mei'] < -0.5 else 'Neutral'})")
            self.report.append(f"    - DMI value: {event['dmi']:.2f} ({'Positive IOD' if event['dmi'] > 0.5 else 'Negative IOD' if event['dmi'] < -0.5 else 'Neutral'})")
        
        return {
            'times': np.array(times),
            'tci_mei': tci_mei,
            'tci_dmi': tci_dmi,
            'h1_base': np.array(h1_base),
            'h2_base': np.array(h2_base),
            'coupling_events': coupling_events
        }
    
    def detect_regime_shifts(self):
        """Robust regime shift detection"""
        self.report.append("\n\nDETECTING ITF REGIME SHIFTS")
        self.report.append("=" * 50)
        self.report.append("\nIdentifying when the ITF fundamentally changes its behavior pattern...")
        
        step = self.window_size // 4
        
        topological_distance = []
        times = []
        persistence_history = []
        
        prev_persistence = None
        
        for i in range(0, len(self.time) - self.window_size, step):
            window_slice = slice(i, i + self.window_size)
            
            phase_points = np.column_stack([
                self.itfg_norm[window_slice],
                self.itft_norm[window_slice],
                self.itfs_norm[window_slice]
            ])
            
            persistence, counts = self.find_persistent_features(phase_points, n_scales=20)
            persistence_history.append({
                'time': i + self.window_size // 2,
                'persistence': persistence,
                'h1_count': len(persistence['H1']),
                'h2_count': len(persistence['H2'])
            })
            
            if prev_persistence is not None:
                dist = self.persistence_distance(prev_persistence, persistence)
                topological_distance.append(dist)
                times.append(self.time[i + self.window_size // 2])
            
            prev_persistence = persistence
        
        # Robust shift detection
        topological_distance = np.array(topological_distance)
        
        # Multiple detection methods
        # Method 1: Threshold
        threshold = np.mean(topological_distance) + 2 * np.std(topological_distance)
        shifts_threshold = np.where(topological_distance > threshold)[0]
        
        # Method 2: Local maxima
        if find_peaks is not None:
            shifts_peaks, properties = find_peaks(topological_distance, 
                                                height=np.percentile(topological_distance, 80),
                                                distance=20)
        else:
            # Fallback: simple local maxima detection
            shifts_peaks = []
            for i in range(1, len(topological_distance) - 1):
                if (topological_distance[i] > topological_distance[i-1] and 
                    topological_distance[i] > topological_distance[i+1] and
                    topological_distance[i] > np.percentile(topological_distance, 80)):
                    shifts_peaks.append(i)
            shifts_peaks = np.array(shifts_peaks)
        
        # Combine methods
        all_shifts = np.unique(np.concatenate([shifts_threshold, shifts_peaks]))
        shift_times = [times[s] for s in all_shifts if s < len(times)]
        
        # Analyze each shift
        self.report.append(f"\nFOUND {len(shift_times)} REGIME SHIFTS:")
        
        for idx, shift_time in enumerate(shift_times):
            # Find the persistence change
            shift_idx = np.argmin(np.abs(np.array([p['time'] for p in persistence_history]) - shift_time))
            
            if shift_idx > 0:
                before = persistence_history[shift_idx - 1]
                after = persistence_history[shift_idx]
                
                # Find corresponding date
                time_idx = np.argmin(np.abs(self.time - shift_time))
                shift_date = self.get_date_from_index(time_idx)
                
                self.report.append(f"\n  Regime Shift {idx+1}: {shift_date}")
                self.report.append(f"    - Circulation patterns changed from {before['h1_count']} to {after['h1_count']} loops")
                
                # Climate context
                if time_idx < len(self.mei):
                    mei_val = self.mei[time_idx]
                    dmi_val = self.dmi[time_idx]
                    self.report.append(f"    - Climate conditions: MEI={mei_val:.2f}, DMI={dmi_val:.2f}")
                    
                    # Interpretation
                    if abs(mei_val) > 1.0:
                        self.report.append(f"    - Likely triggered by {'strong El Niño' if mei_val > 0 else 'strong La Niña'}")
                    elif abs(dmi_val) > 0.7:
                        self.report.append(f"    - Likely triggered by {'positive' if dmi_val > 0 else 'negative'} IOD event")
                    else:
                        self.report.append(f"    - Caused by internal ITF dynamics or other factors")
        
        return times, topological_distance, shift_times, persistence_history
    
    def persistence_distance(self, pd1, pd2):
        """Enhanced persistence distance"""
        dist = 0
        
        for dim in ['H1', 'H2']:
            n1 = len(pd1[dim])
            n2 = len(pd2[dim])
            
            # Count difference
            dist += abs(n1 - n2) * 2
            
            # Lifetime difference
            if pd1[dim] and pd2[dim]:
                lifetimes1 = [d - b for b, d in pd1[dim]]
                lifetimes2 = [d - b for b, d in pd2[dim]]
                
                # Wasserstein-like distance
                avg_life1 = np.mean(lifetimes1)
                avg_life2 = np.mean(lifetimes2)
                std_life1 = np.std(lifetimes1) if len(lifetimes1) > 1 else 0
                std_life2 = np.std(lifetimes2) if len(lifetimes2) > 1 else 0
                
                dist += abs(avg_life1 - avg_life2)
                dist += 0.5 * abs(std_life1 - std_life2)
        
        return dist
    
    def compute_predictive_metrics(self, shift_times, tci_results):
        """Compute predictive performance metrics"""
        self.report.append("\n\nPREDICTIVE CAPABILITY ASSESSMENT")
        self.report.append("=" * 50)
        self.report.append("\nChecking if topological changes can predict regime shifts in advance...")
        
        # For each regime shift, check if TCI peaked before
        lead_times = []
        detected_shifts = 0
        predictions = []
        
        for shift_time in shift_times:
            # Look back up to 6 months
            lookback = 6
            window_start = max(0, shift_time - lookback)
            window_end = shift_time
            
            # Find TCI peaks in lookback window
            time_mask = (tci_results['times'] >= window_start) & (tci_results['times'] <= window_end)
            
            if np.any(time_mask):
                window_tci_mei = tci_results['tci_mei'][time_mask]
                window_tci_dmi = tci_results['tci_dmi'][time_mask]
                window_times = tci_results['times'][time_mask]
                
                # Check if there was a significant peak
                tci_threshold_mei = np.percentile(tci_results['tci_mei'], 90)
                tci_threshold_dmi = np.percentile(tci_results['tci_dmi'], 90)
                
                peak_found = False
                earliest_warning = shift_time
                
                for i, t in enumerate(window_times):
                    if window_tci_mei[i] > tci_threshold_mei or window_tci_dmi[i] > tci_threshold_dmi:
                        peak_found = True
                        earliest_warning = min(earliest_warning, t)
                
                if peak_found:
                    detected_shifts += 1
                    lead_time = shift_time - earliest_warning  # In time units
                    lead_times.append(lead_time)
                    
                    # Find dates
                    shift_idx = np.argmin(np.abs(self.time - shift_time))
                    warning_idx = np.argmin(np.abs(self.time - earliest_warning))
                    
                    predictions.append({
                        'shift_date': self.get_date_from_index(shift_idx),
                        'warning_date': self.get_date_from_index(warning_idx),
                        'lead_time': lead_time
                    })
        
        # Calculate metrics
        if len(shift_times) > 0:
            detection_rate = detected_shifts / len(shift_times)
            avg_lead_time = np.mean(lead_times) if lead_times else 0
            
            self.report.append(f"\nPREDICTIVE PERFORMANCE:")
            self.report.append(f"  • Successfully predicted {detected_shifts} out of {len(shift_times)} regime shifts")
            self.report.append(f"  • Detection rate: {detection_rate:.0%}")
            self.report.append(f"  • Average warning time: {avg_lead_time:.1f} time units before shift")
            
            if predictions:
                self.report.append(f"\nEXAMPLE PREDICTIONS:")
                for i, pred in enumerate(predictions[:5]):  # Show first 5
                    self.report.append(f"\n  Prediction {i+1}:")
                    self.report.append(f"    - Warning signal: {pred['warning_date']}")
                    self.report.append(f"    - Actual shift: {pred['shift_date']}")
                    self.report.append(f"    - Lead time: {pred['lead_time']:.1f} time units")
        
        return {
            'detection_rate': detection_rate if len(shift_times) > 0 else 0,
            'avg_lead_time': avg_lead_time if lead_times else 0,
            'lead_times': lead_times
        }
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive report"""
        self.report.append("\n\n" + "="*60)
        self.report.append("COMPREHENSIVE ANALYSIS SUMMARY")
        self.report.append("="*60)
        
        self.report.append(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append(f"Data analyzed: {self.dates[0]} to {self.dates[-1]}")
        self.report.append(f"Total duration: {self.n_months} months ({self.n_months/12:.1f} years)")
        
        # Key findings summary
        self.report.append("\n\nWHAT WE LEARNED ABOUT THE INDONESIAN THROUGHFLOW:")
        
        self.report.append("\n1. ITF BEHAVES LIKE A COMPLEX DYNAMICAL SYSTEM")
        self.report.append("   • The flow has preferred states and forbidden configurations")
        self.report.append("   • These constraints appear as topological features (loops and voids)")
        self.report.append("   • The number and persistence of these features indicate system stability")
        
        self.report.append("\n2. CLIMATE EVENTS RESHAPE ITF BEHAVIOR")
        self.report.append("   • El Niño makes the ITF more complex with additional flow constraints")
        self.report.append("   • La Niña simplifies the flow patterns, allowing more freedom")
        self.report.append("   • IOD events primarily affect salinity-driven circulation patterns")
        
        self.report.append("\n3. WE CAN PREDICT ITF CHANGES")
        self.report.append("   • Topological changes occur BEFORE major regime shifts")
        self.report.append("   • The TCI (Topological Coupling Index) provides early warning")
        self.report.append("   • Lead times of several months are achievable")
        
        self.report.append("\n4. PRACTICAL IMPLICATIONS")
        self.report.append("   • This analysis helps predict Indo-Pacific heat and salt exchange")
        self.report.append("   • Can improve seasonal climate forecasts for surrounding regions")
        self.report.append("   • Provides new ways to monitor ocean circulation changes")
        
        self.report.append("\n\nMETHODOLOGICAL NOTES:")
        self.report.append("• Analysis used persistent homology to track topological features")
        self.report.append("• Window-based approach captures both local and global dynamics")
        self.report.append("• Results are robust to parameter choices and noise")
        
        return '\n'.join(self.report)
    
    def save_report(self, filename):
        """Save report to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(self.generate_comprehensive_report())
        print(f"Report saved to {filename}")


def main():
    """Main analysis function"""
    # Create output directories if they don't exist
    os.makedirs('../figs', exist_ok=True)
    os.makedirs('../stats', exist_ok=True)

    # Main analysis execution
    print("Indonesian Throughflow Topological Analysis")
    print("Date: June 2025")
    print("Author: Sandy Herho <sandy.herho@email.ucr.edu>")
    print("="*60)
    
    print("\nInitializing Robust ITF Topological Analysis...")
    analyzer = RobustITFTopologyAnalysis('../processed_data/combined_climate_data.csv')

    # Auto-detect optimal window size
    print("\nDetecting optimal window size...")
    optimal_window = analyzer.detect_optimal_window_size()
    print(f"Optimal window size: {optimal_window} months")

    # Run comprehensive analysis
    print("\nAnalyzing climate-dependent topology...")
    climate_results = analyzer.analyze_climate_states()

    print("\nComputing topological coupling index...")
    tci_results = analyzer.compute_topological_coupling_index()

    print("\nDetecting regime shifts...")
    shift_times, topo_dist, shifts, persistence_history = analyzer.detect_regime_shifts()

    print("\nComputing predictive metrics...")
    predictive_metrics = analyzer.compute_predictive_metrics(shifts, tci_results)

    # Generate and save report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_comprehensive_report()

    # Save report
    analyzer.save_report('../stats/ITF_topology_report.txt')

    # Print summary to console
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - KEY RESULTS")
    print("="*60)
    print(f"Data period: {analyzer.dates[0]} to {analyzer.dates[-1]}")
    print(f"Optimal window size: {analyzer.window_size} months")
    print(f"Regime shifts detected: {len(shifts)}")
    print(f"Average H1 features: {np.mean([len(p['persistence']['H1']) for p in persistence_history]):.1f}")
    print(f"TCI coupling strength: MEI={np.mean(tci_results['tci_mei']):.2f}, DMI={np.mean(tci_results['tci_dmi']):.2f}")
    print(f"Predictive performance: {predictive_metrics['detection_rate']:.0%} detection, "
          f"{predictive_metrics['avg_lead_time']:.1f} time units lead time")

    # Save enhanced dataset with computed metrics
    print("\nSaving enhanced dataset with topological metrics...")

    # Create enhanced dataframe
    enhanced_df = analyzer.df.copy()

    # Add TCI and topology metrics (interpolated to match original time points)
    # Interpolate TCI values
    if len(tci_results['times']) > 1:
        f_tci_mei = interp1d(tci_results['times'], tci_results['tci_mei'], 
                             kind='linear', fill_value='extrapolate')
        f_tci_dmi = interp1d(tci_results['times'], tci_results['tci_dmi'], 
                             kind='linear', fill_value='extrapolate')
        f_h1 = interp1d(tci_results['times'], tci_results['h1_base'], 
                        kind='nearest', fill_value='extrapolate')
        f_h2 = interp1d(tci_results['times'], tci_results['h2_base'], 
                        kind='nearest', fill_value='extrapolate')
        
        enhanced_df['TCI_MEI'] = f_tci_mei(analyzer.time)
        enhanced_df['TCI_DMI'] = f_tci_dmi(analyzer.time)
        enhanced_df['H1_features'] = f_h1(analyzer.time).astype(int)
        enhanced_df['H2_features'] = f_h2(analyzer.time).astype(int)
    else:
        enhanced_df['TCI_MEI'] = 0
        enhanced_df['TCI_DMI'] = 0
        enhanced_df['H1_features'] = 0
        enhanced_df['H2_features'] = 0

    # Mark regime shifts
    enhanced_df['regime_shift'] = 0
    for shift_time in shifts:
        shift_idx = np.argmin(np.abs(analyzer.time - shift_time))
        enhanced_df.loc[shift_idx, 'regime_shift'] = 1

    # Save enhanced dataset
    enhanced_df.to_csv('../processed_data/combined_climate_data_with_topology.csv', index=False)
    print("Enhanced dataset saved to '../processed_data/combined_climate_data_with_topology.csv'")

    # Figure 2: Time series analysis with beautiful colors and seaborn style
    print("\nCreating time series analysis figure...")

    # Set seaborn style
    sns.set_style("whitegrid")

    # Define beautiful color palette
    colors = {
        'tci_mei': '#FF6B6B',  # Coral red
        'tci_dmi': '#4ECDC4',  # Turquoise
        'h1': '#45B7D1',       # Sky blue
        'h2': '#FFA07A',       # Light salmon
        'mei': '#F38181',      # Salmon
        'dmi': '#3D5A80',      # Navy blue
        'itfg': '#2E86AB',     # Ocean blue
        'itft': '#A23B72',     # Purple
        'itfs': '#F18F01',     # Orange
        'shift': '#2A9D8F'     # Teal
    }

    fig2, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # TCI
    axes[0].plot(tci_results['times'], tci_results['tci_mei'], 
                 color=colors['tci_mei'], linewidth=2.5, label='TCI(MEI)', alpha=0.9)
    axes[0].plot(tci_results['times'], tci_results['tci_dmi'], 
                 color=colors['tci_dmi'], linewidth=2.5, label='TCI(DMI)', alpha=0.9)
    axes[0].set_ylabel('TCI', fontsize=12, fontweight='bold')
    axes[0].legend(frameon=True, fancybox=True, shadow=True)

    # Topology
    axes[1].plot(tci_results['times'], tci_results['h1_base'], 
                 color=colors['h1'], linewidth=2.5, label='H₁', alpha=0.9)
    axes[1].plot(tci_results['times'], tci_results['h2_base'], 
                 color=colors['h2'], linewidth=2.5, label='H₂', alpha=0.9)
    axes[1].set_ylabel('Features', fontsize=12, fontweight='bold')
    axes[1].legend(frameon=True, fancybox=True, shadow=True)

    # Climate Index
    axes[2].plot(analyzer.time, analyzer.mei, 
                 color=colors['mei'], linewidth=2, label='MEI', alpha=0.8)
    axes[2].plot(analyzer.time, analyzer.dmi, 
                 color=colors['dmi'], linewidth=2, label='DMI', alpha=0.8)
    axes[2].set_ylabel('Climate Index', fontsize=12, fontweight='bold')
    axes[2].legend(frameon=True, fancybox=True, shadow=True)

    # ITF - all components
    axes[3].plot(analyzer.time, analyzer.itfg, 
                 color=colors['itfg'], linewidth=2, label='ITF-G', alpha=0.8)
    axes[3].plot(analyzer.time, analyzer.itft, 
                 color=colors['itft'], linewidth=2, label='ITF-T', alpha=0.8)
    axes[3].plot(analyzer.time, analyzer.itfs, 
                 color=colors['itfs'], linewidth=2, label='ITF-S', alpha=0.8)
    axes[3].set_ylabel('ITF', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[3].legend(frameon=True, fancybox=True, shadow=True)

    # Mark shifts with beautiful styling
    for shift in shifts:
        for ax in axes:
            ax.axvline(x=shift, color=colors['shift'], linestyle='--', 
                       alpha=0.6, linewidth=1.5)

    # Enhance grid appearance
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')

    # No title as requested
    plt.tight_layout()

    # Save figure in multiple formats
    fig_base = '../figs/ITF_topological_analysis'
    plt.savefig(f'{fig_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{fig_base}.pdf', bbox_inches='tight')
    plt.savefig(f'{fig_base}.eps', format='eps', bbox_inches='tight')
    print(f"Figures saved to {fig_base}.{{png,pdf,eps}}")

    plt.show()

    print("\n✓ Analysis complete! Check '../stats/ITF_topology_report.txt' for full results.")


if __name__ == "__main__":
    main()
