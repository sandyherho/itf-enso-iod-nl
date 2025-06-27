# Supplementary Materials for "Topological and Information-Theoretic Analysis of Climate-Driven Indonesian Throughflow Dynamics"
[![DOI](https://zenodo.org/badge/1009813500.svg)](https://doi.org/10.5281/zenodo.15757950)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Data](https://img.shields.io/badge/Data-1984--2017-green)](./raw_data/)
[![Research](https://img.shields.io/badge/Research-Complex%20Systems-red)](https://en.wikipedia.org/wiki/Complex_system)
[![Climate](https://img.shields.io/badge/Climate-Ocean%20Dynamics-lightblue)](https://www.nature.com/subjects/ocean-sciences)

## Research Overview

This repository contains comprehensive supplementary materials for our investigation of the Indonesian Throughflow (ITF) system's response to large-scale climate modes (ENSO and IOD) using advanced nonlinear analysis techniques. We employ topological data analysis, information theory, and extreme value statistics to quantify the complex interactions between climate indices and ocean transport dynamics.

## Key Scientific Contributions

- **Novel Application of Persistent Homology**: First application of topological data analysis to characterize ITF phase space structure
- **Multi-Scale Entropy Framework**: Comprehensive information-theoretic quantification of climate-ocean coupling
- **Predictive Regime Detection**: Identification of early warning signals for ITF regime shifts with ~2.3 time units lead time
- **Extrema Coupling Analysis**: Statistical characterization of coincident extreme events across ITF-ENSO-IOD system

## Repository Structure

```
.
├── code/                      # Analysis scripts
│   ├── aligned_ts.py         # Time series alignment and preprocessing
│   ├── annual_cycle.py       # Bootstrap-based seasonal analysis
│   ├── topology.py           # Topological data analysis
│   ├── entropy.py            # Multi-entropy coupling analysis
│   ├── extrema.py            # Extreme value evaluation
│   └── map.py                # PyGMT visualization
│
├── raw_data/                 # Original datasets
│   ├── itf_ts.csv           # ITF transport components (G, T, S)
│   ├── meiv2.csv            # Multivariate ENSO Index v2
│   └── dmi.csv              # Dipole Mode Index
│
├── processed_data/           # Processed outputs
│   ├── combined_climate_data.csv
│   ├── annual_cycle_results.csv
│   ├── entropy_analysis_results.csv
│   └── extrema_evaluation_results.csv
│
├── figs/                     # Publication-quality figures
│   ├── map.png              # Study region map
│   ├── annual_cycle_analysis.*
│   ├── entropy_ensemble_heatmap.*
│   ├── ITF_topological_analysis.*
│   └── extrema_evaluation_comprehensive.*
│
└── stats/                    # Statistical reports
    ├── ITF_topology_report.txt
    ├── annual_cycle_report.txt
    ├── entropy_analysis_report.txt
    └── extrema_evaluation_report.txt
```

## Methodological Framework

### 1. Data Preprocessing (`aligned_ts.py`)
- Temporal alignment of multi-source climate datasets
- NaN interpolation using mean imputation
- Conversion to decimal year format
- Block maxima extraction (30-day blocks)

### 2. Annual Cycle Analysis (`annual_cycle.py`)
- Bootstrap resampling (n=20,000 iterations)
- 95% confidence interval estimation
- One-way ANOVA for seasonal significance
- Phase relationship quantification

### 3. Topological Analysis (`topology.py`)
- Persistent homology computation
- Automatic window size detection (45 months optimal)
- Topological Coupling Index (TCI) calculation
- Regime shift detection with predictive assessment

### 4. Information Theory (`entropy.py`)
- Shannon entropy calculation
- Transfer entropy with lag optimization (1-12 months)
- Multiscale entropy (scales 1-10)
- Ensemble coupling score synthesis

### 5. Extrema Evaluation (`extrema.py`)
- Eight complementary detection methods
- Composite scoring approach
- Cross-variable coincidence analysis
- GEV parameter estimation

## Key Findings

### Climate-ITF Coupling Strength
```
ENSO → ITF: 0.524 (Strong Coupling)
IOD → ITF:  0.500 (Strong Coupling)
Dominant Driver: ENSO (5% stronger influence)
```

### Seasonal Patterns
- ITF maximum: September (13.02 Sv)
- ITF minimum: April (-0.26 Sv)
- Seasonal amplitude: 13.28 Sv
- All components show highly significant seasonality (p < 0.001)

### Topological Insights
- 2 major regime shifts detected (1990, 2012)
- 100% predictive success rate for regime shifts
- Average warning time: 2.3 time units
- El Niño increases ITF complexity
- La Niña simplifies flow patterns

## Reproducibility

### System Requirements
- Python 3.8+
- NumPy, Pandas, SciPy, Matplotlib
- Seaborn, PyGMT, scikit-learn

### Installation
```bash
git clone https://github.com/yourusername/itf-climate-coupling.git
cd itf-climate-coupling
pip install -r requirements.txt
```

### Running the Analysis
```bash
# 1. Data preprocessing
python code/aligned_ts.py

# 2. Annual cycle analysis
python code/annual_cycle.py

# 3. Topological analysis
python code/topology.py

# 4. Entropy analysis
python code/entropy.py

# 5. Extrema evaluation
python code/extrema.py

# 6. Generate map
python code/map.py
```

## Data Availability

All datasets used in this study are provided in the `raw_data/` directory:
- **ITF Transport**: Monthly means (1984-2017) for geostrophic (G), temperature (T), and salinity (S) components
- **MEI v2**: NOAA Multivariate ENSO Index version 2
- **DMI**: Dipole Mode Index from HadISST1.1

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{herho202xitf,
  title={Topological and Information-Theoretic Analysis of Climate-Driven Indonesian Throughflow Dynamics},
  author={Herho, Sandy H. S. and Herho, Katarina E. P. and Anwar, Iwan P. and Suwarman, Rusmawan},
  journal={xxxx},
  year={202X},
  volume={XX},
  pages={XXX--XXX},
  doi={10.XXXX/XXXXXX}
}
```

## Contact

**Corresponding Author**: Sandy H. S. Herho  
**Email**: sandy.herho@email.ucr.edu  
**Institution**: University of California, Riverside  
**ORCID**: [0000-0001-8330-2095](https://orcid.org/0000-0001-8330-2095)

## License

This project is licensed under the WTFPL License - see the [LICENSE](LICENSE) file for details.
