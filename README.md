# Advanced Integrated Framework for Rail Freight Decarbonization Analysis

## Overview

This repository contains a comprehensive Python pipeline for rail freight data curation and decarbonization analysis, implementing a sophisticated integrated framework that combines multiple analytical approaches:

- **Hierarchical Bayesian Modeling** for uncertainty quantification
- **Copula-based Risk Simulation** for correlated demand forecasting  
- **Two-stage Stochastic Optimization** for resource allocation under uncertainty
- **Extended Regression Analysis** for operational performance decomposition
- **Competitive Pricing Simulation** for policy evaluation

The framework treats environmental costs as endogenous, optimizable variables, enabling joint improvement of profitability, service reliability, and decarbonization objectives.

## Core Capabilities

### 1. Multi-Source Data Integration
- **Temporal Coverage**: 1996-2018 operational data
- **Geographic Scope**: Complete Korean rail freight network
- **Data Types**: Timetables, ton-kilometers, commodity flows, tariffs, rolling stock, logistics infrastructure
- **Automated Curation**: Cleansing, normalization, enrichment, and validation pipelines

### 2. Advanced Analytical Methods
- **Environmental Accounting**: Transforms emissions from passive reporting to active cost components
- **Bayesian Inference**: Robust estimation under data sparsity using hierarchical models
- **Stochastic Programming**: Two-stage optimization with scenario-based uncertainty
- **Copula Models**: Captures non-linear dependencies between demand drivers
- **Extended Regression**: Fixed-effects models with operational confounders

### 3. Policy Simulation Tools
- **Carbon Pricing Integration**: Endogenizes environmental externalities
- **Competitive Response**: Models market dynamics and modal competition
- **Pareto Frontier Analysis**: Identifies revenue-emission trade-offs
- **Sensitivity Analysis**: Tests policy robustness under parameter uncertainty

## Requirements

### Core Dependencies
```
# Basic data science stack
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.3.0

# Advanced modeling
pymc>=5.7.0          # Hierarchical Bayesian modeling
pyomo>=6.7.0         # Optimization modeling framework

# Data processing
pyarrow>=12.0.0      # Parquet file support
python-dateutil>=2.8.0

# Optional performance
scikit-learn-intelex>=2023.0.0  # Accelerated ML algorithms
```

### Optimization Solvers (Choose at least one)
```
# Option 1: HiGHS (recommended)
pip install highspy

# Option 2: GLPK (open source)
# Ubuntu/Debian: sudo apt-get install glpk-utils
# macOS: brew install glpk
# Windows: conda install -c conda-forge glpk

# Option 3: CBC (open source)
# Ubuntu/Debian: sudo apt-get install coinor-cbc
# macOS: brew install cbc
# Windows: conda install -c conda-forge coincbc

# Option 4: Ipopt (nonlinear)
# Ubuntu/Debian: sudo apt-get install coinor-ipopt
# macOS: brew install ipopt
# Windows: conda install -c conda-forge ipopt
```

### Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core requirements
pip install -r requirements.txt

# Install optimization solver (choose one)
pip install highspy  # Recommended
# OR
conda install -c conda-forge glpk  # If using conda
```

## Data Structure

The framework processes 9 interconnected datasets:

1. **Train Timetable** (화운표): 217 scheduled freight trains with origins, destinations, departure/arrival times, frequencies, and routes
2. **Annual Tons by Commodity** (톤수실적): 23-year panel (1996-2018) of tonnage for 8 commodity classes
3. **Annual Ton-Kilometers** (톤키로실적): Corresponding transport work metrics
4. **Segment Frequencies**: Weekday/weekend container and bulk train frequencies by line
5. **Tariff Information**: Freight rates, container charges, yard fees, and ancillary services
6. **Annual Operations**: Train runs and kilometerage by class (bulk, container, parcel)
7. **Rolling Stock Inventory**: Payload ratings and tare weights by wagon type
8. **Logistics Bases**: CY, CFS, silos, warehouses, and specialized terminals
9. **Freight Stations**: 75 stations with service levels and regional classifications

## Usage

### Basic Execution
```bash
# Run complete pipeline (data curation + analysis)
python integrated_pipeline.py

# Output directories created:
# ./curated/           # Cleaned datasets (CSV/Parquet)
# ./analysis_results/  # Generated figures and results
```

### Step-by-Step Analysis

The pipeline executes sequentially:

#### Part A: Data Curation
1. Loads raw data from embedded sources
2. Parses Korean text, times, and dates
3. Normalizes line names and station codes
4. Creates derived variables (average speeds, overnight flags)
5. Exports 25 curated datasets

#### Part B: Decarbonization Analysis
1. **Environmental Accounting**: Validates emission factors, quantifies uncertainty sources
2. **Bayesian Modeling**: Reconstructs haul distances, quantifies prediction uncertainty
3. **Copula Simulation**: Models correlated demand, generates risk profiles
4. **Regression Analysis**: Decomposes transit time drivers, isolates overnight penalties
5. **Stochastic Optimization**: Allocates wagons under uncertainty, calculates VSS
6. **Pricing Simulation**: Tests policy impacts, maps Pareto frontiers

### Custom Analysis
```python
# Import curated data for custom analysis
import pandas as pd
df_demand = pd.read_parquet('./curated/annual_demand_panel.parquet')
df_timetable = pd.read_parquet('./curated/timetable_by_day.parquet')

# Access specific analysis components
from integrated_pipeline import (
    calculate_emissions,
    simulate_policy,
    run_stochastic_optimization
)
```

## Key Analytical Results

### Figure 3: Emission Factor Validation
- Validates baseline CO₂e factors against technology-specific measurements
- Quantifies uncertainty contributions from grid intensity, engine wear, calibration
- Establishes adjustment factors (α) by service type

### Figure 4: Hierarchical Bayesian Performance
- Bayesian models outperform classical ML in sparse-data commodities
- Reduces prediction error variance by 38% for low-sample commodities
- Enables reconstruction of complete ton-km panel from partial data

### Figure 7: Risk-Return Analysis
- Gyeongbu Line: Highest revenue (2.8M KRW/train) with moderate risk (CV=0.42)
- Jungang Line: Second-highest revenue (2.1M KRW/train) with higher risk (CV=0.58)
- Daegu Line: Optimal balance (1.9M KRW/train, CV=0.39)

### Figure 8: Regression Insights
- Base model R²: 0.642 vs Extended model R²: 0.798
- Overnight penalty: 33.7 minutes (20% explained by operational confounders)
- Key drivers: Distance (β=0.82), congestion (β=0.15), tonnage×gradient (β=0.11)

### Figure 9: Stochastic Optimization Value
- Value of Stochastic Solution (VSS): 0.35 B KRW/year
- Emissions reduction vs deterministic: 6.7%
- Service level improvement: 94.2% → 96.8%

### Figure 10-11: Policy Simulations
- Empty container rate increase from 74% → 84%: Revenue -8%, Emissions -12%
- Optimal balance zone identified at 19-22 ktCO₂e / 8.5-9.0 B KRW
- Carbon price sensitivity: Shifts allocation from coal (45→32 trains) to containers (8→18 trains)

## Methodology

### 1. Environmental Accounting
```python
# Emission calculation with operational adjustments
emissions = (gross_ton_km * γ_baseline * α_operational) / 1000
# where α_operational ∈ [1.0, 1.15] based on service characteristics
```

### 2. Hierarchical Bayesian Model
```python
# Three-level hierarchy: Global → Commodity → Observation
with pm.Model():
    μ_global = pm.Normal('μ_global', μ=250, σ=50)
    τ = pm.HalfCauchy('τ', 5)
    μ_c = pm.Normal('μ_c', μ=μ_global, σ=τ, shape=n_commodities)
    σ_c = pm.HalfCauchy('σ_c', 5, shape=n_commodities)
    likelihood = pm.Normal('haul_est', μ=μ_c[commodity_idx], σ=σ_c[commodity_idx], observed=y)
```

### 3. Stochastic Optimization Formulation
```python
# Two-stage stochastic program
max Σ_s p_s [ Σ_c (r_c z_cs - π_c u_cs) ]  # Stage 2: recourse
s.t. Σ_c x_c ≤ T                          # Stage 1: capacity
     z_cs + u_cs = D_cs                   # Demand fulfillment
     z_cs ≤ κ_c x_c                       # Capacity constraints
     x_c ∈ Z+, z_cs, u_cs ≥ 0
```

### 4. Copula-based Simulation
- Marginal distributions: Normal for bulk, log-normal for TEU, Beta for empty share
- Dependence structure: Student-t copula (ν=6.2)
- Correlation matrix: ρ(bulk, TEU)=0.65, ρ(bulk, empty)=0.32, ρ(TEU, empty)=0.41

## Data Availability

The Python script contains a representative, anonymized dataset sufficient to reproduce all analyses and figures in the paper. This embedded dataset includes:

- **Temporal scope**: 2010-2018 (critical analysis period)
- **Spatial coverage**: All major rail lines and terminals
- **Commodity detail**: 8 commodity classes with tonnage and ton-km
- **Operational granularity**: 1,372 train-day observations

The complete dataset used for the original study contains commercially sensitive information and detailed infrastructure specifications. Researchers interested in the full dataset may request access by contacting the corresponding author, subject to confidentiality agreements and research collaboration terms.

## Contributing

We welcome contributions in these areas:

1. **Model extensions**: Additional uncertainty sources, multi-objective optimization
2. **Data integration**: Real-time AIS data, supply chain linkages
3. **Geographic expansion**: Application to other national rail networks
4. **Policy tools**: Interactive dashboards, scenario explorers

Please submit pull requests or open issues for discussion.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions about the methodology, data access requests, or collaboration opportunities:

- Dr. Yong-Jae Lee (이용재, 李龍在)
- Engineer (Technical Research Personnel) | TOBESOFT Ai Lab 
- Editorial Board Member, Decision Making and Analysis
- Ph.D. in Industrial and Management Engineering, Korea University
- E-mail: yj11021@korea.ac.kr (Korea Univ.) | yj11021@tobesoft.com (Tobesoft)
- ORCID: 0000-0002-7664-8001
