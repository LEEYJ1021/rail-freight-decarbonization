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

### Supplementary Experiments (Additional Analysis)
```bash
# Run supplementary robustness experiments independently
python rail_freight_decarbonization.py

# Output directory created:
# ./outputs/           # All 23 output files (PDFs + CSVs)
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

---

## Supplementary Robustness Experiments

The script `rail_freight_decarbonization.py` implements six additional experiments conducted to strengthen the robustness and empirical grounding of the core analyses. These experiments require only standard scientific Python libraries (`numpy`, `pandas`, `matplotlib`, `scipy`) and no optimization solvers. All outputs are written to `./outputs/`.

### Requirements for Supplementary Experiments

```bash
pip install numpy pandas matplotlib scipy
```

No optimization solvers or probabilistic programming libraries (pymc, pyomo) are required for these experiments. The Bayesian sensitivity analysis uses closed-form Normal–Normal conjugate posteriors.

### Experiment 1 — Bayesian Prior Sensitivity [P-EXP1]

Tests whether prior distribution choice materially affects the Value of the Stochastic Solution (VSS). Three configurations are compared: Baseline (σ₀ = 300 M tkm/yr), Weakly Informative (σ₀ = 600), and Tight/Dogmatic (σ₀ = 150). The likelihood standard deviation σ_L = 56.7 M tkm/yr is derived directly from the sample standard deviation of KORAIL 2015–2018 annual ton-km, ensuring the posterior is likelihood-dominated rather than prior-driven.

**Key result:** Max|ΔCI| = 4.8%, Max|ΔVSS| = 5.0% across all prior configurations and K-ETS price scenarios. Prior uncertainty does not propagate to policy conclusions.

**Outputs:** `exp1_prior_sensitivity_table.csv`, `exp1_vss_by_price_table.csv`, `exp1_ci_plot.pdf`, `exp1_vss_kETS_sensitivity.pdf`

### Experiment 2 — PM₁₀ Uncertainty Propagation & Tornado Diagram [P-EXP2]

One-at-a-time sensitivity analysis (tornado diagram) ranking five uncertainty sources by their swing on total social savings: K-ETS carbon price, modal shift rate, VSC_PM₁₀ valuation, PM₁₀ emission factor measurement bias (±8.35%), and rail emission factor. Uncertainty type (scenario range vs. empirical measurement uncertainty) is annotated for each parameter to clarify their conceptually distinct roles.

**Key result:** PM₁₀ EF measurement bias ranks 4th of 5 (swing = 0.098 B KRW, 6.67% of base case). It affects only post-hoc social valuation and does not enter the stochastic optimization objective. K-ETS price and modal shift rate are the dominant policy levers.

**Outputs:** `exp2_pm10_tornado_table.csv`, `exp2_tornado_diagram.pdf`

### Experiment 3 — Grid Carbon Intensity Dynamic Recalibration [P-EXP3]

Annual Korean electricity grid carbon intensity is reconstructed for 2010–2024 from KEPCO generation mix statistics and IPCC Tier 1 technology-specific emission factors. A static 2018 baseline emission factor approach is compared against this dynamic time series.

**Key result:** EF_electric declined from 17.88 (2010) to 15.95 gCO₂e/GTK (2024, −10.8%). A 2019 spike (+7.7% vs. 2018) is attributable to coal capacity additions (Samcheok, Shin-seocheon) and concurrent nuclear curtailment under Korea's post-Fukushima phase-down policy. A 2021 structural inflection point (−15.8% vs. 2018) reflects coal curtailment under the 3rd National Energy Plan — a feature invisible to static single-baseline approaches.

**Outputs:** `exp3_grid_ef_timeseries.csv`, `exp3_ef_kETS_crosstab.csv`, `exp3_grid_recalibration.pdf`

### Experiment 4 — Peak-to-Off-Peak Freight Shift [P-EXP4]

Quantitative simulation of CO₂e and block-time effects from shifting freight trains from peak to off-peak hours across four corridors (Gyeongbu, Chungbuk, Yeongdong, Jungang). Each corridor has an empirically derived passenger-conflict ceiling beyond which slot reallocation becomes infeasible without dedicated freight infrastructure. A cold-start CO₂ penalty (λ_start = 0.12, KEPCO operational survey) is modeled explicitly and shown to be dominated by peak congestion relief across all shift levels and corridors.

**Key result:** At 20% shift, Gyeongbu achieves 1,511 tCO₂e/yr reduction and 33.0 M KRW/yr in monetized K-ETS savings. Combined reduction across Chungbuk, Yeongdong, and Jungang (all below their passenger ceilings of 25–30%) is 3,769 tCO₂e/yr and 82.3 M KRW/yr. All 32 non-zero scenarios show net CO₂e reduction; direction is robust across cold-start sensitivity λ_start ∈ [0.05, 0.35].

**Outputs:** `exp4_peak_shift_table.csv`, `exp4_peak_shift_tradeoff.pdf`, `exp4_coldstart_sensitivity.pdf`

### Experiment 5 — Lambda Competitive Response Validation [P-EXP5]

Demonstrates that the stochastic planning advantage (VSS > 0) holds across the full competitive response parameter space λ ∈ [0.05, 0.95] for all four corridors and all five K-ETS price scenarios. The demand uncertainty parameter σ_D is set at 8% of baseline demand for each corridor — derived from the cross-corridor mean of KORAIL 2015–2018 year-on-year ton-km changes — and is explicitly decoupled from λ. This design isolates demand-side volume uncertainty from competitive demand level, so that VSS captures only the former.

**Key result:** VSS is strictly positive across the full λ range for all corridors and scenarios. VSS/E[RP] ratios span 6.41–10.77%, confirming a non-trivial stochastic planning premium regardless of competitive intensity.

**Outputs:** `exp5_lambda_table.csv`, `exp5_lambda_sensitivity_table.csv`, `exp5_lambda_validation.pdf`

### Experiment 6 — Empirical Lambda Calibration & Triangulation [P-EXP6]

Replaces expert-elicited corridor λ values with a three-stream triangulated empirical calibration:

- **EMP-1 (Natural Experiment / DiD):** The 2015 (+7.5%) and 2017 (+4.7%) KORAIL tariff increases serve as quasi-natural experiments. Using the price elasticity ε = −0.68 from the timetable regression, back-calculation yields a network-level λ_IVW = 0.365 [90% CI: 0.241, 0.489]. Bootstrap confidence intervals (N = 2,000) propagate elasticity uncertainty.
- **EMP-3 (Infrastructure Proxy Index):** A corridor-level IPI constructed from parallel expressway lane density, average route gradient, and haul distance reproduces the expert λ ranking with Pearson r = 0.994 and MAE = 0.068, providing structural validation of the competitive hierarchy.
- **EMP-4 (Out-of-Sample Directional Validation):** The corridor λ ranking correctly predicts the ordering of 2019 freight volume declines (r = 0.907, p = 0.093), providing directional out-of-sample confirmation.

Triangulated λ values are propagated through the full VSS model. VSS remains positive across all 16 corridor–K-ETS scenario pairs, with a maximum VSS sensitivity of 0.5% relative to expert-based values.

**Key result:** The corridor share-weighted average expert λ (0.467) falls within the EMP-1 network CI [0.241, 0.489]. IPI rank correlation (r = 0.994) and OOS directional check (r = 0.907) together confirm that expert-calibrated λ values are empirically grounded and that main conclusions are insensitive to λ uncertainty within the supported range.

**Outputs:** `exp6_lambda_triangulation.csv`, `exp6_vss_robustness.csv`, `exp6_lambda_empirical_final.csv`, `exp6_lambda_empirical_validation.pdf`

### Supplementary — K-ETS & KORAIL Validation

Reconstructs the full K-ETS allowance price history (2015–2024) with phase annotations and validates the AR(1)+COVID structural-break demand model against KORAIL actuals for 2019–2023.

**Key result:** MAPE (AR(1)+COVID specification) = 4.85% for 2019–2022; 2023 APE = 12.56%, reflecting accelerating secular freight demand contraction (−29.8% total ton-km from 2018 to 2023) beyond the model's structural scope. The 2018 base year represents the last structurally stable year of KORAIL's commodity-level panel prior to COVID-19 and K-ETS Phase III transition. The AR(1)+COVID specification achieves a 23.1% RMSE improvement over the unadjusted AR(1) baseline.

**Outputs:** `supp_kETS_timeseries.csv`, `supp_validation_korail.csv`, `supp_kETS_validation.pdf`

### Supplementary Output File Reference

| File | Type | Description |
|------|------|-------------|
| `exp1_prior_sensitivity_table.csv` | CSV | Full corridor × prior configuration sensitivity results |
| `exp1_vss_by_price_table.csv` | CSV | VSS by prior × K-ETS price scenario |
| `exp1_ci_plot.pdf` | Figure | 95% posterior credible intervals by corridor and prior |
| `exp1_vss_kETS_sensitivity.pdf` | Figure | Absolute and relative VSS vs K-ETS carbon price |
| `exp2_pm10_tornado_table.csv` | CSV | One-at-a-time sensitivity ranking with uncertainty type annotation |
| `exp2_tornado_diagram.pdf` | Figure | Tornado diagram with PM₁₀ EF bias rank annotation |
| `exp3_grid_ef_timeseries.csv` | CSV | Annual grid carbon intensity and generation mix, 2010–2024 |
| `exp3_ef_kETS_crosstab.csv` | CSV | CO₂ social savings matrix: EF scenario × K-ETS price |
| `exp3_grid_recalibration.pdf` | Figure | Grid CI time series, generation mix stacked bar, EF × K-ETS interaction surface |
| `exp4_peak_shift_table.csv` | CSV | Peak-shift scenarios with passenger-conflict ceiling flags per corridor |
| `exp4_peak_shift_tradeoff.pdf` | Figure | Block-time savings, CO₂e reduction, and monetized savings vs shift level |
| `exp4_coldstart_sensitivity.pdf` | Figure | Cold-start CO₂ penalty vs net saving across λ_start ∈ [0.05, 0.35] |
| `exp5_lambda_table.csv` | CSV | Corridor-specific VSS at baseline λ across K-ETS scenarios |
| `exp5_lambda_sensitivity_table.csv` | CSV | VSS vs λ continuous scan, λ ∈ [0.05, 0.95] |
| `exp5_lambda_validation.pdf` | Figure | VSS vs λ lines, corridor uncertainty ranges, VSS heatmap (λ × K-ETS) |
| `exp6_lambda_triangulation.csv` | CSV | IVW triangulated λ estimates with 90% CI per corridor |
| `exp6_vss_robustness.csv` | CSV | VSS robustness scan within empirical λ CI for all 16 corridor-scenario pairs |
| `exp6_lambda_empirical_final.csv` | CSV | Integrated λ validation summary across all empirical streams |
| `exp6_lambda_empirical_validation.pdf` | Figure | 4-panel: DiD natural experiment, IPI triangulation, VSS robustness, OOS validation |
| `supp_kETS_timeseries.csv` | CSV | K-ETS annual average, min, max prices 2015–2024 with phase annotations |
| `supp_validation_korail.csv` | CSV | KORAIL out-of-sample validation results 2019–2023 |
| `supp_kETS_validation.pdf` | Figure | K-ETS price history, illustrative scenario weights, KORAIL OOS validation |
| `Table0_integrated_summary.csv` | CSV | Experiment summary mapping key findings and outputs to analytical objectives |

### Key Quantitative Results from Supplementary Experiments

| Metric | Value |
|--------|-------|
| Max prior sensitivity on VSS (Exp 1) | \|ΔVSS\| ≤ 5.0% |
| PM₁₀ EF bias rank in tornado analysis (Exp 2) | 4th of 5 |
| Grid EF 2021 structural inflection (Exp 3) | −15.8% vs 2018 |
| 2024 grid EF vs 2018 baseline (Exp 3) | −17.1% (15.95 vs 19.25 gCO₂e/GTK) |
| CO₂e reduction at 20% peak shift, 3 corridors (Exp 4) | 3,769 tCO₂e/yr |
| Monetized savings at 20% peak shift, 3 corridors (Exp 4) | 82.3 M KRW/yr |
| VSS positive across λ ∈ [0.05, 0.95] (Exp 5) | 100% of corridor-scenario pairs |
| VSS/E[RP] range across corridors and scenarios (Exp 5) | 6.41–10.77% |
| Network λ from tariff natural experiment, EMP-1 (Exp 6) | 0.365 [90% CI: 0.241, 0.489] |
| IPI-expert λ rank correlation, EMP-3 (Exp 6) | r = 0.994 |
| OOS 2019 directional validation, EMP-4 (Exp 6) | r = 0.907 |
| Max ΔVSS using triangulated vs expert λ (Exp 6) | 0.5% |
| KORAIL demand model MAPE 2019–2022 (Supp) | 4.85% |
| AR(1)+COVID vs AR(1) RMSE improvement (Supp) | 23.1% |

### Reproducibility

All stochastic scenario generation uses corridor-specific fixed random seeds under the Common Random Numbers scheme, ensuring exact reproducibility:

```python
CORRIDOR_SEEDS = {"Gyeongbu": 1001, "Chungbuk": 1002, "Yeongdong": 1003, "Jungang": 1004}
```

Re-running `rail_freight_decarbonization.py` with no modifications produces identical numerical outputs.

---

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
