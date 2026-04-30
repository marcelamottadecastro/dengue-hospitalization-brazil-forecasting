# Climate, Clinical Search Behavior, and Dengue Hospitalizations in Brazil

## Overview

This repository contains the code and data processing pipeline for a study investigating the association between climatic variables, clinical search patterns on a medical decision-support platform (Afya Whitebook), and dengue-related hospitalizations across Brazilian Immediate Geographic Regions (IGRs).

The analysis uses weekly time series data (2021–2024) for 27 IGRs and compares deep learning (LSTM) and statistical (SARIMAX) models under different input configurations to assess whether climate and clinical search data improve hospitalization forecasting.

## Relationship with the Article

This codebase supports the experiments described in the associated peer-reviewed article. The pipeline reproduces all reported results, including:

* Feature selection via LSTM-based univariate screening per IGR and climate category (temperature, precipitation, humidity, season)
* Model training under four input configurations: (1) hospitalization only, (2) hospitalization + clinical search, (3) hospitalization + climate, and (4) hospitalization + clinical search + climate
* Evaluation with and without an 8-week delay on the target variable
* Triplicate experiments with different random seeds for statistical robustness
* SHAP-based interpretability analysis
* Residual diagnostics (autocorrelation, Ljung-Box test, time-lagged cross-correlation)
* Figures and tables presented in the manuscript (boxplots, heatmaps, time series plots)

## Data Sources

| Source | Description | Granularity |
| --- | --- | --- |
| SIH/DATASUS | Dengue hospitalization counts (Sistema de Informações Hospitalares) | Weekly, by IGR |
| Afya Whitebook (WB) | Clinical search access counts and active physician counts on the medical platform | Weekly, by IGR |
| INMET | Meteorological station data: temperature, precipitation, relative humidity | Weekly, by IGR |
| IBGE | Geographic mapping of municipalities to Immediate Geographic Regions (IGRs) | Static reference |

**Note:** Raw Whitebook data derives from proprietary Afya databases. The publicly shared version uses pre-processed, aggregated CSVs.

## Reproducing Experiments

### Prerequisites

* Python 3.10+
* Databricks workspace (notebooks use `%pip install` and `%restart_python` commands)
* Alternatively, the notebooks can be adapted to run in any Jupyter-compatible environment

### Execution Order

Run the pipeline sequentially following the numbered folder structure:

1. **`1_Data/`** — Prepare raw data sources
   * `Data Climate/1_unificating_raw_files` → unify INMET station files
   * `Data Climate/2_filling_gaps` → fill missing values (max 30-day gap threshold)
   * `Data Climate/3_create_features_lag` → engineer lagged and derived climate features
   * `Data WB Access/processing_wb_access` → process Whitebook access data
   * `Data WB Users/get_wb_users_year` → extract yearly active physician counts
   * `Data SIH Hospitalization/processing_sih_hospitalization` → process SIH hospitalization data
2. **`2_Data_Union/union_data`** — Merge all data sources by IGR and epidemiological week → produces `ready_data_all_features.csv`
3. **`3_Feature_Selection/model_feature_selection`** — Run LSTM-based feature selection per IGR; selects the best climate variable per category (temperature, precipitation, humidity, season) using RMSE → produces `metricas_comparacao_2025_10_17.xlsx`
4. **`4_Model_Training/`** — Train models in four configurations:
   * `Without_Delay/Modelo LSTM - Sem delay` — LSTM without target delay
   * `With_Delay/Modelo LSTM - Com delay` — LSTM with 8-week target delay
   * `Sarimax_Without_Delay/Modelo Sarimax - Seleção de variáveis` — SARIMAX without delay
   * `Sarimax_With_Delay/Modelo Sarimax - With Delay` — SARIMAX with 8-week delay
   * Each notebook runs 3 replications with different random seeds
5. **`5_Analysis_Autocorrelation_Residuos/analysis_residuos`** — Residual analysis: ACF, Ljung-Box test, TLCC, Naive comparison
6. **`6_Boxpot_And_Table/boxplot`** — Generate RMSE boxplots and statistical comparison tables (t-test)
7. **`7_Create_Timeseries_Graph/time_series_graph`** — Generate time series visualization plots
8. **`8_SHAP/`** — SHAP interpretability analysis:
   * `SHAP` — Compute SHAP values for trained LSTM models
   * `Visualization_SHAP` — Generate SHAP summary figures
9. **`9_Heatmap/correlation_heatmap`** — Pearson correlation heatmap across all IGRs

### Key Hyperparameters

| Parameter | Value |
| --- | --- |
| Sequence length (lookback) | 24 weeks |
| Hidden size | {20, 40} |
| Number of LSTM layers | {1, 2, 3, 4} |
| Output size (forecast horizon) | 8 weeks |
| Delay size | 8 weeks (when applicable) |
| Learning rate | 0.01 |
| Epochs | 300 (training) / 200 (feature selection) |
| Train/test split | 70% / 30% |
| Replications | 3 (different random seeds per configuration) |

### Model Architectures

| Architecture | Input Variables |
| --- | --- |
| 1 LSTM | Hospitalization (univariate baseline) |
| 2 LSTMs | Hospitalization + Clinical search rate |
| 5 LSTMs | Hospitalization + Climate variables (4 selected per IGR) |
| 6 LSTMs | Hospitalization + Clinical search rate + Climate variables |
| SARIMAX | Statistical baseline with exogenous variables |

## Folder Structure

```
clima_saude/
├── README.md
├── 1_Data/                              # Raw data and preprocessing
│   ├── Data Climate/                    # INMET meteorological data
│   │   ├── 1_unificating_raw_files      # Unify station CSV files
│   │   ├── 2_filling_gaps               # Gap-filling (30-day threshold)
│   │   ├── 3_create_features_lag        # Lag and derived features
│   │   └── Data Climate/               # Processed CSVs
│   ├── Data WB Access/                  # Whitebook clinical search data
│   │   ├── processing_wb_access         # Processing notebook
│   │   └── Data WB Access/             # Processed CSVs
│   ├── Data WB Users/                   # Whitebook active physicians
│   │   ├── get_wb_users_year            # Yearly physician extraction
│   │   └── Data WB Users/             # Output CSVs
│   ├── Data SIH Hospitalization/        # DATASUS hospitalization data
│   │   ├── processing_sih_hospitalization
│   │   └── Data SIH Hospitalization/   # Processed CSVs
│   ├── Data IBGE/                       # Geographic reference data
│   └── README                           # Data provenance notes
├── 2_Data_Union/                        # Data integration
│   ├── union_data                       # Merge notebook
│   └── ready_data_all_features.csv      # Unified dataset
├── 3_Feature_Selection/                 # Climate feature selection
│   ├── model_feature_selection          # LSTM-based selection
│   ├── metricas_comparacao_*.xlsx       # Selection results
│   └── best_data_feature_selction.xlsx  # Best features per IGR
├── 4_Model_Training/                    # Model experiments
│   ├── With_Delay/                      # LSTM with 8-week delay
│   ├── Without_Delay/                   # LSTM without delay
│   ├── Sarimax_With_Delay/             # SARIMAX with delay
│   └── Sarimax_Without_Delay/          # SARIMAX without delay
├── 5_Analysis_Autocorrelation_Residuos/ # Residual diagnostics
│   └── analysis_residuos               # ACF, Ljung-Box, TLCC
├── 6_Boxpot_And_Table/                  # Statistical comparison
│   ├── boxplot                          # Boxplot generation
│   ├── result_*.xlsx                    # Model results
│   ├── testT_*.xlsx                     # T-test results
│   └── boxplot_rmse_triplicata*.tif     # Output figures
├── 7_Create_Timeseries_Graph/           # Time series visualization
│   ├── time_series_graph
│   └── Series_Temporais.tif
├── 8_SHAP/                              # Model interpretability
│   ├── SHAP                             # SHAP value computation
│   ├── Visualization_SHAP              # SHAP figure generation
│   ├── SHAP_*_delay_*.xlsx             # SHAP values per seed
│   └── shap_*.tif                      # Output figures
└── 9_Heatmap/                           # Correlation analysis
    ├── correlation_heatmap
    └── heatmap.tif
```

## Dependencies

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
torch>=2.0               # PyTorch (LSTM models)
scikit-learn>=1.2         # MinMaxScaler, regression metrics
statsmodels>=0.14         # SARIMAX, ACF, Ljung-Box test
shap>=0.42                # SHAP interpretability
openpyxl>=3.1             # Excel I/O
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn statsmodels shap openpyxl
```

## Geographic Scope

The analysis covers 28 Immediate Geographic Regions (IGRs) across Brazil:

Alegre, Belo Horizonte, Campina Grande, Campos dos Goytacazes, Catalão, Cruz Alta, Distrito Federal, Frederico Westphalen, Ijuí, Juiz de Fora, Linhares, Maringá, Marília, Oliveira, Passo Fundo, Passos, Pirapora, Porto Alegre, Ribeirão Preto, Rio de Janeiro, Salvador, Santa Cruz do Sul, Santa Maria, São Miguel do Oeste, São Paulo, Uberaba, Uberlândia.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{clima_saude_2025,
  title   = {[Article Title]},
  author  = {[Authors]},
  journal = {[Journal Name]},
  year    = {2025},
  doi     = {[DOI]}
}
```

> **Note:** Replace the placeholder fields with the final publication details once the article is published.

## License

[Specify license here — e.g., MIT, Apache 2.0, or CC BY 4.0 for academic use.]

## Contact

Marcela Motta — marcela.motta@afya.com.br
