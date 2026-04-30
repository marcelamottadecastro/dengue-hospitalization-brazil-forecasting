# Databricks notebook source
# MAGIC %md
# MAGIC ### 📊 Statistical Evaluation of LSTM Model (Time-Series)
# MAGIC ####🎯 Objective
# MAGIC Validate the performance of the LSTM to rule out systematic delays (lag). This analysis provides statistical rigor for the article, moving validation beyond visual inspection.

# COMMAND ----------

!pip install openpyxl
!pip install statsmodels

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, ccf
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os

# COMMAND ----------

# Importing data

df_without_delay = pd.read_excel(f"../4_Model_Training/Without_Delay/best_lstm_without_delay_5555_70.xlsx")
df_with_delay = pd.read_excel(f"../4_Model_Training/With_Delay/best_lstm_with_delay_2108_70.xlsx")

# COMMAND ----------

# Correcting the data type/format of the columns with time series

def parse_np_array_str(s):
    # Match each inner array between [ and ] (handles multi-line wrapping)
    inner = re.findall(r'\[([^\[\]]+)\]', s)
    return np.array([[float(x) for x in row.split()] for row in inner])

def apply_parse_np_array_str(df):
    df['true_test'] = df['true_test'].apply(parse_np_array_str)
    df['pred_test'] = df['pred_test'].apply(parse_np_array_str)
    return df

df_without_delay = apply_parse_np_array_str(df_without_delay)
df_with_delay = apply_parse_np_array_str(df_with_delay)

# COMMAND ----------

depara_architecture = {
    '1 LSTM': 'Hospitalization',
    '2 LSTMs': 'Hospitalization + Clinical search',
    '5 LSTMs': 'Hospitalization + Climate',
    '6 LSTMs': 'Hospitalization + Clinical search + Climate',
    'SARIMA_UNIVARIADO': 'Sarimax - Hospitalization'
}

df_without_delay['network_architecture'] = df_without_delay['network_architecture'].map(depara_architecture)
df_with_delay['network_architecture'] = df_with_delay['network_architecture'].map(depara_architecture)

# COMMAND ----------

list_igr_ok = [
    'Alegre',
    'Belo Horizonte',
    'Campina Grande',
    'Campos dos Goytacazes',
    'Catalão',
    'Cruz Alta',
    'Distrito Federal',
    'Frederico Westphalen',
    'Ijuí',
    'Juiz de Fora',
    'Linhares',
    'Maringá',
    'Marília',
    'Oliveira',
    'Passo Fundo',
    'Passos',
    'Pirapora',
    'Porto Alegre',
    'Ribeirão Preto',
    'Rio de Janeiro',
    'Salvador',
    'Santa Cruz do Sul',
    'Santa Maria',
    'São Miguel do Oeste',
    'São Paulo',
    'Uberaba',
    'Uberlândia'
]

df_without_delay = df_without_delay[df_without_delay['igr'].isin(list_igr_ok)]
df_with_delay = df_with_delay[df_with_delay['igr'].isin(list_igr_ok)]

# COMMAND ----------

def create_metrics_table(df, save_path=None): # Added save_path as argument
    metrics_list = []

    group_cols = ['igr', 'network_architecture', 'features']

    for idx, row in df.iterrows():
        igr_value = row['igr']
        net_arch = row['network_architecture']
        feats = row['features']

        # 1. Each row is a specific series; compare corresponding lists for residuals
        y_true_lists = row['true_test']
        y_pred_lists = row['pred_test']

        # Ensure both are lists of lists and have same length
        n_preds = min(len(y_true_lists), len(y_pred_lists))
        residuals_all = []
        acf_vals_all = []
        ljungbox_stats = []
        dw_stats = []
        ccf_vals_all = []
        max_lags = []
        max_corrs = []
        theil_us = []
        mdas = []
        acf_significant = []
        acf_count = 0

        for i in range(n_preds):
            y_true = np.array(y_true_lists[i])
            y_pred = np.array(y_pred_lists[i])

            # Calculate residuals
            residuals = y_true - y_pred
            residuals_all.extend(residuals)

            # ACF
            acf_vals, confint = acf(residuals, nlags=8, alpha=0.05)
            acf_vals_all.append(acf_vals)
            
            significant = False
            for lag in range(1, len(acf_vals)):
                lower, upper = confint[lag]
                val = acf_vals[lag]
                if val < lower or val > upper:
                    significant = True
            
            acf_significant.append(significant)

            if significant:
                result_acf = 'There is significant autocorrelation'
                acf_count += 1
            else:
                result_acf = 'There is no significant autocorrelation'

            # Ljung-Box
            ljungbox_res = acorr_ljungbox(residuals, lags=min(8, len(residuals)-1), return_df=True)
            ljungbox_stats.append(ljungbox_res)

            # Cross-correlation (TLCC)
            ccf_vals = ccf(y_true, y_pred)[:9]
            ccf_vals_all.append(ccf_vals)
            max_lag = np.argmax(np.abs(ccf_vals))
            max_corr = ccf_vals[max_lag]
            max_lags.append(max_lag)
            max_corrs.append(max_corr)

        # Mean TLCC
        mean_corr = np.nanmean(ccf_vals_all, axis=0)
        std_corr = np.std(ccf_vals_all, axis=0)
        lags = np.arange(0, 9)
        tlcc_best_lag = lags[np.argmax(mean_corr)]

        # Aggregate metrics (mean across prediction lists)
        metrics_dict = {
            'igr': igr_value,
            'network_architecture': net_arch,
            "tlcc_best_lag": tlcc_best_lag,
            "result_acf": result_acf,
            "acf_count": acf_count
        }

        # ACF
        mean_acf_vals = np.mean(acf_vals_all, axis=0)
        for lag_idx, val in enumerate(mean_acf_vals):
            metrics_dict[f'ACF_Lag_{lag_idx}'] = round(val, 4)

        # Add mean Ljung-Box stats per lag
        n_lags = min(len(lb) for lb in ljungbox_stats)
        for lag in range(n_lags):
            lb_stats = [ljungbox_stats[j].iloc[lag]['lb_stat'] for j in range(len(ljungbox_stats))]
            lb_pvalues = [ljungbox_stats[j].iloc[lag]['lb_pvalue'] for j in range(len(ljungbox_stats))]
            metrics_dict[f'Ljung-Box Stat (lag {lag+1})'] = round(np.mean(lb_stats), 4)
            metrics_dict[f'Ljung-Box p-value (lag {lag+1})'] = round(np.mean(lb_pvalues), 4)

        metrics_list.append(metrics_dict)

    metrics_df = pd.DataFrame(metrics_list)
    
    # Add count_lb_pvalue_lt_0.05 before save
    ljung_cols = [col for col in metrics_df.columns if col.startswith('Ljung-Box p-value')]
    metrics_df['count_lb_pvalue_lt_0.05'] = metrics_df[ljung_cols].lt(0.05).sum(axis=1)
    
    if save_path is not None:
        metrics_df.to_excel(save_path, index=False)
    return metrics_df

# COMMAND ----------

df_analysis = create_metrics_table(df_without_delay, save_path="autocorrelation_analysis_without_delay.xlsx")
display(df_analysis)

# COMMAND ----------

# ACF
df_analysis['acf_count'].value_counts()

# COMMAND ----------

# Ljung-Box
df_analysis['count_lb_pvalue_lt_0.05'].value_counts()

# COMMAND ----------

# TLCC (Time-Lagged Cross-Correlation)
df_analysis['tlcc_best_lag'].value_counts()

# COMMAND ----------

df_analysis = create_metrics_table(df_with_delay, save_path="autocorrelation_analysis_with_delay.xlsx")
df_analysis

# COMMAND ----------

# ACF
df_analysis['acf_count'].value_counts()

# COMMAND ----------

# Ljung-Box
df_analysis['count_lb_pvalue_lt_0.05'].value_counts()

# COMMAND ----------

# TLCC (Time-Lagged Cross-Correlation)
df_analysis['tlcc_best_lag'].value_counts()