# Databricks notebook source
!pip install openpyxl

# COMMAND ----------

import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt

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

# COMMAND ----------

depara = {
    'count_hospitalization': 'Weekly hospitalizations',
    'rate_access_per_physician': 'Clinical search rate (per 10,000 physicians)',
    'precipitation_weekly_sum': 'Weekly precipitation (sum)',
    'rain_days': 'Rainy days',
    'sequential_rain': 'Consecutive rainy days',
    'temp_max_weekly_mean': 'Maximum temperature (weekly mean)',
    'temp_mean_weekly_mean': 'Mean temperature (weekly mean)',
    'temp_min_weekly_mean': 'Minimum temperature (weekly mean)',
    'humidity_weekly_mean': 'Relative humidity (weekly mean)',
    'lag_1_precipitation': 'Precipitation (1-week lag)',
    'lag_2_precipitation': 'Precipitation (2-week lag)',
    'lag_3_precipitation': 'Precipitation (3-week lag)',
    'lag_4_precipitation': 'Precipitation (4-week lag)',
    'lag_1_temp_mean': 'Mean temperature (1-week lag)',
    'lag_2_temp_mean': 'Mean temperature (2-week lag)',
    'lag_3_temp_mean': 'Mean temperature (3-week lag)',
    'lag_4_temp_mean': 'Mean temperature (4-week lag)',
    'lag_1_humidity': 'Humidity (1-week lag)',
    'lag_2_humidity': 'Humidity (2-week lag)',
    'lag_3_humidity': 'Humidity (3-week lag)',
    'lag_4_humidity': 'Humidity (4-week lag)',
    'change_temperature': 'Temperature change (week-over-week)',
    'percentile_temp_mean_weekly_mean': 'Mean temperature percentile (weekly)',
    'percentile_lag_temp_mean_weekly_mean': 'Mean temperature percentile (lagged)',
    'change_temperature_above75': 'Temperature change (>75th percentile)',
    'change_precipitation': 'Precipitation change (week-over-week)',
    'percentile_precipitation_weekly_sum': 'Precipitation percentile (weekly)',
    'percentile_lag_precipitation_weekly_sum': 'Precipitation percentile (lagged)',
    'change_precipitation_above75': 'Precipitation change (>75th percentile)',
    'change_humidity': 'Humidity change (week-over-week)',
    'percentile_humidity_weekly_mean': 'Humidity percentile (weekly)',
    'percentile_lag_humidity_weekly_mean': 'Humidity percentile (lagged)',
    'change_humidity_above75': 'Humidity change (>75th percentile)',
    'extreme_weather_temperature': 'Extreme weather: temperature',
    'extreme_weather_precipitation': 'Extreme weather: precipitation',
    'extreme_weather_humidity': 'Extreme weather: humidity',
    'season_summer': 'Season: summer',
    'season_autumn': 'Season: autumn',
    'season_winter': 'Season: winter',
    'season_spring': 'Season: spring',
    'temperature_category_quantile_01': 'Temperature category (Q1)',
    'temperature_category_quantile_02': 'Temperature category (Q2)',
    'temperature_category_quantile_03': 'Temperature category (Q3)',
    'temperature_category_quantile_04': 'Temperature category (Q4)',
    'precipitation_occurrence': 'Precipitation occurrence',
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Without delay

# COMMAND ----------

# Importing data

folder_path = '.'  # substitua pelo caminho da pasta se necessário
files = [f for f in os.listdir(folder_path) if 'without_' in f and f.endswith('.xlsx')]

dfs = [pd.read_excel(os.path.join(folder_path, file)) for file in files]

df_union_raw = pd.concat(dfs, ignore_index=True)

# COMMAND ----------

df_union = df_union_raw[df_union_raw['igr'].isin(list_igr_ok)]

# COMMAND ----------

df_union['mean_shap'] = df_union['mean_shap'].apply(lambda s: np.fromstring(s.strip('[]'), sep=' ').tolist() if isinstance(s, str) else s)
df_union['features'] = df_union['features'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

# COMMAND ----------

df_union['features'] = df_union['features'].apply(lambda feats: [depara.get(f, f) for f in feats])

# COMMAND ----------

igr_list = df_union['igr'].unique()
n_igr = len(igr_list)
n_cols = 3
n_rows = int(np.ceil(n_igr / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows))  # tamanho reduzido

axes = axes.flatten()

for idx, igr in enumerate(igr_list):
    igr_rows = df_union[df_union['igr'] == igr]
    mean_shap_matrix = np.vstack(igr_rows['mean_shap'].values)
    mean_shap_avg = mean_shap_matrix.mean(axis=0)
    mean_shap_std = mean_shap_matrix.std(axis=0)
    feature_names = igr_rows['features'].iloc[0]
    axes[idx].barh(feature_names, mean_shap_avg, xerr=mean_shap_std, color="blue", edgecolor="black", linewidth=0.7, alpha=0.7)
    axes[idx].set_title(igr, fontsize=16)  # fonte aumentada
    axes[idx].set_xlabel("SHAP mean", fontsize=12)  # fonte aumentada

# Remove unused axes
for ax in axes[n_igr:]:
    ax.axis('off')

plt.tight_layout()

# COMMAND ----------

fig.savefig("shap_without_delay.tif", format="tiff", dpi=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### With delay

# COMMAND ----------

# Importing data

folder_path = '.'  # substitua pelo caminho da pasta se necessário
files = [f for f in os.listdir(folder_path) if 'with_' in f and f.endswith('.xlsx')]

dfs = [pd.read_excel(os.path.join(folder_path, file)) for file in files]

df_union_raw = pd.concat(dfs, ignore_index=True)

# COMMAND ----------

df_union = df_union_raw[df_union_raw['igr'].isin(list_igr_ok)]

# COMMAND ----------

df_union['mean_shap'] = df_union['mean_shap'].apply(lambda s: np.fromstring(s.strip('[]'), sep=' ').tolist() if isinstance(s, str) else s)
df_union['features'] = df_union['features'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

# COMMAND ----------

df_union

# COMMAND ----------

df_union['features'] = df_union['features'].apply(lambda feats: [depara.get(f, f) for f in feats])

# COMMAND ----------

igr_list = df_union['igr'].unique()
n_igr = len(igr_list)
n_cols = 3
n_rows = int(np.ceil(n_igr / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows))  # tamanho reduzido

axes = axes.flatten()

for idx, igr in enumerate(igr_list):
    igr_rows = df_union[df_union['igr'] == igr]
    mean_shap_matrix = np.vstack(igr_rows['mean_shap'].values)
    mean_shap_avg = mean_shap_matrix.mean(axis=0)
    mean_shap_std = mean_shap_matrix.std(axis=0)
    feature_names = igr_rows['features'].iloc[0]
    n = len(feature_names)
    axes[idx].barh(feature_names, mean_shap_avg[:n], xerr=mean_shap_std[:n], color="blue", edgecolor="black", linewidth=0.7, alpha=0.7)
    axes[idx].set_title(igr, fontsize=16)  # fonte aumentada
    axes[idx].set_xlabel("SHAP mean", fontsize=12)  # fonte aumentada

# Remove unused axes
for ax in axes[n_igr:]:
    ax.axis('off')

plt.tight_layout()

# COMMAND ----------

fig.savefig("shap_with_delay.tif", format="tiff", dpi=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RJ

# COMMAND ----------

# Importing data without delay

folder_path = '.'  # substitua pelo caminho da pasta se necessário
files = [f for f in os.listdir(folder_path) if 'without_' in f and f.endswith('.xlsx')]

dfs = [pd.read_excel(os.path.join(folder_path, file)) for file in files]

df_union_raw = pd.concat(dfs, ignore_index=True)

single_without_delay = df_union_raw[df_union_raw['igr'].isin(['Rio de Janeiro'])]

single_without_delay['subtitle'] = 'Ideal-data model'

# COMMAND ----------

# Importing data with delay

folder_path = '.'  # substitua pelo caminho da pasta se necessário
files = [f for f in os.listdir(folder_path) if 'with_' in f and f.endswith('.xlsx')]

dfs = [pd.read_excel(os.path.join(folder_path, file)) for file in files]

df_union_raw = pd.concat(dfs, ignore_index=True)

single_with_delay = df_union_raw[df_union_raw['igr'].isin(['Rio de Janeiro'])]

single_with_delay['subtitle'] = 'Real-world model'

# COMMAND ----------

single_union = pd.concat([single_without_delay, single_with_delay], ignore_index=True)
single_union

# COMMAND ----------

single_union['mean_shap'] = single_union['mean_shap'].apply(lambda s: np.fromstring(s.strip('[]'), sep=' ').tolist() if isinstance(s, str) else s)
single_union['features'] = single_union['features'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

# COMMAND ----------

single_union['features'] = single_union['features'].apply(lambda feats: [depara.get(f, f) for f in feats])

# COMMAND ----------

single_union

# COMMAND ----------

subtitle_list = single_union['subtitle'].unique()
n_subtitle = len(subtitle_list)
n_cols = 2
n_rows = 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))  # tamanho reduzido

igr_name = single_union['igr'].iloc[0]
fig.suptitle(igr_name, fontsize=18, y=0.96, ha='center')  # título centralizado mais próximo do gráfico

axes = axes.flatten()

for idx, subtitle in enumerate(subtitle_list):
    subtitle_rows = single_union[single_union['subtitle'] == subtitle]
    mean_shap_matrix = np.vstack(subtitle_rows['mean_shap'].values)
    mean_shap_avg = mean_shap_matrix.mean(axis=0)
    mean_shap_std = mean_shap_matrix.std(axis=0)
    feature_names = subtitle_rows['features'].iloc[0]
    axes[idx].barh(feature_names, mean_shap_avg, xerr=mean_shap_std, color="blue", edgecolor="black", linewidth=0.7, alpha=0.7)
    axes[idx].set_title(subtitle, fontsize=16)  # fonte aumentada
    axes[idx].set_xlabel("SHAP mean", fontsize=12)  # fonte aumentada

# Remove unused axes
for ax in axes[n_subtitle:]:
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("shap_rj.tif", format="tiff", dpi=100)