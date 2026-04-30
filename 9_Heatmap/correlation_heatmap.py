# Databricks notebook source
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

data_union = pd.read_csv('../2_Data_Union/ready_data_all_features.csv')
data_union

# COMMAND ----------

list_igr_ok = [
    'Alegre',
    'Belo Horizonte',
    'Campina Grande',
    'Campos dos Goytacazes',
    'Caratinga',
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

data = data_union[data_union['igr'].isin(list_igr_ok)]

data = data.drop(['access_count', 'count_physician'], axis=1)

data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Analysis

# COMMAND ----------

target = 'count_hospitalization'
results = {}

depara = {
    'count_hospitalization': 'Weekly hospitalizations',
    'rate_access_per_physician': 'Clinical search rate per physician',
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

for igr in data['igr'].unique():
    corr = data[data['igr'] == igr].corr(numeric_only=True)[[target]]
    corr.columns = [igr]  # rename the column to the name of the igr
    results[igr] = corr

# Correlation of all variables with the target variable
corr_with_target = data.corr(numeric_only=True)[[target]].sort_values(by=target, ascending=False)

corr_by_igr = pd.concat(results.values(), axis=1)

# Removing variables that start with 'interaction_'
corr_by_igr = corr_by_igr[~corr_by_igr.index.str.startswith('interaction_')]

# Putting 'count_hospitalization' as the first row
corr_by_igr = corr_by_igr.reindex(
    ['count_hospitalization'] + [idx for idx in corr_by_igr.index if idx != 'count_hospitalization']
)

corr_by_igr = corr_by_igr.rename(index=depara)

corr_by_igr

# COMMAND ----------

plt.figure(figsize=(8, 10))
sns.heatmap(
    corr_by_igr,
    #annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    cbar=True
)

plt.xlabel("Immediate Geographic Region", fontsize=12)
plt.ylabel("Variables", fontsize=12)

plt.savefig('heatmap.tif', 
            dpi=100, 
            bbox_inches='tight')

plt.show()