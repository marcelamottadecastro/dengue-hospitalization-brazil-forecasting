# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

df = pd.read_csv('Data Climate/Data Climate Processed/processed_validated_stations_filled_gap30.csv')
df['Data Medicao'] = pd.to_datetime(df['Data Medicao'])
df

# COMMAND ----------

igrs = df['igr'].unique()
igrs

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1 Lag

# COMMAND ----------

# MAGIC %md
# MAGIC - lag_1_semana_precipitacao (soma da semana anterior para precipitacao)
# MAGIC - lag_2_semanas_precipitacao (soma da semana duas semanas atrás para precipitacao)
# MAGIC - lag_3_semanas_precipitacao (soma da semana três semanas atrás para precipitacao)
# MAGIC - lag_4s_semanas_precipitacao (soma da semana quatro semanas atrás para precipitacao)
# MAGIC - lag_1_semana_umidade (média da semana anterior para umidade)
# MAGIC - lag_2_semanas_umidade (média da semana duas semanas atrás para umidade)
# MAGIC - lag_3_semanas_umidade (média da semana três semanas atrás para umidade)
# MAGIC - lag_4_semanas_umidade (média da semana quatro semanas atrás para umidade)
# MAGIC - lag_1_semana_temperatura (média da semana anterior para temperatura)
# MAGIC - lag_2_semanas_temperatura (média da semana duas semanas atrás para temperatura)
# MAGIC - lag_3_semanas_temperatura (média da semana três semanas atrás para temperatura)
# MAGIC - lag_4_semanas_temperatura (média da semana quatro semanas atrás para temperatura)

# COMMAND ----------

def create_lag(column, reference):
    global df, igrs

    for lag in [1, 2, 3, 4]:
        new_column_name = f'lag_{lag}_{reference}'
        
        # Initialize the column with NaNs only once
        if new_column_name not in df.columns:
            df[new_column_name] = np.nan

        for igr_i in igrs:
            mask = df['igr'] == igr_i
            df.loc[mask, new_column_name] = df.loc[mask, column].shift(lag)

# COMMAND ----------

create_lag(column='precipitation_weekly_sum', reference='precipitation')
create_lag(column='temp_mean_weekly_mean', reference='temp_mean')
create_lag(column='humidity_weekly_mean', reference='humidity')

# COMMAND ----------

# Check
df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2 Change Between Weeks

# COMMAND ----------

# MAGIC %md
# MAGIC - change_temperature (0 or 1 - percentile criterion between 0.75 and 0.85, when the previous week was below 0.75)
# MAGIC - change_precipitation (0 or 1 - criterion above 10mm accumulated rainfall)
# MAGIC - change_humidity (0 or 1 - percentile criterion between 0.75 and 0.85, when the previous week was below 0.75)

# COMMAND ----------

def is_between(a, x, b):
    return min(a, b) <= x <= max(a, b)

# COMMAND ----------

def create_variable_change(column, reference):
    global df, igrs

    new_column_name = f'change_{reference}'

    auxiliary_percentile_column = f'percentile_{column}'
    auxiliary_percentile_lag_column = f'percentile_lag_{column}'

        
    # Initialize the column with NaNs only once
    if new_column_name not in df.columns:
        df[new_column_name] = np.nan
        df[auxiliary_percentile_column] = np.nan

    for igr_i in igrs:
        mask = df['igr'] == igr_i

        # Calculate the percentile
        df.loc[mask, auxiliary_percentile_column] = df.loc[mask, column].rank(pct=True)
        df.loc[mask, auxiliary_percentile_lag_column] = df.loc[mask, auxiliary_percentile_column].shift(1)

    # Rules
    df[f'change_{reference}'] = df.apply(lambda row: 1 if is_between(0.75, row[auxiliary_percentile_column], 0.85) & (row[auxiliary_percentile_lag_column] < 0.75)
                                      else 0
                                      , axis=1)
    
    df[f'change_{reference}_above75'] = df.apply(lambda row: 1 if (row[auxiliary_percentile_column] >= 0.75) & (row[auxiliary_percentile_lag_column] < 0.75)
                                      else 0
                                      , axis=1)

# COMMAND ----------

create_variable_change(column='temp_mean_weekly_mean', reference='temperature')
create_variable_change(column='precipitation_weekly_sum', reference='precipitation')
create_variable_change(column='humidity_weekly_mean', reference='humidity')

# COMMAND ----------

# Check
df

# COMMAND ----------

# Check
print(df[['change_temperature', 'change_temperature_above75']].value_counts())
print(df[['change_precipitation', 'change_precipitation_above75']].value_counts())
print(df[['change_humidity', 'change_humidity_above75']].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.3 Interaction

# COMMAND ----------

# MAGIC %md
# MAGIC - interaction_temperature_precipitation (0 or 1 - when change_temperature and change_precipitation were equal to 1, but humidity was 0)
# MAGIC - interaction_temperature_humidity (0 or 1 - when change_temperature and change_humidity were equal to 1, but precipitation was 0)
# MAGIC - interaction_precipitation_humidity (0 or 1 - when change_precipitation and change_humidity were equal to 1, but temperature was 0)
# MAGIC - interaction_temperature_precipitation_humidity (0 or 1 - when change_temperature, interaction_temperature_humidity and change_humidity were equal to 1)

# COMMAND ----------

df['interaction_temperature_precipitation'] = df.apply(lambda row: 1 if (row['change_temperature'] == 1) & (row['change_precipitation'] == 1) & (row['change_humidity'] == 0)
        else 0
        , axis=1)

df['interaction_temperature_humidity'] = df.apply(lambda row: 1 if (row['change_temperature'] == 1) & (row['change_precipitation'] == 0) & (row['change_humidity'] == 1)
        else 0
        , axis=1)

df['interaction_precipitation_humidity'] = df.apply(lambda row: 1 if (row['change_temperature'] == 0) & (row['change_precipitation'] == 1) & (row['change_humidity'] == 1)
        else 0
        , axis=1)

df['interaction_temperature_precipitation_humidity'] = df.apply(lambda row: 1 if (row['change_temperature'] == 1) & (row['change_precipitation'] == 1) & (row['change_humidity'] == 1)
        else 0
        , axis=1)

# COMMAND ----------

# Check
df[['change_temperature', 'change_precipitation', 'change_humidity', 'interaction_temperature_precipitation_humidity']].value_counts()

# COMMAND ----------

df['interaction_temperature_precipitation_above75'] = df.apply(lambda row: 1 if (row['change_temperature_above75'] == 1) & (row['change_precipitation_above75'] == 1) & (row['change_humidity_above75'] == 0)
        else 0
        , axis=1)

df['interaction_temperature_humidity_above75'] = df.apply(lambda row: 1 if (row['change_temperature_above75'] == 1) & (row['change_precipitation_above75'] == 0) & (row['change_humidity_above75'] == 1)
        else 0
        , axis=1)

df['interaction_precipitation_humidity_above75'] = df.apply(lambda row: 1 if (row['change_temperature_above75'] == 0) & (row['change_precipitation_above75'] == 1) & (row['change_humidity_above75'] == 1)
        else 0
        , axis=1)

df['interaction_temperature_precipitation_humidity_above75'] = df.apply(lambda row: 1 if (row['change_temperature_above75'] == 1) & (row['change_precipitation_above75'] == 1) & (row['change_humidity_above75'] == 1)
        else 0
        , axis=1)

# COMMAND ----------

# Check
df[['change_temperature_above75', 'change_precipitation_above75', 'change_humidity_above75', 'interaction_temperature_precipitation_humidity_above75']].value_counts(dropna=False, sort=False)

# COMMAND ----------

# Check
df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.4 Extreme Weather

# COMMAND ----------

# MAGIC %md
# MAGIC - extreme_weather_temperature (0 or 1 - percentile above 0.95)
# MAGIC - extreme_weather_rain (0 or 1 - percentile above 0.95)
# MAGIC - extreme_weather_humidity (0 or 1 - percentile above 0.95)

# COMMAND ----------

def create_extreme_weather(column, reference):
    global df

    new_column_name = f'extreme_weather_{reference}'

    auxiliary_percentile_column = f'percentile_{column}'
    
    df[new_column_name] = df.apply(lambda row: 1 if (row[auxiliary_percentile_column] > 0.95)
        else 0
        , axis=1)

# COMMAND ----------

create_extreme_weather(column='temp_mean_weekly_mean', reference='temperature')
create_extreme_weather(column='precipitation_weekly_sum', reference='precipitation')
create_extreme_weather(column='humidity_weekly_mean', reference='humidity')

# COMMAND ----------

# Check
df

# COMMAND ----------

#Check
print(df[df['percentile_temp_mean_weekly_mean'] > 0.95].shape[0])
print(df[df['percentile_precipitation_weekly_sum'] > 0.95].shape[0])
print(df[df['percentile_humidity_weekly_mean'] > 0.95].shape[0])

# COMMAND ----------

#Check
print(df['extreme_weather_temperature'].value_counts(dropna=False, sort=False))
print(df['extreme_weather_precipitation'].value_counts(dropna=False, sort=False))
print(df['extreme_weather_humidity'].value_counts(dropna=False, sort=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.5 Season

# COMMAND ----------

# MAGIC %md
# MAGIC - season_summer (0 or 1 - summer criterion)
# MAGIC - season_spring (0 or 1 - spring criterion)
# MAGIC - season_autumn (0 or 1 - autumn criterion)
# MAGIC - season_winter (0 or 1 - winter criterion)

# COMMAND ----------

def create_summer_season(date):
    year = date.year
    if pd.Timestamp(f'{year}-12-21') <= date or date < pd.Timestamp(f'{year}-03-21'):
        return 1
    else:
        return 0
    
def create_autumn_season(date):
    year = date.year
    if pd.Timestamp(f'{year}-03-21') <= date < pd.Timestamp(f'{year}-06-21'):
        return 1
    else:
        return 0
    
def create_winter_season(date):
    year = date.year
    if pd.Timestamp(f'{year}-06-21') <= date < pd.Timestamp(f'{year}-09-23'):
        return 1
    else:
        return 0
    
def create_spring_season(date):
    year = date.year
    if pd.Timestamp(f'{year}-09-23') <= date < pd.Timestamp(f'{year}-12-21'):
        return 1
    else:
        return 0

# COMMAND ----------

df['season_summer'] = df['Data Medicao'].apply(create_summer_season)
df['season_autumn'] = df['Data Medicao'].apply(create_autumn_season)
df['season_winter'] = df['Data Medicao'].apply(create_winter_season)
df['season_spring'] = df['Data Medicao'].apply(create_spring_season)
df

# COMMAND ----------

# Check
df[['season_summer', 'season_autumn', 'season_winter', 'season_spring']].value_counts(dropna=False, sort=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.5 Temperature Category

# COMMAND ----------

# MAGIC %md
# MAGIC - temperature_category_quantile_01 (0 or 1 - temperatures in percentile 0% - 25%)
# MAGIC - temperature_category_quantile_02 (0 or 1 - temperatures in percentile 26% - 50%)
# MAGIC - temperature_category_quantile_03 (0 or 1 - temperatures in percentile 51% - 75%)
# MAGIC - temperature_category_quantile_04 (0 or 1 - temperatures in percentile 76% - 100%)

# COMMAND ----------

df['temperature_category_quantile_01'] = df['percentile_temp_mean_weekly_mean'].apply(lambda x: 1 if x <= 0.25 else 0)
df['temperature_category_quantile_02'] = df['percentile_temp_mean_weekly_mean'].apply(lambda x: 1 if 0.25 < x <= 0.5 else 0)
df['temperature_category_quantile_03'] = df['percentile_temp_mean_weekly_mean'].apply(lambda x: 1 if 0.5 < x <= 0.75 else 0)
df['temperature_category_quantile_04'] = df['percentile_temp_mean_weekly_mean'].apply(lambda x: 1 if x > 0.75 else 0)

# COMMAND ----------

#Check
print(df[['temperature_category_quantile_01', 'percentile_temp_mean_weekly_mean']].value_counts(dropna=False, sort=False))
print(df[['temperature_category_quantile_02', 'percentile_temp_mean_weekly_mean']].value_counts(dropna=False, sort=False))
print(df[['temperature_category_quantile_03', 'percentile_temp_mean_weekly_mean']].value_counts(dropna=False, sort=False))
print(df[['temperature_category_quantile_04', 'percentile_temp_mean_weekly_mean']].value_counts(dropna=False, sort=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.6 Precipitation Occurrence

# COMMAND ----------

# MAGIC
# MAGIC
# MAGIC %md
# MAGIC - precipitation_occurrence (0 or 1 - criterion: rainfall occurred)

# COMMAND ----------

df['precipitation_occurrence'] = (df['precipitation_weekly_sum'] > 0).astype(int)

# COMMAND ----------

#Check
df[['precipitation_weekly_sum', 'precipitation_occurrence']]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.7 Rainfall

# COMMAND ----------

# MAGIC %md
# MAGIC - sequential_rain (sum of the number of sequential rainy days)
# MAGIC - rain_days (sum of the number of rainy days)

# COMMAND ----------

df.rename(columns={'precipitation_days_with_rain': 'rain_days', 'precipitation_consecutive_days_with_rain': 'sequential_rain'}, inplace=True)
df

# COMMAND ----------

# Converting date to week format, as used in the other hospitalization and WB access tables
df.rename(columns={'Data Medicao': 'date'}, inplace=True)
df['date'] = df['date'].dt.strftime('%Y-%U')
display(df)

# COMMAND ----------

# Export data
df.to_csv('Data Climate/Data Climate Processed/ready_validated_stations_filled_gap30_all_features.csv', index=False)