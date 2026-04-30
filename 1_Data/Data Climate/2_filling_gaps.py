# Databricks notebook source
# MAGIC %md
# MAGIC ### Notebook steps:
# MAGIC
# MAGIC - Filter the date range of interest (2021 to 2024)
# MAGIC - Filter stations with gaps of missing data shorter than 30 consecutive days
# MAGIC - Aggregate data by week
# MAGIC - Fill missing weekly data with moving average
# MAGIC - Include micro-regions
# MAGIC - Deduplicate micro-regions with more than one station, keeping the station with the fewest missing values

# COMMAND ----------

# MAGIC %pip install geopandas
# MAGIC %pip install pykrige
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from pykrige.ok import OrdinaryKriging

# COMMAND ----------

pd.set_option("display.float_format", "{:.1f}".format)
pd.set_option("display.max_columns", None)

# COMMAND ----------

df = pd.read_csv('Data Climate/Data Climate Processed/processed_all_stations_with_measurements_A2000_2024.csv', encoding='utf-8', low_memory=False)
df

# COMMAND ----------

df['Data Medicao'] = pd.to_datetime(df['Data Medicao'], errors='coerce')

# COMMAND ----------

# Getting the minimum and maximum date from the dataframe
min_date = df['Data Medicao'].min()
max_date = df['Data Medicao'].max()

# Creating a complete daily date range
all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

# Getting all unique stations
all_stations = df['Station Code'].unique()

# Creating all possible combinations (station × date)
combinations = pd.MultiIndex.from_product([all_dates, all_stations], names=['Data Medicao', 'Station Code'])
df_combinations = pd.DataFrame(index=combinations).reset_index().sort_values(by=['Station Code', 'Data Medicao'])

# Merging with the original dataframe
df = pd.merge(df_combinations, df, on=['Data Medicao', 'Station Code'], how="left")
df

# COMMAND ----------

df['Station'] = df.groupby('Station Code')['Station'].ffill().bfill()
df['Latitude'] = df.groupby('Station Code')['Latitude'].ffill().bfill()
df['Longitude'] = df.groupby('Station Code')['Longitude'].ffill().bfill()
df['Start Date'] = df.groupby('Station Code')['Start Date'].ffill().bfill()
df['End Date'] = df.groupby('Station Code')['End Date'].ffill().bfill()
df

# COMMAND ----------

## Gap climate features analysis

def filter_stations_without_30_day_gaps(df, climate_columns):
    """Filters stations based on Station Code, removing those with gaps larger than 15 days."""

    # Helper function to calculate the largest consecutive NaN gap for each variable
    def calculate_max_gap(df_station, colunas):
        max_gap = 0
        for column in colunas:
            series = df_station[column]
            gaps = []
            gap = 0
            for value in series.isnull():
                if value:
                    gap += 1
                else:
                    if gap > 0:
                        gaps.append(gap)
                    gap = 0
            if gap > 0:
                gaps.append(gap)
            if gaps:
                max_gap = max(max_gap, max(gaps))
        return max_gap

    # Calculate maximum gaps grouped by Station Code
    max_gaps = df.groupby('Station Code').apply(calculate_max_gap, colunas=climate_columns)

    # Identify stations with gaps larger than 30 days
    stations_with_30_day_gaps = max_gaps[max_gaps > 30].index

    # Filter the DataFrame
    df_filtered = df[~df['Station Code'].isin(stations_with_30_day_gaps)]

    return df_filtered, max_gaps

# List of climate variables
climate_columns = [
    'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)',
    'TEMPERATURA MAXIMA, DIARIA (AUT)(°C)',
    'TEMPERATURA MEDIA, DIARIA (AUT)(°C)',
    'TEMPERATURA MINIMA, DIARIA (AUT)(°C)',
    'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'
]

# Apply filtering
df_period_gaps, max_gaps = filter_stations_without_30_day_gaps(df, climate_columns)

# Display the number of stations before and after filtering
print(f"Number of stations before filtering: {df['Station Code'].nunique()}")
print(f"Number of stations after filtering: {df_period_gaps['Station Code'].nunique()}")

# COMMAND ----------

# Histogram of maximum gaps
max_gaps.hist(bins=100)

# COMMAND ----------

## Adjusting variables so they can be converted to numeric

def adjust_variables(column_name):
    global df_period_gaps
    # Converting to string
    df_period_gaps[column_name] = df_period_gaps[column_name].astype(str)
    
    # Adding '0' before numbers with a comma
    df_period_gaps[column_name] = df_period_gaps[column_name].apply(lambda x: f'{x}' if x.startswith('-') else 
                                                                     f'0{x}' if ',' in x else 
                                                                     x
                                                                    )
    
    # Replacing ',' with '.'
    df_period_gaps[column_name] = df_period_gaps[column_name].str.replace(',', '.')

    # Converting to float
    df_period_gaps[column_name] = df_period_gaps[column_name].astype(float)

columns_selected = ['PRECIPITACAO TOTAL, DIARIO (AUT)(mm)',
                    'TEMPERATURA MAXIMA, DIARIA (AUT)(°C)',
                    'TEMPERATURA MEDIA, DIARIA (AUT)(°C)',
                    'TEMPERATURA MINIMA, DIARIA (AUT)(°C)',
                    'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'
                   ]

for i in columns_selected:
    adjust_variables(column_name = i)

df_period_gaps

# COMMAND ----------

## Evaluating completeness of final stations

# Calculate the percentage of nulls per station
null_percentage_per_station = (
    df_period_gaps.groupby(['Station Code', 'Station'])
      .apply(lambda x: x.isnull().mean() * 100)
      .round(2)
)

null_percentage_per_station.to_csv('Data Climate/Data Climate Processed/analysis_null_percentage_per_station_gap30.csv')

null_percentage_per_station

# COMMAND ----------

## Creating new features: Consecutive Precipitation Days

stations = df_period_gaps['Station Code'].unique()

def calculate_consecutive_precipitation_days():
    global df_period_gaps, stations

    new_column_name = 'consecutive_precipitation_days_count'

    if new_column_name not in df_period_gaps.columns:
        df_period_gaps[new_column_name] = np.nan

    for station_i in stations:
        # Filter station data
        mask = df_period_gaps['Station Code'] == station_i
        df_temp = df_period_gaps.loc[mask].copy()

        # Define mask for precipitation > 0
        precipitation_mask = df_temp['PRECIPITACAO TOTAL, DIARIO (AUT)(mm)'] > 0

        # Identify groups of consecutive days with precipitation > 0
        group = (precipitation_mask != precipitation_mask.shift()).cumsum()

        # Count consecutive days > 0 within each group, otherwise set to zero
        df_temp[new_column_name] = (
            precipitation_mask.groupby(group)
            .cumsum()
            .where(precipitation_mask, 0)
            .astype(int)
        )

        # Update the original DataFrame
        df_period_gaps.loc[mask, new_column_name] = df_temp[new_column_name]

calculate_consecutive_precipitation_days()

# COMMAND ----------

## Aggregating data from daily to weekly

# Setting the date column as index
df_period_gaps.set_index('Data Medicao', inplace=True)

df_weekly = pd.DataFrame()

for station_name_i in stations:

    df_station_i = df_period_gaps[df_period_gaps['Station Code'] == station_name_i]

    df_station_weekly_i = df_station_i.copy()

    ## TOTAL PRECIPITATION
    
    # Count of filled days
    df_precipitation_filled_days = pd.DataFrame(df_station_i['PRECIPITACAO TOTAL, DIARIO (AUT)(mm)'].resample('W').count()).rename(columns={'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)': 'precipitation_filled_days'})
    df_station_weekly_i = df_station_weekly_i.merge(df_precipitation_filled_days, left_index=True, right_index=True, how='outer')
    
    # Weekly sum calculation
    df_precipitation_sum = pd.DataFrame(df_station_i['PRECIPITACAO TOTAL, DIARIO (AUT)(mm)'].resample('W').sum()).rename(columns={'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)': 'precipitation_weekly_sum'})
    df_station_weekly_i = df_station_weekly_i.merge(df_precipitation_sum, left_index=True, right_index=True)

    # Count of days with precipitation > 0
    df_precipitation_days_with_rain = pd.DataFrame(df_station_i['PRECIPITACAO TOTAL, DIARIO (AUT)(mm)'].resample('W').apply(lambda x: (x > 0).sum())).rename(columns={'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)': 'precipitation_days_with_rain'})
    df_station_weekly_i = df_station_weekly_i.merge(df_precipitation_days_with_rain, left_index=True, right_index=True)

    # Count of consecutive days with precipitation > 0
    df_precipitation_consecutive_days = pd.DataFrame(df_station_i['consecutive_precipitation_days_count'].resample('W').max()).rename(columns={'consecutive_precipitation_days_count': 'precipitation_consecutive_days_with_rain'})
    df_station_weekly_i = df_station_weekly_i.merge(df_precipitation_consecutive_days, left_index=True, right_index=True)
    
    
    ## MAXIMUM TEMPERATURE
    
    # Count of filled days
    df_temp_max_filled_days = pd.DataFrame(df_station_i['TEMPERATURA MAXIMA, DIARIA (AUT)(°C)'].resample('W').count()).rename(columns={'TEMPERATURA MAXIMA, DIARIA (AUT)(°C)': 'temp_max_filled_days'})
    df_station_weekly_i = df_station_weekly_i.merge(df_temp_max_filled_days, left_index=True, right_index=True)
    
    # Weekly mean calculation
    df_temp_max_mean = pd.DataFrame(df_station_i['TEMPERATURA MAXIMA, DIARIA (AUT)(°C)'].resample('W').mean()).rename(columns={'TEMPERATURA MAXIMA, DIARIA (AUT)(°C)': 'temp_max_weekly_mean'})
    df_station_weekly_i = df_station_weekly_i.merge(df_temp_max_mean, left_index=True, right_index=True)
    
    
    ## MEAN TEMPERATURE
    
    # Count of filled days
    df_temp_mean_filled_days = pd.DataFrame(df_station_i['TEMPERATURA MEDIA, DIARIA (AUT)(°C)'].resample('W').count()).rename(columns={'TEMPERATURA MEDIA, DIARIA (AUT)(°C)': 'temp_mean_filled_days'})
    df_station_weekly_i = df_station_weekly_i.merge(df_temp_mean_filled_days, left_index=True, right_index=True)
    
    # Weekly mean calculation
    df_temp_mean_weekly = pd.DataFrame(df_station_i['TEMPERATURA MEDIA, DIARIA (AUT)(°C)'].resample('W').mean()).rename(columns={'TEMPERATURA MEDIA, DIARIA (AUT)(°C)': 'temp_mean_weekly_mean'})
    df_station_weekly_i = df_station_weekly_i.merge(df_temp_mean_weekly, left_index=True, right_index=True)
    
    
    ## MINIMUM TEMPERATURE
    
    # Count of filled days
    df_temp_min_filled_days = pd.DataFrame(df_station_i['TEMPERATURA MINIMA, DIARIA (AUT)(°C)'].resample('W').count()).rename(columns={'TEMPERATURA MINIMA, DIARIA (AUT)(°C)': 'temp_min_filled_days'})
    df_station_weekly_i = df_station_weekly_i.merge(df_temp_min_filled_days, left_index=True, right_index=True)
    
    # Weekly mean calculation
    df_temp_min_mean = pd.DataFrame(df_station_i['TEMPERATURA MINIMA, DIARIA (AUT)(°C)'].resample('W').mean()).rename(columns={'TEMPERATURA MINIMA, DIARIA (AUT)(°C)': 'temp_min_weekly_mean'})
    df_station_weekly_i = df_station_weekly_i.merge(df_temp_min_mean, left_index=True, right_index=True)
    
    
    ## RELATIVE HUMIDITY
    
    # Count of filled days
    df_humidity_filled_days = pd.DataFrame(df_station_i['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'].resample('W').count()).rename(columns={'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)': 'humidity_filled_days'})
    df_station_weekly_i = df_station_weekly_i.merge(df_humidity_filled_days, left_index=True, right_index=True)
    
    # Weekly mean calculation
    df_humidity_mean = pd.DataFrame(df_station_i['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'].resample('W').mean()).rename(columns={'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)': 'humidity_weekly_mean'})
    df_station_weekly_i = df_station_weekly_i.merge(df_humidity_mean, left_index=True, right_index=True)

    df_weekly = pd.concat([df_weekly, df_station_weekly_i])
    
df_weekly

# COMMAND ----------

# Fixing precipitation-derived columns by setting null values when precipitation_filled_days = 0
df_weekly.loc[df_weekly['precipitation_filled_days'] == 0, 'precipitation_weekly_sum'] = np.nan
df_weekly.loc[df_weekly['precipitation_filled_days'] == 0, 'precipitation_days_with_rain'] = np.nan
df_weekly.loc[df_weekly['precipitation_filled_days'] == 0, 'precipitation_consecutive_days_with_rain'] = np.nan

# COMMAND ----------

# Count of null values per column
df_weekly.isnull().sum()

# COMMAND ----------

station_names = df_weekly['Station Code'].unique()

columns_to_fill = ['precipitation_weekly_sum', 
                    'precipitation_days_with_rain', 
                    'precipitation_consecutive_days_with_rain',
                    'temp_max_weekly_mean', 
                    'temp_mean_weekly_mean', 
                    'temp_min_weekly_mean', 
                    'humidity_weekly_mean']

# Defining the window size for the moving average (2 previous + 2 following + the value itself)
window = 5

df_weekly_filled = pd.DataFrame()

for station_name_i in station_names:

    df_station_i = df_weekly[df_weekly['Station Code'] == station_name_i]

    # Sort by date
    df_station_i.sort_values(by='Data Medicao', inplace=True)

    for column_i in columns_to_fill:
        # Filling missing values with the centered moving average
        df_station_i[column_i] = df_station_i[column_i].fillna(
            df_station_i[column_i].rolling(window=window, min_periods=1, center=True).mean()
        )

    df_weekly_filled = pd.concat([df_station_i, df_weekly_filled])

df_weekly_filled

# COMMAND ----------

df_weekly_filled_final = df_weekly_filled[[
    'Station', 
    'Station Code', 
    'Latitude', 
    'Longitude', 
    'Start Date', 
    'End Date', 
    'precipitation_weekly_sum',
    'precipitation_days_with_rain',
    'precipitation_consecutive_days_with_rain', 
    'temp_max_weekly_mean', 
    'temp_mean_weekly_mean', 
    'temp_min_weekly_mean', 
    'humidity_weekly_mean']].reset_index()
df_weekly_filled_final

# COMMAND ----------

# MAGIC %md
# MAGIC ### Converting Station -> IGR

# COMMAND ----------

# Convert df_final to GeoDataFrame with geometry
gdf_final = gpd.GeoDataFrame(
        df_weekly_filled_final,
        geometry=gpd.points_from_xy(df_weekly_filled_final['Longitude'], df_weekly_filled_final['Latitude']),
        crs="EPSG:4326"
)
gdf_final

# COMMAND ----------

# Load and align the IGR shapefile
gdf_igrs = gpd.read_file('../Data IBGE/BR_RG_Imediatas_2024/BR_RG_Imediatas_2024.shp')
gdf_igrs

# COMMAND ----------

# Reproject gdf_igrs to the same coordinate reference system (CRS) as gdf_final
gdf_igrs = gdf_igrs.to_crs(gdf_final.crs)

# Perform spatial join and update df_final
gdf_final = gpd.sjoin(gdf_final, gdf_igrs, how="left", predicate="within")

# Drop the geometry column
gdf_final = gdf_final.drop(columns=['geometry'], errors='ignore')

# Update df_final with the new micro-region columns
df_final = pd.DataFrame(gdf_final)
df_final

# COMMAND ----------

display(df_final[['Station Code', 'NM_RGI', 'SIGLA_UF']].drop_duplicates().groupby(['NM_RGI', 'SIGLA_UF']).count().reset_index())

# COMMAND ----------

station_count = df_final[['Station Code', 'NM_RGI', 'SIGLA_UF']].drop_duplicates().groupby(['NM_RGI', 'SIGLA_UF']).count().reset_index()

station_count[station_count['Station Code'] > 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Choosing the duplicate stations

# COMMAND ----------

duplicate_station_codes_to_exclude = []

# COMMAND ----------

# MAGIC %md
# MAGIC #### Belo Horizonte

# COMMAND ----------

# Getting NM_RGI from Belo Horizonte
x = df_final[['Station Code', 'Station', 'NM_RGI']].drop_duplicates()
x[x['NM_RGI'] == 'Belo Horizonte']

# COMMAND ----------

bh_analysis = df_period_gaps[df_period_gaps['Station Code'].isin(['dados_F501_D_2013-12-26_2025-08-01', 
                                        'dados_A521_D_2006-10-09_2025-08-01'])]
bh_analysis.groupby('Station Code').count()

# COMMAND ----------

## Looking at the above, the most complete station in Belo Horizonte is 'dados_A521_D_2006-10-09_2025-08-01'

duplicate_station_codes_to_exclude.append('dados_F501_D_2013-12-26_2025-08-01')
duplicate_station_codes_to_exclude

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cruz Alta

# COMMAND ----------

# Getting NM_RGI from Cruz Alta
x = df_final[['Station Code', 'Station', 'NM_RGI']].drop_duplicates()
x[x['NM_RGI'] == 'Cruz Alta']

# COMMAND ----------

cruz_alta_analysis = df_period_gaps[df_period_gaps['Station Code'].isin(['dados_A883_D_2012-12-12_2025-08-01',
'dados_A853_D_2007-05-30_2025-08-01'])]
cruz_alta_analysis.groupby('Station Code').count()

# COMMAND ----------

## Looking at the above, the most complete station in Cruz Alta is 'dados_A853_D_2007-05-30_2025-08-01'

duplicate_station_codes_to_exclude.append('dados_A883_D_2012-12-12_2025-08-01')
duplicate_station_codes_to_exclude

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distrito Federal

# COMMAND ----------

# Getting NM_RGI from Distrito Federal
x = df_final[['Station Code', 'Station', 'NM_RGI']].drop_duplicates()
x[x['NM_RGI'] == 'Distrito Federal']

# COMMAND ----------

brasilia_analysis = df_period_gaps[df_period_gaps['Station Code'].isin(['dados_A045_D_2008-10-02_2025-08-01', 
                                        'dados_A042_D_2017-07-18_2025-08-01', 
                                        'dados_A046_D_2014-09-30_2025-08-01',
                                        'dados_A047_D_2017-02-06_2025-08-01',
                                        'dados_A001_D_2000-05-06_2025-08-01'])]
brasilia_analysis.groupby('Station Code').count()

# COMMAND ----------

## Looking at the above, the most complete station in Brasília is 'dados_A001_D_2000-05-06_2025-08-01'

duplicate_station_codes_to_exclude.extend(['dados_A042_D_2017-07-18_2025-08-01', 'dados_A045_D_2008-10-02_2025-08-01',
'dados_A046_D_2014-09-30_2025-08-01',
'dados_A047_D_2017-02-06_2025-08-01'
])

duplicate_station_codes_to_exclude

# COMMAND ----------

# MAGIC %md
# MAGIC #### Juiz de Fora

# COMMAND ----------

# Getting NM_RGI from Juiz de Fora
x = df_final[['Station Code', 'Station', 'NM_RGI']].drop_duplicates()
x[x['NM_RGI'] == 'Juiz de Fora']

# COMMAND ----------

jf_analysis = df_period_gaps[df_period_gaps['Station Code'].isin([
    'dados_A557_D_2012-10-16_2025-08-01', 
    'dados_A518_D_2007-05-25_2025-08-01'
])]
jf_analysis.groupby('Station Code').count()

# COMMAND ----------

## Looking at the above, the most complete station in Juiz de Fora is 'dados_A557_D_2012-10-16_2025-08-01'

duplicate_station_codes_to_exclude.extend([
    'dados_A518_D_2007-05-25_2025-08-01'
])

duplicate_station_codes_to_exclude

# COMMAND ----------

# MAGIC %md
# MAGIC #### Rio de Janeiro

# COMMAND ----------

# Getting NM_RGI from Rio de Janeiro
x = df_final[['Station Code', 'Station', 'NM_RGI']].drop_duplicates()
x[x['NM_RGI'] == 'Rio de Janeiro']

# COMMAND ----------

rj_analysis = df_period_gaps[df_period_gaps['Station Code'].isin([
    'dados_A636_D_2017-08-09_2025-08-01', 
    'dados_A621_D_2007-04-12_2025-08-01'
])]
rj_analysis.groupby('Station Code').count()

# COMMAND ----------

## Looking at the above, the most complete station in Rio de Janeiro is 'dados_A636_D_2017-08-09_2025-08-01'

duplicate_station_codes_to_exclude.extend([
    'dados_A621_D_2007-04-12_2025-08-01'
])

duplicate_station_codes_to_exclude

# COMMAND ----------

# MAGIC %md
# MAGIC ### Removing duplicate 'Station Code' entries that we will not use

# COMMAND ----------

df_final_single_igr = df_final[~df_final['Station Code'].isin(duplicate_station_codes_to_exclude)]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Removing 'Station Code' entries with incomplete data

# COMMAND ----------

df_final_single_igr[['Station Code', 'Station']].value_counts(ascending=True).head(20)

# COMMAND ----------

# Check
display(df_final_single_igr[['Station Code', 'NM_RGI']].drop_duplicates().groupby('NM_RGI').count().reset_index())

# COMMAND ----------

df_final_single_igr.columns

# COMMAND ----------

df_final_single_igr = df_final_single_igr[['Data Medicao', 'NM_RGI', 'precipitation_weekly_sum',
       'precipitation_days_with_rain',
       'precipitation_consecutive_days_with_rain', 'temp_max_weekly_mean',
       'temp_mean_weekly_mean', 'temp_min_weekly_mean', 'humidity_weekly_mean']].rename(columns={'NM_RGI': 'igr'})
df_final_single_igr

# COMMAND ----------

# Export data
df_final_single_igr.to_csv('Data Climate/Data Climate Processed/processed_validated_stations_filled_gap30.csv', index=False)