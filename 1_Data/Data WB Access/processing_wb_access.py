# Databricks notebook source
import pandas as pd

# COMMAND ----------

# Importing WB Dengue Access 
wb_access = pd.read_csv('Data WB Access/raw_wb_access_dengue_20260416.csv', sep=',', encoding='latin1')
wb_access = wb_access.rename(columns={'count(1)': 'count'})
wb_access

# COMMAND ----------

## Converting city -> igr

# Import 
ibge = pd.read_csv('../Data IBGE/derived_igr_paper.csv')
ibge

# COMMAND ----------

wb_access_igr = wb_access.merge(ibge[['CITY_STANDARD', 'UF', 'IGR']], left_on=['access_city', 'access_state'], right_on=['CITY_STANDARD', 'UF'], how='inner')
wb_access_igr

# COMMAND ----------

# Check the count of cities after merge
wb_access_igr[['access_city', 'access_state']].drop_duplicates()

# COMMAND ----------

# Check the count of igrs after merge
print(wb_access_igr['IGR'].nunique())
sorted(wb_access_igr['IGR'].unique())

# COMMAND ----------

# Grouping to sum accesses from the same micro-region
wb_access_igr = wb_access_igr.groupby(['access_date', 'IGR'])['count'].sum().reset_index().rename(columns={'count': 'access_count'})

# Adjusting the date format
wb_access_igr['access_date'] = pd.to_datetime(wb_access_igr['access_date'], format='%Y-%m-%d')

# Sorting
wb_access_igr.sort_values(['IGR', 'access_date'], inplace=True)

wb_access_igr = wb_access_igr[
    (wb_access_igr['access_date'] >= '2021-01-01') &
    (wb_access_igr['access_date'] <= '2024-12-31')
]

wb_access_igr

# COMMAND ----------

# Filling missing days with 0

# Create a DataFrame with all possible combinations between dates and micro-regions
all_dates_wb = pd.date_range(wb_access_igr['access_date'].min(), wb_access_igr['access_date'].max(), freq='D')
all_micros_wb = wb_access_igr['IGR'].unique()

combinations_wb = pd.MultiIndex.from_product(
    [all_dates_wb, all_micros_wb],
    names=['access_date', 'IGR']
).to_frame(index=False)

# 3. Merge with the original DataFrame
wb_access_igr_complete = combinations_wb.merge(wb_access_igr, on=['access_date', 'IGR'], how='left')

# 4. Fill missing values with 0
wb_access_igr_complete['access_count'] = wb_access_igr_complete['access_count'].fillna(0).astype(int)

# Sorting
wb_access_igr_complete.sort_values(['IGR', 'access_date'], inplace=True)

wb_access_igr_complete

# COMMAND ----------

# Converting daily to weekly

# Creating week date column
wb_access_igr_complete['access_date_week'] = wb_access_igr_complete['access_date'].dt.strftime('%Y-%W')

# Creating year column from access_date
wb_access_igr_complete['year'] = wb_access_igr_complete['access_date'].dt.year.astype(str)

# Creating weekly grouping
wb_access_weekly = wb_access_igr_complete.groupby(['year', 'access_date_week', 'IGR'])['access_count'].sum().reset_index()
wb_access_weekly

# COMMAND ----------

import matplotlib.pyplot as plt

# Plot temporal access_count for each IGR
plt.figure(figsize=(12, 6))
for igr in wb_access_weekly['IGR'].unique():
    data = wb_access_weekly[wb_access_weekly['IGR'] == igr]
    plt.plot(data['access_date_week'], data['access_count'], label=str(igr))

plt.xlabel('Semana')
plt.ylabel('Access Count')
plt.title('Access Count semanal por IGR')
plt.legend(title='IGR', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# COMMAND ----------

wb_access_weekly['IGR'].nunique()

# COMMAND ----------

# Normalize per wb user year

active_physician_wb_yearly = pd.read_csv(
    '../Data WB Users/Data WB Users/active_physician_wb_yearly_igr.csv',
    dtype={'year_access': 'str'}
)
active_physician_wb_yearly

# COMMAND ----------

active_physician_wb_yearly.dtypes

# COMMAND ----------

wb_access_weekly.dtypes

# COMMAND ----------

wb_access_weekly = wb_access_weekly.merge(active_physician_wb_yearly, left_on=['IGR', 'year'], right_on=['IGR', 'year_access'], how='inner')

# COMMAND ----------

wb_access_weekly['rate_access_per_physician'] = wb_access_weekly['access_count'] / wb_access_weekly['count_physician'] * 10000
wb_access_weekly

# COMMAND ----------

wb_access_weekly[wb_access_weekly['rate_access_per_physician'] > 0]

# COMMAND ----------

# Export data
wb_access_weekly.to_csv('Data WB Access/ready_data_wb_access.csv', sep=';', index=False)