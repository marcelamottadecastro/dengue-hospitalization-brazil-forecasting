# Databricks notebook source
import pandas as pd

# COMMAND ----------

# Data WB Access

wb_access = pd.read_csv('../1_Data/Data WB Access/Data WB Access/ready_data_wb_access.csv', sep=';')
wb_access

# COMMAND ----------

# Data SIH Hospitalization

sih_dengue = pd.read_csv('../1_Data/Data SIH Hospitalization/Data SIH Hospitalization/ready_data_sih_hospitalization.csv')
sih_dengue
#

# COMMAND ----------

# Data Climate

climate = pd.read_csv('../1_Data/Data Climate/Data Climate/Data Climate Processed/ready_validated_stations_filled_gap30_all_features.csv')
climate

# COMMAND ----------

# Merge wb_access, sih_dengue, and climate tables
merged_df = wb_access.merge(sih_dengue, how='inner', left_on=['access_date_week', 'IGR'], right_on=['date_week', 'igr']).merge(climate, how='inner', left_on=['access_date_week', 'IGR'], right_on=['date', 'igr'])

merged_df.sort_values(by=['IGR', 'date_week'], inplace=True)
merged_df

# COMMAND ----------

#Columns to drop
columns_to_drop = ['year', 'access_date_week', 'year_access', 'igr_x', 'igr_y', 'date']
merged_df.drop(columns=columns_to_drop, inplace=True)

# Move 'date_week' to the first column
cols = merged_df.columns.tolist()
cols.insert(0, cols.pop(cols.index('date_week')))
merged_df = merged_df[cols]

merged_df.rename(columns={'IGR': 'igr'}, inplace=True)

merged_df.dropna(axis=0, inplace=True)

merged_df

# COMMAND ----------

#merged_df.to_csv('ready_data_all_features.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization

# COMMAND ----------

import matplotlib.pyplot as plt

merged_df = pd.read_csv('ready_data_all_features.csv')

# COMMAND ----------

igrs = merged_df['igr'].unique()
fig, axes = plt.subplots(len(igrs), 1, figsize=(12, 4 * len(igrs)), sharex=True)

if len(igrs) == 1:
    axes = [axes]

for ax, microrregiao in zip(axes, igrs):
    df = merged_df[merged_df['igr'] == microrregiao].sort_values('date_week')
    ax.plot(df['date_week'], df['access_count'], label='access_count')
    ax.plot(df['date_week'], df['count_hospitalization'], label='count_hospitalization')
    ax.set_title(microrregiao)
    ax.set_ylabel('Number')
    ax.legend()
    ax.grid(True)

plt.xlabel('Week')
plt.tight_layout()
display(fig)
plt.close(fig)