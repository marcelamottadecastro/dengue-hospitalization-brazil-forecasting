# Databricks notebook source
import pandas as pd

# COMMAND ----------

data_union = pd.read_csv('../2_Data_Union/ready_data_all_features.csv')
data_union

# COMMAND ----------

data_union['date_week'] = pd.to_datetime(data_union['date_week'].str[:4] + data_union['date_week'].str[5:] + '1', format='%Y%W%w')

# COMMAND ----------

igr_graph = [
    'Belo Horizonte',
    'Distrito Federal',
    'Maringá',
    'Passos',
    'Uberaba',
    'Uberlândia'
]

data_graph = data_union[data_union['igr'].isin(igr_graph)]
data_graph

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

for i, igr in enumerate(data_graph['igr'].unique()):
    df = data_graph[data_graph['igr'] == igr]
    weeks = df['date_week']
    ax1 = axes[i]
    ax2 = ax1.twinx()
    ax1.plot(weeks, df['rate_access_per_physician'], color='#4F81BD', label='Clinical search rate per 10,000 physicians', linewidth=1)
    ax2.plot(weeks, df['count_hospitalization'], color='#FF6F40', label='Weekly hospitalizations', linewidth=1)
    ax1.set_title(f'{igr}')
    ax1.set_ylabel('Clinical search rate per 10,000 physicians', color='#4F81BD')
    ax2.set_ylabel('Weekly hospitalizations', color='#FF6F40')
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 0.78))
    # Set x-axis ticks to years only
    years = sorted(df['date_week'].dt.year.unique())
    year_ticks = [df[df['date_week'].dt.year == year]['date_week'].iloc[0] for year in years]
    ax1.set_xticks(year_ticks)
    ax1.set_xticklabels([str(year) for year in years])

plt.tight_layout()
plt.show()

# COMMAND ----------

fig.savefig('Series_Temporais.tif', dpi=100, format='tif')