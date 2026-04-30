# Databricks notebook source
!pip install xlrd

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# Importing hospitalization data
sih_dengue = pd.read_csv('Data SIH Hospitalization/raw_data_sih_hospitalization.csv', encoding='latin1')

# Remove raws with column names
mask = (sih_dengue == sih_dengue.columns).all(axis=1)
sih_dengue = sih_dengue[~mask]
sih_dengue

# COMMAND ----------

# Importing ibge data
ibge = pd.read_excel('../Data IBGE/RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.xls', skiprows = 6)

# Remove last character from each value in 'Código Município Completo'
ibge['Código Município Completo'] = ibge['Código Município Completo'].astype(str).str[:-1]

# Rename columns
ibge.rename(columns={'Nome Região Geográfica Imediata': 'igr'}, inplace=True)

# Filtering interested columns
ibge = ibge[['Código Município Completo', 'Nome_Município', 'igr']]
ibge

# COMMAND ----------

ibge.nunique()

# COMMAND ----------

sih_dengue = sih_dengue.astype({'MUNIC_RES': 'str'})
ibge = ibge.astype({'Código Município Completo': 'str'})

# COMMAND ----------

sih_dengue_igr = sih_dengue.merge(ibge, left_on='MUNIC_RES', right_on='Código Município Completo')
sih_dengue_igr

# COMMAND ----------

sih_dengue_igr['date'] = pd.to_datetime(sih_dengue_igr['DT_INTER'])
sih_dengue_igr_daily = sih_dengue_igr[['date', 'igr', 'N_AIH']].groupby(['date', 'igr']).count().reset_index().rename(columns={'N_AIH': 'count_hospitalization'})
sih_dengue_igr_daily

# COMMAND ----------

# Completing missing date with zero counts


# Cria um DataFrame com todas as combinações possíveis entre datas e microrregiões
all_dates_sih = pd.date_range(sih_dengue_igr_daily['date'].min(), sih_dengue_igr_daily['date'].max(), freq='D')
all_igr_sih = sih_dengue_igr_daily['igr'].unique()


combinations_sih_dates = pd.MultiIndex.from_product(
    [all_dates_sih, all_igr_sih],
    names=['date', 'igr']
).to_frame(index=False)


sih_dengue_igr_daily_completed = combinations_sih_dates.merge(sih_dengue_igr_daily, on=['date', 'igr'], how='left')
sih_dengue_igr_daily_completed['count_hospitalization'] = sih_dengue_igr_daily_completed['count_hospitalization'].fillna(0).astype(int)

sih_dengue_igr_daily_completed.sort_values(['igr', 'date'], inplace=True)

sih_dengue_igr_daily_completed = sih_dengue_igr_daily_completed[
    (sih_dengue_igr_daily_completed['date'] >= '2021-01-01') &
    (sih_dengue_igr_daily_completed['date'] <= '2024-12-31')
]

sih_dengue_igr_daily_completed

# COMMAND ----------

# Converting daily to weekly

# Creating week date column
sih_dengue_igr_daily_completed['date_week'] = sih_dengue_igr_daily_completed['date'].dt.strftime('%Y-%W')

# Creating weekly grouping
sih_dengue_weekly = sih_dengue_igr_daily_completed.groupby(['date_week', 'igr'])['count_hospitalization'].sum().reset_index()
sih_dengue_weekly

# COMMAND ----------

plt.figure(figsize=(16, 8))
for igr in sih_dengue_weekly['igr'].unique():
    data = sih_dengue_weekly[sih_dengue_weekly['igr'] == igr]
    plt.plot(data['date_week'], data['count_hospitalization'], label=igr)
plt.xlabel('Semana')
plt.ylabel('Hospitalizações')
plt.title('Hospitalizações por Dengue por Microrregião ao longo do tempo')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# COMMAND ----------

# sih_dengue_weekly.to_csv('Data SIH Hospitalization/ready_data_sih_hospitalization.csv', index=False)