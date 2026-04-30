# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import copy
import os
import random

warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection

# COMMAND ----------

hospitalization_climate = pd.read_excel('../../3_Feature_Selection/metricas_comparacao_2025_10_17.xlsx')

hospitalization_climate.rename(columns={'microrregiao': 'igr', 'modelo': 'model', 'arquitetura_rede': 'network_architecture', 'variaveis': 'features'}, inplace=True)

hospitalization_climate['features_analysis'] = (
    hospitalization_climate['features']
    .str.replace(r"'|NUMERO_INTERNACAO|[,\]\[]|\s", "", regex=True)
)
display(hospitalization_climate)

# COMMAND ----------

translation_features = {
    'CATEGORIA_TEMPERATURA_QUANTIL_01': 'temperature_category_quantile_01',
    'CATEGORIA_TEMPERATURA_QUANTIL_02': 'temperature_category_quantile_02',
    'CATEGORIA_TEMPERATURA_QUANTIL_03': 'temperature_category_quantile_03',
    'CATEGORIA_TEMPERATURA_QUANTIL_04': 'temperature_category_quantile_04',
    'CLIMA_EXTREMO_PRECIPITACAO': 'extreme_weather_precipitation',
    'CLIMA_EXTREMO_TEMPERATURA': 'extreme_weather_temperature',
    'CLIMA_EXTREMO_UMIDADE': 'extreme_weather_humidity',
    'DIAS_CHUVA': 'rain_days',
    'ESTACAO_INVERNO': 'season_winter',
    'ESTACAO_OUTONO': 'season_autumn',
    'ESTACAO_PRIMAVERA': 'season_spring',
    'ESTACAO_VERAO': 'season_summer',
    'LAG_1_PRECIPITACAO': 'lag_1_precipitation',
    'LAG_1_TEMPERATURA_MEDIA': 'lag_1_temp_mean',
    'LAG_1_UMIDADE': 'lag_1_humidity',
    'LAG_2_PRECIPITACAO': 'lag_2_precipitation',
    'LAG_2_TEMPERATURA_MEDIA': 'lag_2_temp_mean',
    'LAG_2_UMIDADE': 'lag_2_humidity',
    'LAG_3_PRECIPITACAO': 'lag_3_precipitation',
    'LAG_3_TEMPERATURA_MEDIA': 'lag_3_temp_mean',
    'LAG_3_UMIDADE': 'lag_3_humidity',
    'LAG_4_PRECIPITACAO': 'lag_4_precipitation',
    'LAG_4_TEMPERATURA_MEDIA': 'lag_4_temp_mean',
    'LAG_4_UMIDADE': 'lag_4_humidity',
    'MUDANCA_PRECIPITACAO': 'change_precipitation',
    'MUDANCA_PRECIPITACAO_MAIOR75': 'change_precipitation_above75',
    'MUDANCA_TEMPERATURA': 'change_temperature',
    'MUDANCA_TEMPERATURA_MAIOR75': 'change_temperature_above75',
    'MUDANCA_UMIDADE': 'change_humidity',
    'MUDANCA_UMIDADE_MAIOR75': 'change_humidity_above75',
    'NUMERO_ACESSO': 'access_count',
    'OCORRENCIA_PRECIPITACAO': 'precipitation_occurrence',
    'PERCENTIL_LAG_PRECIPITACAO_SOMA_SEMANAL': 'percentile_lag_precipitation_weekly_sum',
    'PERCENTIL_LAG_TEMPERATURA_MEDIA_MEDIA_SEMANAL': 'percentile_lag_temp_mean_weekly_mean',
    'PERCENTIL_LAG_UMIDADE_MEDIA_SEMANAL': 'percentile_lag_humidity_weekly_mean',
    'PERCENTIL_PRECIPITACAO_SOMA_SEMANAL': 'percentile_precipitation_weekly_sum',
    'PERCENTIL_TEMPERATURA_MEDIA_MEDIA_SEMANAL': 'percentile_temp_mean_weekly_mean',
    'PERCENTIL_UMIDADE_MEDIA_SEMANAL': 'percentile_humidity_weekly_mean',
    'PRECIPITACAO_SOMA_SEMANAL': 'precipitation_weekly_sum',
    'TEMPERATURA_MAX_MEDIA_SEMANAL': 'temp_max_weekly_mean',
    'TEMPERATURA_MEDIA_MEDIA_SEMANAL': 'temp_mean_weekly_mean',
    'TEMPERATURA_MINIMA_MEDIA_SEMANAL': 'temp_min_weekly_mean',
    'UMIDADE_MEDIA_SEMANAL': 'humidity_weekly_mean',
}

hospitalization_climate['features_analysis'] = hospitalization_climate['features_analysis'].map(translation_features)

display(hospitalization_climate)

# COMMAND ----------

map_features = {
'temperature_category_quantile_01': 'temperature',
'temperature_category_quantile_02': 'temperature',
'temperature_category_quantile_03': 'temperature',
'temperature_category_quantile_04': 'temperature',
'extreme_weather_precipitation': 'precipitation',
'extreme_weather_temperature': 'temperature',
'extreme_weather_humidity': 'humidity',
'rain_days': 'precipitation',
'season_winter': 'season',
'season_autumn': 'season',
'season_spring': 'season',
'season_summer': 'season',
'lag_1_precipitation': 'precipitation',
'lag_1_temp_mean': 'temperature',
'lag_1_humidity': 'humidity',
'lag_2_precipitation': 'precipitation',
'lag_2_temp_mean': 'temperature',
'lag_2_humidity': 'humidity',
'lag_3_precipitation': 'precipitation',
'lag_3_temp_mean': 'temperature',
'lag_3_humidity': 'humidity',
'lag_4_precipitation': 'precipitation',
'lag_4_temp_mean': 'temperature',
'lag_4_humidity': 'humidity',
'change_precipitation': 'precipitation',
'change_precipitation_above75': 'precipitation',
'change_temperature': 'temperature',
'change_temperature_above75': 'temperature',
'change_humidity': 'humidity',
'change_humidity_above75': 'humidity',
'rate_access_per_physician': 'access',
'precipitation_occurrence': 'precipitation',
'percentile_lag_precipitation_weekly_sum': 'precipitation',
'percentile_lag_temp_mean_weekly_mean': 'temperature',
'percentile_lag_humidity_weekly_mean': 'humidity',
'percentile_precipitation_weekly_sum': 'precipitation',
'percentile_temp_mean_weekly_mean': 'temperature',
'percentile_humidity_weekly_mean': 'humidity',
'precipitation_weekly_sum': 'precipitation',
'temp_max_weekly_mean': 'temperature',
'temp_mean_weekly_mean': 'temperature',
'temp_min_weekly_mean': 'temperature',
'humidity_weekly_mean': 'humidity'
}

# # Função para aplicar a lógica "contém"
# def categorization(texto):
#     for chave, valor in map_features.items():
#         if chave.lower() in texto.lower():  # comparação case-insensitive
#             return valor
#     return None  # ou texto, se quiser manter o original

# Criando coluna categoria
hospitalization_climate['category'] = hospitalization_climate['features_analysis'].map(map_features)

display(hospitalization_climate)

# COMMAND ----------

best_each_type = hospitalization_climate.loc[
    hospitalization_climate.groupby(['igr', 'category'])['RMSE'].idxmin()
]


# Agrupando por 'igr' e colocando os valores da coluna 'features_analysis' em listas
features_climate = (
    best_each_type
    .groupby('igr')['features_analysis']
    .agg(list)
    .apply(lambda x: x[::-1])  # inverte a ordem da lista
    .reset_index()
)

display(features_climate)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing data

# COMMAND ----------

data_union = pd.read_csv('../../2_Data_Union/ready_data_all_features.csv')
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
# MAGIC ### Sarimax

# COMMAND ----------

# Set the seed
seed_number = random.randint(0, 10000)
np.random.seed(seed_number)

col_target = ['count_hospitalization']
comparison_metrics = []

train_test_split = 0.70

delay_size = 8

# COMMAND ----------

# Adjusting SARIMA model
for igr_i in data['igr'].unique():
    print(f"\n▶️ STARTING: {igr_i}")

    try:
        df_igr = data[data['igr'] == igr_i].copy()
        df_igr = df_igr.sort_values('date_week')

        df_igr['date_week'] = pd.to_datetime(df_igr['date_week'] + '-1', format='%Y-%W-%w')
        df_igr.set_index('date_week', inplace=True)

        y = df_igr[col_target].astype(float).round(1)

        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
        y_scaled = pd.Series(y_scaled.flatten(), index=y.index)

        split = int(len(y_scaled) * train_test_split)
        y_train, y_test = y_scaled.iloc[:split-delay_size], y_scaled.iloc[split:]

        variants = [
            {'model_name': 'SARIMA_UNIVARIATE', 'exog_col': None},
        ]

        for variant in variants:
            model_name = variant['model_name']
            exog_col = variant['exog_col']

            try:
                print(f"   → Adjusting {model_name}")

                X_train = df_igr[exog_col].iloc[:split] if exog_col else None
                X_test = df_igr[exog_col].iloc[split:] if exog_col else None

                # Definition of hyperparameter ranges
                p = q = [0, 1]   # AR and MA: 0 or 1
                d = [1]          # I: usually 1 for non-stationary series
                P = Q = [0, 1]   # Seasonal AR and MA: 0 or 1
                D = [1]          # Seasonal I: usually 1
                s = [52]         # Weekly seasonality

                # Create all possible combinations
                param_combinations = list(itertools.product(p, d, q, P, D, Q, s))

                for comb in param_combinations:
                    order = (comb[0], comb[1], comb[2])
                    seasonal_order = (comb[3], comb[4], comb[5], comb[6])
                
                    model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
                    fitted = model.fit(disp=False)
                    pred = fitted.predict(start=split, end=len(df_igr)-1, exog=X_test)

                    resid = y_test - pred
                    mse = mean_squared_error(y_test, pred)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    mae = mean_absolute_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    naive_pred = y_train.shift(1).bfill()
                    mae_naive = mean_absolute_error(y_train[1:], naive_pred[1:])
                    mase = mae / mae_naive if mae_naive != 0 else np.nan

                    comparison_metrics.append({
                        'igr': igr_i,
                        'model': model_name,
                        'exogenous_variables': exog_col if exog_col else [],
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'MSE': mse,
                        'y_train': y_train,
                        'y_test': y_test,
                        'pred': pred,
                        'order': str(order),
                        'seasonal_order': str(seasonal_order),
                    })

            except Exception as e:
                print(f"   ❌ ERROR in adjustment {model_name} | {igr_i} | {str(e)}")


    except Exception as e:
        print(f"❌ GENERAL ERROR | {igr_i} | {str(e)}")

# COMMAND ----------

best_per_igr = {}

for item in comparison_metrics:
    igr = item["igr"]
    rmse = item["RMSE"]
    
    # If not present or current RMSE is lower than the saved one
    if igr not in best_per_igr or rmse < best_per_igr[igr]["RMSE"]:
        best_per_igr[igr] = item

# Convert back to list if needed
best_list = list(best_per_igr.values())

print(best_list)

# COMMAND ----------

# ==== Visualization ====
for comparison_i in best_list:
    igr = comparison_i['igr']
    y_train = comparison_i['y_train']
    y_test = comparison_i['y_test']
    pred = comparison_i['pred']

    try:
        plt.figure(figsize=(12, 5))

        plt.plot(range(len(y_train)), y_train, label='Train - Actual', color='blue')

        test_start = len(y_train)
        plt.plot(range(test_start, test_start + len(y_test)), y_test, label='Test - Actual', color='green')
        plt.plot(range(test_start, test_start + len(pred)), pred, label='Test - Predicted', linestyle='--', color='lightgreen')

        plt.axvline(test_start, color='gray', linestyle=':', label='Train/Test Split')
        plt.title(f'Hospitalization Forecast with Sarimax - {igr}')
        plt.xlabel('Weeks')
        plt.ylabel('Hospitalizations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"   ❌ ERROR in plot {igr} | {str(e)}")

# COMMAND ----------

# Saving results
try:
    df_final_metrics = pd.DataFrame(best_list)

    df_final_metrics.sort_values(by='igr', axis=0, inplace=True)

    df_final_metrics.drop(columns=['y_train', 'y_test', 'pred'], errors='ignore').to_excel(f"best_sarimax_with_delay_{seed_number}.xlsx", index=False)

    print(f"✔️ Metrics successfully saved!")
    print(f"🔢 Total models saved: {df_final_metrics.shape[0]}")

except Exception as e:
    print(e)
    print("❌ Error saving final metrics.")