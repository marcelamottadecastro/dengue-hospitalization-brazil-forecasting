# Databricks notebook source
pip install openpyxl

# COMMAND ----------

import pandas as pd
import numpy as np
import math
import random

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

import copy
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection

# COMMAND ----------

hospitalization_climate = pd.read_excel('../3_Feature_Selection/metricas_comparacao_2025_10_17.xlsx')

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

# Creating category column
hospitalization_climate['category'] = hospitalization_climate['features_analysis'].map(map_features)

display(hospitalization_climate)

# COMMAND ----------

best_each_type = hospitalization_climate.loc[
    hospitalization_climate.groupby(['igr', 'category'])['RMSE'].idxmin()
]

# Grouping by 'igr' and placing the values from the 'features_analysis' column into lists
features_climate = (
    best_each_type
    .groupby('igr')['features_analysis']
    .agg(list)
    .apply(lambda x: x[::-1])  # reverse the order of the list
    .reset_index()
)

display(features_climate)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing data

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
# MAGIC ### Global parameters

# COMMAND ----------

# Network parameters
OUTPUT_SIZE = 8
DELAY_SIZE = 8
LR = 0.01

TRAIN_TEST_SPLIT = 0.70

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP analysis of the model WITHOUT delay on the hospitalization variable

# COMMAND ----------

# Set seed for reproducibility.
# seed_number = 5555 # Same as the best model
# seed_number = 6703 # Same as the best model
seed_number = 759 # Same as the best model

model_name = 'LSTM'

# COMMAND ----------

# best = pd.read_excel(f"../4_Model_Training/Without_Delay/best_lstm_without_delay_{seed_number}_70.xlsx")

# best_6lstm = best[best['network_architecture'] == '6 LSTMs']
# best_6lstm

# COMMAND ----------

# # Variáveis de clima por igr e acesso WB

# metrics_comparison = []

# for igr_i in data['igr'].unique():

#     torch.manual_seed(seed_number)
#     np.random.seed(seed_number) 

#     loss_list = []
#     test_loss_list = []

#     mse_list = []
#     rmse_list = []  
#     r2_list = []
#     mae_list = []

#     data_igr = data[data['igr'] == igr_i]
#     features_climate_micro = features_climate.loc[features_climate['igr'] == igr_i, 'features_analysis'].iloc[0]

#     # Hiperâmetros do melhor modelo de cada microrregião
#     EPOCHS = best_6lstm.loc[best_6lstm['igr'] == igr_i, 'epoch'].iloc[0] + 1
#     NUM_LAYERS = best_6lstm.loc[best_6lstm['igr'] == igr_i, 'n_layers'].iloc[0]
#     HIDDEN_SIZE = int(best_6lstm.loc[best_6lstm['igr'] == igr_i, 'hidden_size'].iloc[0])
#     SEQ_LEN = best_6lstm.loc[best_6lstm['igr'] == igr_i, 'seq_len'].iloc[0]
    
#     var_exogenicas = ['rate_access_per_physician'] + features_climate_micro
#     var_algo = ['count_hospitalization']
#     var_total = var_algo + var_exogenicas

#     # ==== Dados ====
#     x1_seq = data_igr['count_hospitalization'].values.astype(np.float32)
#     x2_seq = data_igr[var_exogenicas[0]].values.astype(np.float32)
#     x3_seq = data_igr[var_exogenicas[1]].values.astype(np.float32)
#     x4_seq = data_igr[var_exogenicas[2]].values.astype(np.float32)
#     x5_seq = data_igr[var_exogenicas[3]].values.astype(np.float32)
#     x6_seq = data_igr[var_exogenicas[4]].values.astype(np.float32)

#     scaler_x1 = MinMaxScaler()
#     scaler_x2 = MinMaxScaler()
#     scaler_x3 = MinMaxScaler()
#     scaler_x4 = MinMaxScaler()
#     scaler_x5 = MinMaxScaler()
#     scaler_x6 = MinMaxScaler()
#     x1_scaled = scaler_x1.fit_transform(x1_seq.reshape(-1, 1))
#     x1_scaled = x1_scaled.ravel() # Voltar para o shape original
#     x2_scaled = scaler_x2.fit_transform(x2_seq.reshape(-1, 1))
#     x2_scaled = x2_scaled.ravel()
#     x3_scaled = scaler_x3.fit_transform(x3_seq.reshape(-1, 1))
#     x3_scaled = x3_scaled.ravel()
#     x4_scaled = scaler_x4.fit_transform(x4_seq.reshape(-1, 1))
#     x4_scaled = x4_scaled.ravel()
#     x5_scaled = scaler_x5.fit_transform(x5_seq.reshape(-1, 1))
#     x5_scaled = x5_scaled.ravel()
#     x6_scaled = scaler_x6.fit_transform(x6_seq.reshape(-1, 1))
#     x6_scaled = x6_scaled.ravel()

#     # ==== Função para criar janelas ====
#     def create_sequences_duplas(var1, var2, var3, var4, var5, var6, target, seq_len, output_size):
#         x1s, x2s, x3s, x4s, x5s, x6s, ys = [], [], [], [], [], [], []
#         for i in range(len(target) - seq_len- output_size + 1):
#             x1 = var1[i:i+seq_len]
#             x2 = var2[i:i+seq_len]
#             x3 = var3[i:i+seq_len]
#             x4 = var4[i:i+seq_len]
#             x5 = var5[i:i+seq_len]
#             x6 = var6[i:i+seq_len]
#             y = target[i+seq_len:i+seq_len+output_size]
#             x1s.append(x1)
#             x2s.append(x2)
#             x3s.append(x3)
#             x4s.append(x4)
#             x5s.append(x5)
#             x6s.append(x6)
#             ys.append(y)
#         return (
#             torch.tensor(x1s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
#             torch.tensor(x2s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
#             torch.tensor(x3s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
#             torch.tensor(x4s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
#             torch.tensor(x5s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
#             torch.tensor(x6s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
#             torch.tensor(ys).float()   # [N, 1]
#         )

#     # ==== Separação treino/teste ====
#     train_size = int(len(data_igr) * TRAIN_TEST_SPLIT)

#     x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, y_train = create_sequences_duplas(
#         x1_scaled[:train_size],
#         x2_scaled[:train_size],
#         x3_scaled[:train_size],
#         x4_scaled[:train_size],
#         x5_scaled[:train_size],
#         x6_scaled[:train_size],
#         x1_scaled[:train_size],
#         SEQ_LEN,
#         output_size=OUTPUT_SIZE
#     )

#     x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, y_test = create_sequences_duplas(
#         x1_scaled[train_size - SEQ_LEN:],
#         x2_scaled[train_size - SEQ_LEN:],
#         x3_scaled[train_size - SEQ_LEN:],
#         x4_scaled[train_size - SEQ_LEN:],
#         x5_scaled[train_size - SEQ_LEN:],
#         x6_scaled[train_size - SEQ_LEN:],
#         x1_scaled[train_size - SEQ_LEN:],
#         SEQ_LEN,
#         output_size=OUTPUT_SIZE
#     )

#     # ==== Modelo com duas LSTMs ====
#     class MultLSTM(nn.Module):
#         def __init__(self, input_size, hidden_size, num_layers, output_size):
#             super(MultLSTM, self).__init__()
#             self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers)
#             self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers)
#             self.lstm3 = nn.LSTM(input_size, hidden_size, num_layers)
#             self.lstm4 = nn.LSTM(input_size, hidden_size, num_layers)
#             self.lstm5 = nn.LSTM(input_size, hidden_size, num_layers)
#             self.lstm6 = nn.LSTM(input_size, hidden_size, num_layers)
#             self.fc = nn.Linear(hidden_size * 6, output_size)  # concatenar as duas saídas

#         def forward(self, x1, x2, x3, x4, x5, x6):
#             out1, _ = self.lstm1(x1)  # shape: (seq_len, batch, hidden)
#             out2, _ = self.lstm2(x2)
#             out3, _ = self.lstm3(x3)
#             out4, _ = self.lstm4(x4)
#             out5, _ = self.lstm5(x5)
#             out6, _ = self.lstm6(x6)
#             last_out1 = out1[-1]  # shape: (batch, hidden)
#             last_out2 = out2[-1]
#             last_out3 = out3[-1]
#             last_out4 = out4[-1]
#             last_out5 = out5[-1]
#             last_out6 = out6[-1]
#             combined = torch.cat((last_out1, last_out2, last_out3, last_out4, last_out5, last_out6), dim=1)
#             output = self.fc(combined)
#             return output

#     model = MultLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     # ==== Treinamento ====
#     for epoch in range(EPOCHS):
#         model.train()
#         optimizer.zero_grad()

#         inp1 = x1_train.permute(1, 0, 2)  # [seq_len, batch, 1]
#         inp2 = x2_train.permute(1, 0, 2)
#         inp3 = x3_train.permute(1, 0, 2)
#         inp4 = x4_train.permute(1, 0, 2)
#         inp5 = x5_train.permute(1, 0, 2)
#         inp6 = x6_train.permute(1, 0, 2)
#         output = model(inp1, inp2, inp3, inp4, inp5, inp6)
#         loss = criterion(output, y_train)

#         loss.backward()
#         optimizer.step()

#         # ==== Avaliação ====
#         model.eval()
#         with torch.no_grad():
#             pred_train = model(x1_train.permute(1, 0, 2), x2_train.permute(1, 0, 2), x3_train.permute(1, 0, 2), x4_train.permute(1, 0, 2), x5_train.permute(1, 0, 2), x6_train.permute(1, 0, 2)).squeeze().numpy()
#             pred_test = model(x1_test.permute(1, 0, 2), x2_test.permute(1, 0, 2), x3_test.permute(1, 0, 2), x4_test.permute(1, 0, 2), x5_test.permute(1, 0, 2), x6_test.permute(1, 0, 2)).squeeze().numpy()
#             true_train = y_train.squeeze().numpy()
#             true_test = y_test.squeeze().numpy()

#         # ==== Cálculo do erro de teste ====
#         final_loss = criterion(torch.tensor(pred_test), torch.tensor(true_test)).item()
#         test_loss_list.append(final_loss)
#         rmse = np.sqrt(final_loss)
#         rmse_list.append(rmse)
#         r2 = r2_score(true_test, pred_test)
#         r2_list.append(r2)
#         mae = mean_absolute_error(true_test, pred_test)
#         mae_list.append(mae)

#     min_rmse = min(rmse_list)
#     min_rmse_epoch = rmse_list.index(min_rmse)
#     min_mse = test_loss_list[min_rmse_epoch]
#     min_r2 = r2_list[min_rmse_epoch]
#     min_mae = mae_list[min_rmse_epoch]

#     background = [
#         x1_train,
#         x2_train,
#         x3_train,
#         x4_train,
#         x5_train,
#         x6_train
#     ]

#     # Número de amostras que queremos explicar
#     num_amostras = 50  

#     # Limitar ao número disponível
#     num_amostras = min(num_amostras, x1_test.shape[0])

#     # Selecionando múltiplas amostras de cada entrada
#     X = [
#         x1_test[:num_amostras],
#         x2_test[:num_amostras],
#         x3_test[:num_amostras],
#         x4_test[:num_amostras],
#         x5_test[:num_amostras],
#         x6_test[:num_amostras]
#     ]

#     explainer = shap.GradientExplainer(model, background)
#     shap_values = explainer.shap_values(X)

#     metrics_comparison.append({
#         'igr': igr_i,
#         'model': model_name,
#         'network_architecture': '6 LSTMs',
#         'features': var_total,
#         'RMSE': f"{min_rmse:.4f}",
#         'MSE': f"{min_mse:.4f}",
#         'R2': f"{min_r2:.4f}",
#         'MAE': f"{min_mae:.4f}",
#         'epoch': EPOCHS,
#         'n_layers': NUM_LAYERS,
#         'hidden_size': HIDDEN_SIZE,
#         'seq_len': SEQ_LEN,
#         'shap_values': shap_values,
#     })

# COMMAND ----------

# # Dataset creation
# df_metrics_final = pd.DataFrame(metrics_comparison)
# df_metrics_final.sort_values(by='igr', axis=0, inplace=True)

# # Calculate the mean SHAP values for each variable
# def calculate_mean_shap(x):
#     shap_values = x
#     shap_array = np.stack(shap_values, axis=-1)
#     mean_shap = np.abs(shap_array).mean(axis=(0,1,2,3))  # shape: (6,)
#     return mean_shap

# df_metrics_final["mean_shap"] = df_metrics_final["shap_values"].apply(calculate_mean_shap)

# # Saving file
# df_metrics_final.to_excel(f'SHAP_without_delay_6LSTMs_{seed_number}.xlsx', index=False)

# df_metrics_final

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP analysis of the model WITH delay on the hospitalization variable

# COMMAND ----------

# Fixar seed para reprodutibilidade. Escolher seed aleatória
# seed_number = 2108 # Same as the best model
# seed_number = 3201 # Same as the best model
seed_number = 9698 # Same as the best model

model_name = 'LSTM'

# COMMAND ----------

best = pd.read_excel(f"../4_Model_Training/With_Delay/best_lstm_with_delay_{seed_number}_70.xlsx")

best_6lstm = best[best['network_architecture'] == '6 LSTMs']
best_6lstm

# COMMAND ----------

# Variáveis de clima por microrregiao e acesso WB

metrics_comparison = []

for igr_i in data['igr'].unique():

    torch.manual_seed(seed_number)
    np.random.seed(seed_number)

    loss_list = []
    test_loss_list = [] 

    mse_list = []
    rmse_list = []  
    r2_list = []
    mae_list = []

    data_igr = data[data['igr'] == igr_i]
    features_climate_igr = features_climate.loc[features_climate['igr'] == igr_i, 'features_analysis'].iloc[0]

    # Hiperâmetros do melhor modelo de cada microrregião
    EPOCHS = best_6lstm.loc[best_6lstm['igr'] == igr_i, 'epoch'].iloc[0] + 1
    NUM_LAYERS = best_6lstm.loc[best_6lstm['igr'] == igr_i, 'n_layers'].iloc[0]
    HIDDEN_SIZE = int(best_6lstm.loc[best_6lstm['igr'] == igr_i, 'hidden_size'].iloc[0])
    SEQ_LEN = best_6lstm.loc[best_6lstm['igr'] == igr_i, 'seq_len'].iloc[0]
    
    var_exogenicas = ['rate_access_per_physician'] + features_climate_igr
    var_algo = ['count_hospitalization']
    var_total = var_algo + var_exogenicas

    # ==== Dados ====
    x1_seq = data_igr['count_hospitalization'].values.astype(np.float32)
    x2_seq = data_igr[var_exogenicas[0]].values.astype(np.float32)
    x3_seq = data_igr[var_exogenicas[1]].values.astype(np.float32)
    x4_seq = data_igr[var_exogenicas[2]].values.astype(np.float32)
    x5_seq = data_igr[var_exogenicas[3]].values.astype(np.float32)
    x6_seq = data_igr[var_exogenicas[4]].values.astype(np.float32)

    scaler_x1 = MinMaxScaler()
    scaler_x2 = MinMaxScaler()
    scaler_x3 = MinMaxScaler()
    scaler_x4 = MinMaxScaler()
    scaler_x5 = MinMaxScaler()
    scaler_x6 = MinMaxScaler()
    x1_scaled = scaler_x1.fit_transform(x1_seq.reshape(-1, 1))
    x1_scaled = x1_scaled.ravel() # Voltar para o shape original
    x2_scaled = scaler_x2.fit_transform(x2_seq.reshape(-1, 1))
    x2_scaled = x2_scaled.ravel()
    x3_scaled = scaler_x3.fit_transform(x3_seq.reshape(-1, 1))
    x3_scaled = x3_scaled.ravel()
    x4_scaled = scaler_x4.fit_transform(x4_seq.reshape(-1, 1))
    x4_scaled = x4_scaled.ravel()
    x5_scaled = scaler_x5.fit_transform(x5_seq.reshape(-1, 1))
    x5_scaled = x5_scaled.ravel()
    x6_scaled = scaler_x6.fit_transform(x6_seq.reshape(-1, 1))
    x6_scaled = x6_scaled.ravel()

    # ==== Função para criar janelas com delay na internação ====
    def create_sequences_duplas(var1, var2, var3, var4, var5, var6, target, seq_len, output_size, delay):
        x1s, x2s, x3s, x4s, x5s, x6s, ys = [], [], [], [], [], [], []
        for i in range(len(target) - seq_len - output_size + 1):
            # --- Internações com delay ---
            past_intern = var1[i:i + seq_len - delay]  # até t-delay
            intern_padded = np.pad(past_intern, (delay, 0), constant_values=0)  # completa últimos 10 com 0

            # --- Exógenas sem delay ---
            x2 = var2[i:i + seq_len]
            x3 = var3[i:i + seq_len]
            x4 = var4[i:i + seq_len]
            x5 = var5[i:i + seq_len]
            x6 = var6[i:i+seq_len]

            # --- Target continua normal ---
            y = target[i + seq_len:i + seq_len + output_size]

            # --- Armazena ---
            x1s.append(intern_padded)
            x2s.append(x2)
            x3s.append(x3)
            x4s.append(x4)
            x5s.append(x5)
            x6s.append(x6)
            ys.append(y)

        return (
            torch.tensor(x1s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
            torch.tensor(x2s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
            torch.tensor(x3s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
            torch.tensor(x4s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
            torch.tensor(x5s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
            torch.tensor(x6s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
            torch.tensor(ys).float()                  # [N, OUTPUT_SIZE]
        )

    # ==== Separação treino/teste ====
    train_size = int(len(data_igr) * TRAIN_TEST_SPLIT)

    x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, y_train = create_sequences_duplas(
        x1_scaled[:train_size],
        x2_scaled[:train_size],
        x3_scaled[:train_size],
        x4_scaled[:train_size],
        x5_scaled[:train_size],
        x6_scaled[:train_size],
        x1_scaled[:train_size],
        SEQ_LEN,
        output_size=OUTPUT_SIZE,
        delay=DELAY_SIZE
    )

    x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, y_test = create_sequences_duplas(
        x1_scaled[train_size - SEQ_LEN:],
        x2_scaled[train_size - SEQ_LEN:],
        x3_scaled[train_size - SEQ_LEN:],
        x4_scaled[train_size - SEQ_LEN:],
        x5_scaled[train_size - SEQ_LEN:],
        x6_scaled[train_size - SEQ_LEN:],
        x1_scaled[train_size - SEQ_LEN:],
        SEQ_LEN,
        output_size=OUTPUT_SIZE,
        delay=DELAY_SIZE
    )

    # ==== Modelo LSTMs ====
    class MultLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(MultLSTM, self).__init__()
            self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers)
            self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers)
            self.lstm3 = nn.LSTM(input_size, hidden_size, num_layers)
            self.lstm4 = nn.LSTM(input_size, hidden_size, num_layers)
            self.lstm5 = nn.LSTM(input_size, hidden_size, num_layers)
            self.lstm6 = nn.LSTM(input_size, hidden_size, num_layers)
            self.fc = nn.Linear(hidden_size * 6, output_size)  # concatenar as duas saídas

        def forward(self, x1, x2, x3, x4, x5, x6):
            out1, _ = self.lstm1(x1)  # shape: (seq_len, batch, hidden)
            out2, _ = self.lstm2(x2)
            out3, _ = self.lstm3(x3)
            out4, _ = self.lstm4(x4)
            out5, _ = self.lstm5(x5)
            out6, _ = self.lstm6(x6)
            last_out1 = out1[-1]  # shape: (batch, hidden)
            last_out2 = out2[-1]
            last_out3 = out3[-1]
            last_out4 = out4[-1]
            last_out5 = out5[-1]
            last_out6 = out6[-1]
            combined = torch.cat((last_out1, last_out2, last_out3, last_out4, last_out5, last_out6), dim=1)
            output = self.fc(combined)
            return output

    model = MultLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ==== Treinamento ====
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        inp1 = x1_train.permute(1, 0, 2)  # [seq_len, batch, 1]
        inp2 = x2_train.permute(1, 0, 2)
        inp3 = x3_train.permute(1, 0, 2)
        inp4 = x4_train.permute(1, 0, 2)
        inp5 = x5_train.permute(1, 0, 2)
        inp6 = x6_train.permute(1, 0, 2)
        output = model(inp1, inp2, inp3, inp4, inp5, inp6)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        # ==== Avaliação ====
        model.eval()
        with torch.no_grad():
            pred_train = model(x1_train.permute(1, 0, 2), x2_train.permute(1, 0, 2), x3_train.permute(1, 0, 2), x4_train.permute(1, 0, 2), x5_train.permute(1, 0, 2), x6_train.permute(1, 0, 2)).squeeze().numpy()
            pred_test = model(x1_test.permute(1, 0, 2), x2_test.permute(1, 0, 2), x3_test.permute(1, 0, 2), x4_test.permute(1, 0, 2), x5_test.permute(1, 0, 2), x6_test.permute(1, 0, 2)).squeeze().numpy()
            true_train = y_train.squeeze().numpy()
            true_test = y_test.squeeze().numpy()

        # ==== Cálculo do erro de teste ====
        final_loss = criterion(torch.tensor(pred_test), torch.tensor(true_test)).item()
        test_loss_list.append(final_loss)
        rmse = np.sqrt(final_loss)
        rmse_list.append(rmse)
        r2 = r2_score(true_test, pred_test)
        r2_list.append(r2)
        mae = mean_absolute_error(true_test, pred_test)
        mae_list.append(mae)

    min_rmse = min(rmse_list)
    min_rmse_epoch = rmse_list.index(min_rmse)
    min_mse = test_loss_list[min_rmse_epoch]
    min_r2 = r2_list[min_rmse_epoch]
    min_mae = mae_list[min_rmse_epoch]

    background = [
        x1_train,
        x2_train,
        x3_train,
        x4_train,
        x5_train,
        x6_train
    ]

    # Número de amostras que queremos explicar
    num_amostras = 50  

    # Limitar ao número disponível
    num_amostras = min(num_amostras, x1_test.shape[0])

    # Selecionando múltiplas amostras de cada entrada
    X = [
        x1_test[:num_amostras],
        x2_test[:num_amostras],
        x3_test[:num_amostras],
        x4_test[:num_amostras],
        x5_test[:num_amostras],
        x6_test[:num_amostras]
    ]
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X)

    metrics_comparison.append({
        'igr': igr_i,
        'model': model_name,
        'network_architecture': '6 LSTMs',
        'features': var_total,
        'RMSE': f"{min_rmse:.4f}",
        'MSE': f"{min_mse:.4f}",
        'R2': f"{min_r2:.4f}",
        'MAE': f"{min_mae:.4f}",
        'epoch': EPOCHS,
        'n_layers': NUM_LAYERS,
        'hidden_size': HIDDEN_SIZE,
        'seq_len': SEQ_LEN,
        'shap_values': shap_values,
    })

# COMMAND ----------

# Dataset creation
df_metrics_final = pd.DataFrame(metrics_comparison)
df_metrics_final.sort_values(by='igr', axis=0, inplace=True)

# Calculate the mean SHAP values for each variable
def calculate_mean_shap(x):
    shap_values = x
    shap_array = np.stack(shap_values, axis=-1)
    mean_shap = np.abs(shap_array).mean(axis=(0,1,2,3))  # shape: (6,)
    return mean_shap

df_metrics_final["mean_shap"] = df_metrics_final["shap_values"].apply(calculate_mean_shap)

# Saving file
df_metrics_final.to_excel(f'SHAP_with_delay_6LSTMs_{seed_number}.xlsx', index=False)

df_metrics_final