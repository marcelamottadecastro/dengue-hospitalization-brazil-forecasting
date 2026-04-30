# Databricks notebook source
!pip install openpyxl
%restart_python 

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import copy
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importação dos dados

# COMMAND ----------

data_union = pd.read_csv('../2_Data_Union/ready_data_all_features.csv')
data_union

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization

# COMMAND ----------

# # Filtrando somente as microrregiões com soma no número de internações > 800
# filtro = data[['NOME_MICRORREGIAO', 'NUMERO_INTERNACAO']].groupby('NOME_MICRORREGIAO').sum()
# micros_filtradas = filtro[filtro['NUMERO_INTERNACAO'] > 800].index
# data = data[data['NOME_MICRORREGIAO'].isin(micros_filtradas)]
# data[['NOME_MICRORREGIAO', 'NUMERO_INTERNACAO']].groupby('NOME_MICRORREGIAO').sum().sort_values('NUMERO_INTERNACAO', ascending=False)

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
data

# COMMAND ----------

# MAGIC %md
# MAGIC ### LSTM

# COMMAND ----------

# ==== Parameters ====
SEQ_LEN = 24
RANGE_SEQ_LEN = [24]

HIDDEN_SIZE = 40
#RANGE_HIDDEN_SIZE = [20, 40]
RANGE_HIDDEN_SIZE = [40]

NUM_LAYERS = 2
#RANGE_NUM_LAYERS = [1, 2, 3, 4]
RANGE_NUM_LAYERS = [2]

OUTPUT_SIZE = 8
RANGE_OUTPUT_SIZE = [8]

EPOCHS = 200
LR = 0.01

# COMMAND ----------

# # EXEMPLO COM GRÁFICO

# torch.manual_seed(0)
# np.random.seed(0)

# loss_list = []
# test_loss_list = []

# micro_i = 'Porto Alegre'

# data_micro = data[data['NOME_MICRORREGIAO'] == micro_i]
# data_micro

# var_exogenicas = ['NUMERO_ACESSO']
# var_algo = ['NUMERO_INTERNACAO']
# var_total = var_exogenicas + var_algo

# serie = data_micro['NUMERO_INTERNACAO'].values.astype(np.float32)

# # ==== Função para criar janelas ====
# def create_sequences(data, seq_len):
#     xs, ys = [], []
#     for i in range(len(data) - seq_len):
#         x = data[i:i+seq_len]
#         y = data[i+seq_len]
#         xs.append(x)
#         ys.append(y)
#     return torch.tensor(xs).unsqueeze(-1), torch.tensor(ys).unsqueeze(-1)

# # ==== Separação treino/teste ====
# train_size = int(len(serie) * 0.8)
# train_data = serie[:train_size]
# test_data = serie[train_size - SEQ_LEN:]  # inclui janela anterior

# X_train, y_train = create_sequences(train_data, SEQ_LEN)
# X_test, y_test = create_sequences(test_data, SEQ_LEN)

# # ==== Modelo ====
# class LSTMModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[-1])
#         return out

# model = LSTMModel()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # ==== Treinamento ====
# for epoch in range(EPOCHS):
#     model.train()
#     optimizer.zero_grad()
    
#     inputs = X_train.permute(1, 0, 2)
#     output = model(inputs)
#     loss = criterion(output, y_train)
#     loss_list.append(loss.item())
    
#     loss.backward()
#     optimizer.step()
    
#     #if (epoch+1) % 20 == 0:
#     print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

#     # ==== Avaliação ====
#     model.eval()
#     with torch.no_grad():
#         pred_train = model(X_train.permute(1, 0, 2)).squeeze().numpy()
#         pred_test = model(X_test.permute(1, 0, 2)).squeeze().numpy()
#         true_train = y_train.squeeze().numpy()
#         true_test = y_test.squeeze().numpy()

#     # ==== Cálculo do erro de teste ====
#     test_loss = criterion(torch.tensor(pred_test), torch.tensor(true_test)).item()
#     test_loss_list.append(test_loss)
#     rmse = np.sqrt(test_loss)


# print(f"\nErro no conjunto de TESTE:")
# print(f"MSE: {test_loss:.4f}")
# print(f"RMSE: {rmse:.4f}")


# # ==== Visualização ====
# plt.figure(figsize=(12, 5))

# plt.plot(range(len(true_train)), true_train, label='Treino - Real', color='blue')
# plt.plot(range(len(pred_train)), pred_train, label='Treino - Previsto', linestyle='--', color='skyblue')

# test_start = len(true_train)
# plt.plot(range(test_start, test_start + len(true_test)), true_test, label='Teste - Real', color='green')
# plt.plot(range(test_start, test_start + len(pred_test)), pred_test, label='Teste - Previsto', linestyle='--', color='lightgreen')

# plt.axvline(test_start, color='gray', linestyle=':', label='Divisão treino/teste')
# plt.title('Previsão de Internações com LSTM')
# plt.xlabel('Semanas')
# plt.ylabel('Internações')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# COMMAND ----------

# print(f"Mínimo test_loss: {min(test_loss_list)}")

# # ==== Visualização ====
# plt.figure(figsize=(12, 5))

# plt.plot(loss_list, label='Treino')
# plt.plot(test_loss_list, label='Teste')
# plt.title('Erro de teste por época')
# plt.xlabel('Época')
# plt.ylabel('Erro')
# plt.legend()
# plt.grid(True)
# plt.show()

# COMMAND ----------

# ## EXEMPLO COM GRÁFICO

# torch.manual_seed(0)
# np.random.seed(0)

# loss_list = []
# test_loss_list = []

# micro_i = 'Porto Alegre'

# data_micro = data[data['NOME_MICRORREGIAO'] == micro_i]
# data_micro

# var_exogenicas = ['NUMERO_ACESSO']
# var_algo = ['NUMERO_INTERNACAO']
# var_total = var_exogenicas + var_algo

# # ==== Dados ====
# internacoes_seq = data_micro['NUMERO_INTERNACAO'].values.astype(np.float32)
# temperatura_seq = data_micro['NUMERO_ACESSO'].values.astype(np.float32)

# # ==== Função para criar janelas ====
# def create_sequences_duplas(var1, var2, target, seq_len):
#     x1s, x2s, ys = [], [], []
#     for i in range(len(target) - seq_len):
#         x1 = var1[i:i+seq_len]
#         x2 = var2[i:i+seq_len]
#         y = target[i+seq_len]
#         x1s.append(x1)
#         x2s.append(x2)
#         ys.append(y)
#     return (
#         torch.tensor(x1s).unsqueeze(-1),  # [N, SEQ_LEN, 1]
#         torch.tensor(x2s).unsqueeze(-1),  # [N, SEQ_LEN, 1]
#         torch.tensor(ys).unsqueeze(-1)    # [N, 1]
#     )

# # ==== Separação treino/teste ====
# train_size = int(len(df) * 0.8)

# x1_train, x2_train, y_train = create_sequences_duplas(
#     internacoes_seq[:train_size],
#     temperatura_seq[:train_size],
#     internacoes_seq[:train_size],
#     SEQ_LEN
# )

# x1_test, x2_test, y_test = create_sequences_duplas(
#     internacoes_seq[train_size - SEQ_LEN:],
#     temperatura_seq[train_size - SEQ_LEN:],
#     internacoes_seq[train_size - SEQ_LEN:],
#     SEQ_LEN
# )

# # ==== Modelo com duas LSTMs ====
# class DualLSTM(nn.Module):
#     def __init__(self, hidden_size, num_layers):
#         super(DualLSTM, self).__init__()
#         self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
#         self.lstm2 = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
#         self.fc = nn.Linear(hidden_size * 2, 1)  # concatenar as duas saídas

#     def forward(self, x1, x2):
#         out1, _ = self.lstm1(x1)  # shape: (seq_len, batch, hidden)
#         out2, _ = self.lstm2(x2)
#         last_out1 = out1[-1]  # shape: (batch, hidden)
#         last_out2 = out2[-1]
#         combined = torch.cat((last_out1, last_out2), dim=1)
#         output = self.fc(combined)
#         return output

# model = DualLSTM(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # ==== Treinamento ====
# for epoch in range(EPOCHS):
#     model.train()
#     optimizer.zero_grad()

#     inp1 = x1_train.permute(1, 0, 2)  # [seq_len, batch, 1]
#     inp2 = x2_train.permute(1, 0, 2)
#     output = model(inp1, inp2)
#     loss = criterion(output, y_train)
#     loss_list.append(loss.item())

#     loss.backward()
#     optimizer.step()

#     #if (epoch+1) % 20 == 0:
#     print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

#     # ==== Avaliação ====
#     model.eval()
#     with torch.no_grad():
#         pred_train = model(x1_train.permute(1, 0, 2), x2_train.permute(1, 0, 2)).squeeze().numpy()
#         pred_test = model(x1_test.permute(1, 0, 2), x2_test.permute(1, 0, 2)).squeeze().numpy()
#         true_train = y_train.squeeze().numpy()
#         true_test = y_test.squeeze().numpy()

#     # ==== Cálculo do erro de teste ====
#     test_loss = criterion(torch.tensor(pred_test), torch.tensor(true_test)).item()
#     test_loss_list.append(test_loss)
#     print(f"Test_loss: {test_loss:.4f}")

#     rmse = np.sqrt(test_loss)

# print(f"\nErro no conjunto de TESTE:")
# print(f"MSE: {test_loss:.4f}")
# print(f"RMSE: {rmse:.4f}")

# # ==== Visualização ====
# plt.figure(figsize=(12, 5))

# plt.plot(range(len(true_train)), true_train, label='Treino - Real', color='blue')
# plt.plot(range(len(pred_train)), pred_train, label='Treino - Previsto', linestyle='--', color='skyblue')

# test_start = len(true_train)
# plt.plot(range(test_start, test_start + len(true_test)), true_test, label='Teste - Real', color='green')
# plt.plot(range(test_start, test_start + len(pred_test)), pred_test, label='Teste - Previsto', linestyle='--', color='lightgreen')

# plt.axvline(test_start, color='gray', linestyle=':', label='Divisão treino/teste')
# plt.title('Previsão com 2 LSTMs (uma por variável)')
# plt.xlabel('Semanas')
# plt.ylabel('Internações')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# COMMAND ----------

# print(f"Mínimo test_loss: {min(test_loss_list)}")

# # ==== Visualização ====
# plt.figure(figsize=(12, 5))

# plt.plot(loss_list, label='Treino')
# plt.plot(test_loss_list, label='Teste')
# plt.title('Erro de teste por época')
# plt.xlabel('Época')
# plt.ylabel('Erro')
# plt.legend()
# plt.grid(True)
# plt.show()

# COMMAND ----------

# Fixar seed para reprodutibilidade
torch.manual_seed(0)
np.random.seed(0)

model_name = 'LSTM'
metrics_comparison = []


# COMMAND ----------

# # Somente dados de internação. Referência.

# for HIDDEN_SIZE in RANGE_HIDDEN_SIZE:
#     for SEQ_LEN in RANGE_SEQ_LEN:
#         for NUM_LAYERS in RANGE_NUM_LAYERS:
#             for micro_i in data['NOME_MICRORREGIAO'].unique():

#                 torch.manual_seed(0)
#                 np.random.seed(0)

#                 rmse_list = []   

#                 data_micro = data[data['NOME_MICRORREGIAO'] == micro_i]
#                 data_micro

#                 var_exogenicas = ['NUMERO_ACESSO', 'TEMPERATURA_MEDIA_SEMANAL']
#                 var_algo = ['NUMERO_INTERNACAO']
#                 var_total = var_exogenicas + var_algo

#                 serie = data_micro['NUMERO_INTERNACAO'].values.astype(np.float32)

#                 # ==== Função para criar janelas ====
#                 def create_sequences(data, seq_len):
#                     xs, ys = [], []
#                     for i in range(len(data) - seq_len):
#                         x = data[i:i+seq_len]
#                         y = data[i+seq_len]
#                         xs.append(x)
#                         ys.append(y)
#                     return torch.tensor(xs).unsqueeze(-1), torch.tensor(ys).unsqueeze(-1)

#                 # ==== Separação treino/teste ====
#                 train_size = int(len(serie) * 0.8)
#                 train_data = serie[:train_size]
#                 test_data = serie[train_size - SEQ_LEN:]  # inclui janela anterior

#                 X_train, y_train = create_sequences(train_data, SEQ_LEN)
#                 X_test, y_test = create_sequences(test_data, SEQ_LEN)

#                 # ==== Modelo ====
#                 class LSTMModel(nn.Module):
#                     def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
#                         super(LSTMModel, self).__init__()
#                         self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
#                         self.fc = nn.Linear(hidden_size, 1)

#                     def forward(self, x):
#                         out, _ = self.lstm(x)
#                         out = self.fc(out[-1])
#                         return out

#                 model = LSTMModel()
#                 criterion = nn.MSELoss()
#                 optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#                 # ==== Treinamento ====
#                 for epoch in range(EPOCHS):
#                     model.train()
#                     optimizer.zero_grad()
                    
#                     inputs = X_train.permute(1, 0, 2)
#                     output = model(inputs)
#                     loss = criterion(output, y_train)
                    
#                     loss.backward()
#                     optimizer.step()


#                     # ==== Avaliação ====
#                     model.eval()
#                     with torch.no_grad():
#                         pred_train = model(X_train.permute(1, 0, 2)).squeeze().numpy()
#                         pred_test = model(X_test.permute(1, 0, 2)).squeeze().numpy()
#                         true_train = y_train.squeeze().numpy()
#                         true_test = y_test.squeeze().numpy()


#                     # ==== Cálculo do erro de teste ====
#                     test_loss = criterion(torch.tensor(pred_test), torch.tensor(true_test)).item()
#                     rmse = np.sqrt(test_loss)
#                     rmse_list.append(rmse)

#                 min_rmse = min(rmse_list)
#                 min_rmse_epoch = rmse_list.index(min_rmse)

#                 metricas_comparacao.append({
#                     'microrregiao': micro_i,
#                     'modelo': nome_modelo,
#                     'arquitetura_rede': '1 LSTM',
#                     'variaveis': var_algo,
#                     'RMSE': f"{min_rmse:.4f}",
#                     'epoch': min_rmse_epoch,
#                     'n_layers': NUM_LAYERS,
#                     'hidden_size': HIDDEN_SIZE,
#                     'seq_len': SEQ_LEN
#                 })


#                 print(f"\nMétricas: {metricas_comparacao}")

# COMMAND ----------

feature_external_list = [col for col in data.columns if col not in [
    'igr',
    'count_hospitalization',
    'date_week',
    'access_count', 
    'count_physician', 
    'interaction_temperature_precipitation',
    'interaction_temperature_humidity',
    'interaction_precipitation_humidity',
    'interaction_temperature_precipitation_humidity',
    'interaction_temperature_precipitation_above75',
    'interaction_temperature_humidity_above75',
    'interaction_precipitation_humidity_above75',
    'interaction_temperature_precipitation_humidity_above75',
    ]]
feature_external_list

# COMMAND ----------

data.dtypes

# COMMAND ----------

# # Somente 2 variáveis. Testando todas as variáveis de clima.

# for variavel_externa in variavel_externa_lista:
#     for HIDDEN_SIZE in RANGE_HIDDEN_SIZE:
#         for SEQ_LEN in RANGE_SEQ_LEN:
#             for NUM_LAYERS in RANGE_NUM_LAYERS:
#                 for micro_i in data['NOME_MICRORREGIAO'].unique():

#                     torch.manual_seed(0)
#                     np.random.seed(0) 

#                     rmse_list = []

#                     data_micro = data[data['NOME_MICRORREGIAO'] == micro_i]
                    
#                     var_exogenicas = [variavel_externa]
#                     var_algo = ['NUMERO_INTERNACAO']
#                     var_total = var_exogenicas + var_algo

#                     # ==== Dados ====
#                     internacoes_seq = data_micro['NUMERO_INTERNACAO'].values.astype(np.float32)
#                     temperatura_seq = data_micro[variavel_externa].values.astype(np.float32)

#                     # ==== Função para criar janelas ====
#                     def create_sequences_duplas(var1, var2, target, seq_len):
#                         x1s, x2s, ys = [], [], []
#                         for i in range(len(target) - seq_len):
#                             x1 = var1[i:i+seq_len]
#                             x2 = var2[i:i+seq_len]
#                             y = target[i+seq_len]
#                             x1s.append(x1)
#                             x2s.append(x2)
#                             ys.append(y)
#                         return (
#                             torch.tensor(x1s).unsqueeze(-1),  # [N, SEQ_LEN, 1]
#                             torch.tensor(x2s).unsqueeze(-1),  # [N, SEQ_LEN, 1]
#                             torch.tensor(ys).unsqueeze(-1)    # [N, 1]
#                         )

#                     # ==== Separação treino/teste ====
#                     train_size = int(len(data_micro) * 0.8)

#                     x1_train, x2_train, y_train = create_sequences_duplas(
#                         internacoes_seq[:train_size],
#                         temperatura_seq[:train_size],
#                         internacoes_seq[:train_size],
#                         SEQ_LEN
#                     )

#                     x1_test, x2_test, y_test = create_sequences_duplas(
#                         internacoes_seq[train_size - SEQ_LEN:],
#                         temperatura_seq[train_size - SEQ_LEN:],
#                         internacoes_seq[train_size - SEQ_LEN:],
#                         SEQ_LEN
#                     )

#                     # ==== Modelo com duas LSTMs ====
#                     class DualLSTM(nn.Module):
#                         def __init__(self, hidden_size, num_layers):
#                             super(DualLSTM, self).__init__()
#                             self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
#                             self.lstm2 = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
#                             self.fc = nn.Linear(hidden_size * 2, 1)  # concatenar as duas saídas

#                         def forward(self, x1, x2):
#                             out1, _ = self.lstm1(x1)  # shape: (seq_len, batch, hidden)
#                             out2, _ = self.lstm2(x2)
#                             last_out1 = out1[-1]  # shape: (batch, hidden)
#                             last_out2 = out2[-1]
#                             combined = torch.cat((last_out1, last_out2), dim=1)
#                             output = self.fc(combined)
#                             return output

#                     model = DualLSTM(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
#                     criterion = nn.MSELoss()
#                     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#                     # ==== Treinamento ====
#                     for epoch in range(EPOCHS):
#                         model.train()
#                         optimizer.zero_grad()

#                         inp1 = x1_train.permute(1, 0, 2)  # [seq_len, batch, 1]
#                         inp2 = x2_train.permute(1, 0, 2)
#                         output = model(inp1, inp2)
#                         loss = criterion(output, y_train)

#                         loss.backward()
#                         optimizer.step()

#                         # ==== Avaliação ====
#                         model.eval()
#                         with torch.no_grad():
#                             pred_train = model(x1_train.permute(1, 0, 2), x2_train.permute(1, 0, 2)).squeeze().numpy()
#                             pred_test = model(x1_test.permute(1, 0, 2), x2_test.permute(1, 0, 2)).squeeze().numpy()
#                             true_train = y_train.squeeze().numpy()
#                             true_test = y_test.squeeze().numpy()

#                         # ==== Cálculo do erro de teste ====
#                         final_loss = criterion(torch.tensor(pred_test), torch.tensor(true_test)).item()
#                         rmse = np.sqrt(final_loss)
#                         rmse_list.append(rmse)

#                     min_rmse = min(rmse_list)
#                     min_rmse_epoch = rmse_list.index(min_rmse)

#                     metricas_comparacao.append({
#                         'microrregiao': micro_i,
#                         'modelo': nome_modelo,
#                         'arquitetura_rede': '2 LSTMs',
#                         'variaveis': var_total,
#                         'RMSE': f"{min_rmse:.4f}",
#                         'epoch': min_rmse_epoch,
#                         'n_layers': NUM_LAYERS,
#                         'hidden_size': HIDDEN_SIZE,
#                         'seq_len': SEQ_LEN
#                     })
#                     print(f"\nMétricas: {metricas_comparacao}")

# COMMAND ----------

# Somente 2 variáveis. Testando todas as variáveis de clima.

for feature_external in feature_external_list:
    for HIDDEN_SIZE in RANGE_HIDDEN_SIZE:
        for SEQ_LEN in RANGE_SEQ_LEN:
            for NUM_LAYERS in RANGE_NUM_LAYERS:
                for igr_i in data['igr'].unique():

                    torch.manual_seed(0)
                    np.random.seed(0)

                    loss_list = []
                    test_loss_list = []

                    mse_list = []
                    rmse_list = []  
                    r2_list = []
                    mae_list = []

                    data_igr = data[data['igr'] == igr_i]
                    
                    var_exogenicas = [feature_external]
                    var_algo = ['count_hospitalization']
                    var_total = var_algo + var_exogenicas

                    # ==== Data ====
                    x1_seq = data_igr['count_hospitalization'].values.astype(np.float32)
                    x2_seq = data_igr[var_exogenicas[0]].values.astype(np.float32)

                    scaler_x1 = MinMaxScaler()
                    scaler_x2 = MinMaxScaler()
                    x1_scaled = scaler_x1.fit_transform(x1_seq.reshape(-1, 1))
                    x1_scaled = x1_scaled.ravel() # Voltar para o shape original
                    x2_scaled = scaler_x2.fit_transform(x2_seq.reshape(-1, 1))
                    x2_scaled = x2_scaled.ravel()

                    # ==== Function to create windows ====
                    def create_sequences_duplas(var1, var2, target, seq_len, output_size):
                        x1s, x2s, ys = [], [], []
                        for i in range(len(target) - seq_len- output_size + 1):
                            x1 = var1[i:i+seq_len]
                            x2 = var2[i:i+seq_len]
                            y = target[i+seq_len:i+seq_len+output_size]
                            x1s.append(x1)
                            x2s.append(x2)
                            ys.append(y)
                        return (
                            torch.tensor(x1s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
                            torch.tensor(x2s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
                            torch.tensor(ys).float()    # [N, 1]
                        )

                    # ==== Divisions train/test ====
                    train_size = int(len(data_igr) * 0.8)

                    x1_train, x2_train, y_train = create_sequences_duplas(
                        x1_scaled[:train_size],
                        x2_scaled[:train_size],
                        x1_scaled[:train_size],
                        SEQ_LEN,
                        output_size=OUTPUT_SIZE
                    )

                    x1_test, x2_test, y_test = create_sequences_duplas(
                        x1_scaled[train_size - SEQ_LEN:],
                        x2_scaled[train_size - SEQ_LEN:],
                        x1_scaled[train_size - SEQ_LEN:],
                        SEQ_LEN,
                        output_size=OUTPUT_SIZE
                    )

                    # ==== Modelo com duas LSTMs ====
                    class MultLSTM(nn.Module):
                        def __init__(self, input_size, hidden_size, num_layers, output_size):
                            super(MultLSTM, self).__init__()
                            self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers)
                            self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers)
                            self.fc = nn.Linear(hidden_size * 2, output_size)  # concatenar as duas saídas

                        def forward(self, x1, x2):
                            out1, _ = self.lstm1(x1)  # shape: (seq_len, batch, hidden)
                            out2, _ = self.lstm2(x2)
                            last_out1 = out1[-1]  # shape: (batch, hidden)
                            last_out2 = out2[-1]
                            combined = torch.cat((last_out1, last_out2), dim=1)
                            output = self.fc(combined)
                            return output

                    model = MultLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

                    # ==== Training ====
                    for epoch in range(EPOCHS):
                        model.train()
                        optimizer.zero_grad()

                        inp1 = x1_train.permute(1, 0, 2)  # [seq_len, batch, 1]
                        inp2 = x2_train.permute(1, 0, 2)
                        output = model(inp1, inp2)
                        loss = criterion(output, y_train)

                        loss.backward()
                        optimizer.step()

                        # ==== Testing ====
                        model.eval()
                        with torch.no_grad():
                            pred_train = model(x1_train.permute(1, 0, 2), x2_train.permute(1, 0, 2)).squeeze().numpy()
                            pred_test = model(x1_test.permute(1, 0, 2), x2_test.permute(1, 0, 2)).squeeze().numpy()
                            true_train = y_train.squeeze().numpy()
                            true_test = y_test.squeeze().numpy()

                        # ==== Test error ====
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

                    metrics_comparison.append({
                        'igr': igr_i,
                        'model': model_name,
                        'network_architecture': '2 LSTMs',
                        'features': var_total,
                        'RMSE': f"{min_rmse:.4f}",
                        'MSE': f"{min_mse:.4f}",
                        'R2': f"{min_r2:.4f}",
                        'MAE': f"{min_mae:.4f}",
                        'epoch': min_rmse_epoch,
                        'n_layers': NUM_LAYERS,
                        'hidden_size': HIDDEN_SIZE,
                        'seq_len': SEQ_LEN
                    })
                    print(f"\nMetrics: {metrics_comparison}")

# COMMAND ----------

# Salvando resultados
try:
    df_metrics_final = pd.DataFrame(metrics_comparison)

    df_metrics_final.sort_values(by='igr', axis=0, inplace=True)

    df_metrics_final.to_excel("data_feature_selction.xlsx", index=False)

    print(f"✔️ Saved!")
    print(f"🔢 Total of models salved: {df_metrics_final.shape[0]}")

except Exception as e:
    print(e)
    print("❌ Error to save final metrics.")

# COMMAND ----------

df_metrics_final = df_metrics_final.sort_values(by=['igr', 'model', 'network_architecture'])
display(df_metrics_final)

# COMMAND ----------

df_metrics_final['features'] = df_metrics_final['features'].astype(str)
df_metrics_final['RMSE'] = pd.to_numeric(df_metrics_final['RMSE'], errors='coerce')
best_each_type = df_metrics_final.loc[
    df_metrics_final.groupby(['igr', 'network_architecture'])['RMSE'].idxmin()
]
display(best_each_type)


# COMMAND ----------

best_each_type.to_excel('best_data_feature_selction.xlsx', index=False)