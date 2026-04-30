# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import copy
import os

# COMMAND ----------

data_union = pd.read_csv('../../2_Data_Union/ready_data_all_features.csv')
data_union

# COMMAND ----------

# Criar série temporal sintética com 4 ciclos idênticos, cada ciclo com uma onda (tipo senoide) seguida de valor constante baixo

num_points = 200
num_cycles = 4
cycle_length = num_points // num_cycles
wave_length = int(cycle_length * 0.3)  # 30% do ciclo é a onda

synthetic_series = []

for _ in range(num_cycles):
    # Onda senoide (calombo)
    x_wave = np.linspace(0, np.pi, wave_length)
    wave = np.sin(x_wave) * 10 + 5  # amplitude 10, deslocamento 5
    # Constante baixa
    const = np.full(cycle_length - wave_length, 2)
    # Ciclo completo
    cycle = np.concatenate([wave, const])
    synthetic_series.extend(cycle)

synthetic_series = np.array(synthetic_series)

plt.figure(figsize=(12, 6))
plt.plot(synthetic_series)
plt.title('Série temporal sintética com 4 ciclos idênticos')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

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

num_igr = len(data['igr'].unique())
fig, axes = plt.subplots(num_igr, 1, figsize=(14, 4 * num_igr), sharex=True)

for idx, igr in enumerate(data['igr'].unique()):
    igr_data = data[data['igr'] == igr]
    axes[idx].plot(igr_data['date_week'], igr_data['count_hospitalization'])
    axes[idx].set_title(f'IGR: {igr}')
    axes[idx].set_ylabel('Rate Access per Physician')
    axes[idx].grid(True)

axes[-1].set_xlabel('Data')
plt.tight_layout()
plt.show()

# COMMAND ----------

# ==== Parameters ====
SEQ_LEN = 24
RANGE_SEQ_LEN = [24]

HIDDEN_SIZE = 40
RANGE_HIDDEN_SIZE = [20, 40]

NUM_LAYERS = 2
RANGE_NUM_LAYERS = [1, 2, 3, 4]

OUTPUT_SIZE = 8
RANGE_OUTPUT_SIZE = [8]

EPOCHS = 3000
LR = 0.001

TRAIN_TEST_SPLIT = 0.70

# COMMAND ----------

# Fixar seed para reprodutibilidade. Escolher seed aleatória
#seed_number = random.randint(0, 10000)
seed_number = 0
print(seed_number)


model_name = 'LSTM'
metrics_comparison = []

# COMMAND ----------

# Only sih data. Reference.

igr_i = 'Uberlândia'

torch.manual_seed(seed_number)
np.random.seed(seed_number)

loss_list = []
test_loss_list = []

mse_list = []
rmse_list = []  
r2_list = []
mae_list = []
pred_test_list = []
true_test_list =[]

data_igr = data[data['igr'] == igr_i]
data_igr

# COMMAND ----------

target = ['count_hospitalization']
var_total = target

# ==== Data ====
# x1_seq = synthetic_series.astype(np.float32)
x1_seq = data_igr['count_hospitalization'].values.astype(np.float32)
x1_seq

# COMMAND ----------

plt.figure(figsize=(12, 6))
plt.plot(x1_seq)
plt.title('Plot de x1_seq')
plt.xlabel('Índice')
plt.ylabel('count_hospitalization')
plt.grid(True)
plt.show()

# COMMAND ----------

# ==== Scaler ====
scaler_x1 = MinMaxScaler()
x1_scaled = scaler_x1.fit_transform(x1_seq.reshape(-1, 1)).ravel()

# COMMAND ----------

plt.figure(figsize=(12, 6))
plt.plot(x1_scaled)
plt.title('Plot de x1_scaled')
plt.xlabel('Índice')
plt.ylabel('count_hospitalization')
plt.grid(True)
plt.show()

# COMMAND ----------

# ==== Function to create windows ====
def create_sequences_duplas(var1, target, seq_len, output_size):
    x1s, ys = [], []
    for i in range(len(target) - seq_len- output_size + 1):
        x1 = var1[i:i+seq_len]
        y = target[i+seq_len:i+seq_len+output_size]
        x1s.append(x1)
        ys.append(y)
    return (
        torch.tensor(x1s).unsqueeze(-1).float(),  # [N, SEQ_LEN, 1]
        torch.tensor(ys).float()    # [N, 1]
    )

# ==== Divisions train/test ====
train_size = int(len(data_igr) * TRAIN_TEST_SPLIT)

x1_train, y_train = create_sequences_duplas(
    x1_scaled[:train_size],
    x1_scaled[:train_size],
    SEQ_LEN,
    output_size=OUTPUT_SIZE
)

x1_test, y_test = create_sequences_duplas(
    x1_scaled[train_size - SEQ_LEN:],
    x1_scaled[train_size - SEQ_LEN:],
    SEQ_LEN,
    output_size=OUTPUT_SIZE
)

# COMMAND ----------

display(x1_train.squeeze().numpy())

# COMMAND ----------

plt.figure(figsize=(12, 6))
plt.plot(x1_train.squeeze().numpy()[0], label='x1_train')
plt.plot(y_train.squeeze().numpy()[0], label='y_train')
plt.title('Exemplo de x1_train e y_train')
plt.xlabel('Índice')
plt.ylabel('Valor Normalizado')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

plt.figure(figsize=(12, 6))
plt.plot(x1_train.squeeze().numpy()[1], label='x1_train')
plt.plot(y_train.squeeze().numpy()[1], label='y_train')
plt.title('Exemplo de x1_train e y_train')
plt.xlabel('Índice')
plt.ylabel('Valor Normalizado')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# ==== Model ====
class MultLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size * 1, output_size)  # concatenar as duas saídas

    def forward(self, x1):
        out1, _ = self.lstm1(x1)  # shape: (seq_len, batch, hidden)
        last_out1 = out1[-1]  # shape: (batch, hidden)
        output = self.fc(last_out1)
        return output

model = MultLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
)

# ==== Training ====
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    inp1 = x1_train.permute(1, 0, 2)  # [seq_len, batch, 1]
    output = model(inp1)
    loss = criterion(output, y_train)

    loss.backward()
    optimizer.step()

    # ==== Testing ====
    model.eval()
    with torch.no_grad():
        pred_train = model(x1_train.permute(1, 0, 2)).squeeze().numpy()
        pred_test = model(x1_test.permute(1, 0, 2)).squeeze().numpy()
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
    pred_test_list.append(pred_test)
    true_test_list.append(true_test)

    # ==== LR Scheduler ====
    scheduler.step(final_loss)

min_rmse = min(rmse_list)
min_rmse_epoch = rmse_list.index(min_rmse)
min_mse = test_loss_list[min_rmse_epoch]
min_r2 = r2_list[min_rmse_epoch]
min_mae = mae_list[min_rmse_epoch]
min_pred_test = pred_test_list[min_rmse_epoch]
min_true_test = true_test_list[min_rmse_epoch]

metrics_comparison.append({
    'igr': igr_i,
    'model': model_name,
    'network_architecture': '1 LSTM',
    'features': var_total,
    'true_test': min_true_test,
    'pred_test': min_pred_test,
    'RMSE': f"{min_rmse:.4f}",
    'MSE': f"{min_mse:.4f}",
    'R2': f"{min_r2:.4f}",
    'MAE': f"{min_mae:.4f}",
    'epoch': min_rmse_epoch,
    'n_layers': NUM_LAYERS,
    'hidden_size': HIDDEN_SIZE,
    'seq_len': SEQ_LEN
})
#print(f"\nMetrics: {metrics_comparison}")

# COMMAND ----------

plt.plot(test_loss_list)

# COMMAND ----------

print(min_rmse_epoch)

# COMMAND ----------

min_pred_test[0]

# COMMAND ----------

posicao = 31
plt.plot(min_pred_test[posicao], label='Predição')
plt.plot(min_true_test[posicao], label='Verdadeiro')
plt.legend()

# COMMAND ----------

# Save all results
try:
    df_metrics_final = pd.DataFrame(metrics_comparison)

    df_metrics_final.sort_values(by='igr', axis=0, inplace=True)

    df_metrics_final.to_excel(f"unico_result_lstm_with_delay_{seed_number}_70.xlsx", index=False)

    print(f"✔️ Saved!")
    print(f"🔢 Total of models salved: {df_metrics_final.shape[0]}")

except Exception as e:
    print(e)
    print("❌ Error to save final metrics.")

# COMMAND ----------

# Save the best results
try:
    df_metrics_final['RMSE'] = df_metrics_final['RMSE'].astype(float)
    best_each_type = df_metrics_final.loc[
        df_metrics_final.groupby(['igr', 'network_architecture'])['RMSE'].idxmin()
    ]

    best_each_type.to_excel(f"best_lstm_with_delay_{seed_number}_70.xlsx", index=False)

    print(f"✔️ Saved!")
    print(f"🔢 Total of models salved: {best_each_type.shape[0]}")

except Exception as e:
    print(e)
    print("❌ Error to save final metrics.")