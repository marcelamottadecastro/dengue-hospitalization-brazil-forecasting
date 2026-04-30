# Databricks notebook source
!pip install openpyxl
%restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

# COMMAND ----------

today = datetime.today().strftime('%Y_%m_%d')
today

# COMMAND ----------

depara_arquitetura = {
    '1 LSTM': 'Hospitalization',
    '2 LSTMs': 'Hospitalization + Clinical search',
    '5 LSTMs': 'Hospitalization + Climate',
    '6 LSTMs': 'Hospitalization + Clinical search + Climate',
    'SARIMA_UNIVARIADO': 'Sarimax - Hospitalization'
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Funções

# COMMAND ----------

def load_pivot(path, seed):
    """
    - Loads the result table by seed
    - Renames network architectures to article standard
    - Removes Caratinga
    - Selects the best model per microregion
    - Pivots the table
    - Sorts the columns
    - Returns the pivoted table
    """
    try:
        df = pd.read_excel(f"{path}{seed}_70.xlsx")
    except Exception as e:
        print(f"Error loading file: {e}")

    df['network_architecture'] = df['network_architecture'].map(depara_arquitetura)

    # Removing Caratinga
    df = df[df['igr'] != 'Caratinga']

    best_models = df.loc[
        df.groupby(['igr', 'network_architecture'])['RMSE'].idxmin()
    ]

    valid_stations = df['igr'].unique()

    pivoted = (
        best_models.pivot(index="igr", columns="network_architecture", values="RMSE")
        .loc[valid_stations]
    )

    column_order = ['Hospitalization', 'Hospitalization + Clinical search', 'Hospitalization + Climate', 'Hospitalization + Clinical search + Climate']
    pivoted = pivoted[column_order]
    
    return pivoted

# COMMAND ----------

def calculate_means_and_stddevs(stacked, pivoted):
    # Calculates mean and standard deviation along the "experiment axis"
    mean = stacked.mean(axis=0)
    stddev = stacked.std(axis=0, ddof=1)  # ddof=1 for sample stddev

    # Create DataFrames for mean and stddev
    df_mean = pd.DataFrame(mean, index=pivoted.index, columns=pivoted.columns)
    df_stddev = pd.DataFrame(stddev, index=pivoted.index, columns=pivoted.columns)

    # Combine mean and stddev into a single DataFrame
    df_result = df_mean.round(3).astype(str) + " ± " + df_stddev.round(3).astype(str)

    # Rename the index
    df_result.index.name = "Immediate Geographic Region"       

    return df_result

# COMMAND ----------

def t_stat_test(stacked, pivoted):
    results = []
    columns = pivoted[0].columns

    # Loop over each row
    for i, row in enumerate(pivoted[0].index):
        col_A = stacked[:, i, 0]  # hospitalization column
        
        # Compare A vs each of the other columns
        for j in range(1, len(columns)):  # start from index 1 (other columns)
            col_j = stacked[:, i, j]

            # Shapiro-Wilk
            stat, p = stats.shapiro(col_A - col_j)
            if p > 0.05:
                shapiro = 'Yes'
                t_stat, p_value = stats.ttest_rel(col_A, col_j) # Paired t-test
            else:
                shapiro = 'No'
                t_stat, p_value = stats.wilcoxon(col_A, col_j) # Wilcoxon signed-rank test
            
            results.append({
                "Immediate Geographic Region": row,
                "Comparison between models": f"{columns[0]} vs {columns[j]}",
                "normality_shapiro_wilk": shapiro,
                "t-statistic": t_stat,
                "p-value": p_value,
                "position_1": 0,
                "position_2": j
            })

    # DataFrame with results
    f_testT = pd.DataFrame(results)

    df_testT_significant = f_testT[f_testT['p-value'] < 0.05].sort_values('Comparison between models').round(3)

    df_testT_significant['asterisks'] = df_testT_significant['p-value'].apply(lambda x: '***' if x <= 0.001 else '**' if x <= 0.01 else '*' if x < 0.05 else '')

    significant_comparisons = (
        df_testT_significant.sort_values('position_2').groupby('Immediate Geographic Region')
        .apply(lambda x: [(int(a), int(b), s) for a, b, s in zip(x['position_1'], x['position_2'], x['asterisks'])])
        .to_dict()
    )

    return significant_comparisons, df_testT_significant[['Immediate Geographic Region', 'Comparison between models', 't-statistic', 'p-value']].sort_values('Immediate Geographic Region')

# COMMAND ----------

def create_boxplots(pivoted, stacked, significant_comparisons):
    # Class names and igrs
    classes = pivoted.columns
    igrs = pivoted.index

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 9, figsize=(22, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(igrs):
            igr_data = stacked[:, i, :]

            # Create colored boxplots by class
            bp = ax.boxplot(
                [igr_data[:, j] for j in range(len(classes))],
                patch_artist=True,
                medianprops=dict(color="black"),
            )

            # Apply consistent colors
            for patch, color in zip(bp['boxes'], [f"C{j}" for j in range(len(classes))]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Titles and adjustments
            ax.set_title(igrs[i])
            ax.set_ylabel('RMSE')
            ax.set_xticks([])  # remove x-axis labels
            ax.set_xlabel("")  # remove x-axis name

            # === Add significance bars ===
            if igrs[i] in significant_comparisons:
                y_max = np.max(igr_data)
                h = y_max * 0.08  # bar height
                offset = 0
                for (x1, x2, sig) in significant_comparisons[igrs[i]]:
                    y = y_max + h + offset
                    ax.plot([x1+1, x1+1, x2+1, x2+1], 
                            [y, y+h*0.3, y+h*0.3, y], 
                            lw=1.5, c='k')
                    ax.text((x1+x2)/2 + 1, 
                            y+h*0.4, 
                            sig, 
                            ha='center', 
                            va='bottom', 
                            fontsize=11)
                    offset += h * 1.6  # raise the next bar if there is more than one

                # --- Automatic Y-axis limit adjustment ---
                y_lim = y_max + offset + h * 2
                ax.set_ylim(top=y_lim)
        else:
            ax.axis('off')

    # Adjust spacing between plots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, top=0.95)

    # Create global legend
    handles = [
        plt.Line2D([0], [0], color=f"C{j}", lw=8, label=classe, alpha=0.6)
        for j, classe in enumerate(classes)
    ]
    fig.legend(
        handles=handles,
        title="Legend",
        loc='lower center',
        ncol=len(classes),
        bbox_to_anchor=(0.5, 1.02)
    )

    # plt.savefig(f'Graficos artigo 2/boxplot_rmse_triplicata_sem_delay_{today}.png', 
    #             dpi=600, 
    #             bbox_inches='tight')

    return plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LSTM

# COMMAND ----------

path_without_delay = '../4_Model_Training/Without_Delay/best_lstm_without_delay_'
path_with_delay = '../4_Model_Training/With_Delay/best_lstm_with_delay_'

# List of seeds to analyze
seeds_without_delay = ['6703', '759', '5555']
seeds_with_delay = ['9698', '2108', '3201']

pivoted_result = load_pivot(path_without_delay, '6703')

# Load and stack all tables
pivoted_without_delay = [load_pivot(path_without_delay, seed) for seed in seeds_without_delay]
pivoted_with_delay = [load_pivot(path_with_delay, seed) for seed in seeds_with_delay]

# Stack into a 3D array: (experiments x rows x columns)
stacked_without_delay = np.stack([p.values for p in pivoted_without_delay])
stacked_with_delay = np.stack([p.values for p in pivoted_with_delay])
print(stacked_without_delay.shape)
print(stacked_with_delay.shape)

# COMMAND ----------

df_resultado_without_delay = calculate_means_and_stddevs(
    stacked = stacked_without_delay,
    pivoted = pivoted_result
    )
df_resultado_without_delay

# COMMAND ----------

# df_resultado_without_delay.to_excel(f"result_lstm_without_delay.xlsx")

# COMMAND ----------

df_resultado_with_delay = calculate_means_and_stddevs(
    stacked = stacked_with_delay,
    pivoted = pivoted_result
    )
df_resultado_with_delay

# COMMAND ----------

# df_resultado_with_delay.to_excel("result_lstm_with_delay.xlsx")

# COMMAND ----------

significant_comparisons_without_delay, df_testT_significant_without_delay = t_stat_test(
    stacked = stacked_without_delay,
    pivoted = pivoted_without_delay
)

df_testT_significant_without_delay

# COMMAND ----------

df_testT_significant_without_delay.to_excel("testT_lstm_without_delay.xlsx", index=False)

# COMMAND ----------

significant_comparisons_with_delay, df_testT_significant_with_delay = t_stat_test(
    stacked = stacked_with_delay,
    pivoted = pivoted_with_delay
)

df_testT_significant_with_delay

# COMMAND ----------

df_testT_significant_with_delay.to_excel("testT_lstm_with_delay.xlsx", index=False)

# COMMAND ----------

create_boxplots(pivoted = pivoted_result, 
                stacked = stacked_without_delay, 
                significant_comparisons = significant_comparisons_without_delay
                )

# COMMAND ----------

create_boxplots(pivoted = pivoted_result, 
                stacked = stacked_with_delay, 
                significant_comparisons = significant_comparisons_with_delay
                )

# COMMAND ----------

def create_boxplots_two_groups(
    pivoted_A, stacked_A, significant_comparisons_A,
    pivoted_B, stacked_B, significant_comparisons_B,
    igrs_to_plot,
    supplement=False
):
    # Filtra as igrs e índices
    igrs = [igr for igr in pivoted_A.index if igr in igrs_to_plot]
    classes = pivoted_A.columns

    pairs_per_row = 4
    cols = pairs_per_row * 2
    rows = int(np.ceil(len(igrs) / pairs_per_row))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = np.array(axes).reshape(rows, cols)

    for idx, igr in enumerate(igrs):
        row = idx // pairs_per_row
        pair = idx % pairs_per_row
        igr_idx = list(pivoted_A.index).index(igr)
        ax_A = axes[row, pair * 2]
        ax_B = axes[row, pair * 2 + 1]

        # Plot A (esquerda)
        igr_data_A = stacked_A[:, igr_idx, :]
        data_A = [igr_data_A[:, j] for j in range(len(classes))]
        bp_A = ax_A.boxplot(
            data_A,
            patch_artist=True,
            medianprops=dict(color="black"),
            positions=np.arange(len(classes)),
            widths=0.5
        )
        for j, patch in enumerate(bp_A['boxes']):
            patch.set_facecolor(f"C{j}")
            patch.set_alpha(0.6)
        ax_A.set_title(f"{igr}\n(Ideal-data)")
        ax_A.set_ylabel('RMSE')
        ax_A.set_xticks([])
        ax_A.set_xticklabels([])
        ax_A.set_xlabel(None)

        # Significância: A
        if igr in significant_comparisons_A:
            y_max = np.max(igr_data_A)
            h = y_max * 0.08
            offset = 0
            for (x1, x2, sig) in significant_comparisons_A[igr]:
                y = y_max + h + offset
                ax_A.plot([x1, x1, x2, x2],
                          [y, y+h*0.3, y+h*0.3, y],
                          lw=1.5, c='k')
                ax_A.text((x1+x2)/2,
                          y+h*0.4,
                          sig,
                          ha='center',
                          va='bottom',
                          fontsize=11)
                offset += h * 1.6
            y_lim = y_max + offset + h * 2
        else:
            y_max = np.max(igr_data_A)
            y_lim = y_max
            h = y_max * 0.08

        # Plot B (direita)
        igr_data_B = stacked_B[:, igr_idx, :]
        data_B = [igr_data_B[:, j] for j in range(len(classes))]
        bp_B = ax_B.boxplot(
            data_B,
            patch_artist=True,
            medianprops=dict(color="black"),
            positions=np.arange(len(classes)),
            widths=0.5
        )
        for j, patch in enumerate(bp_B['boxes']):
            patch.set_facecolor(f"C{j}")
            patch.set_alpha(0.6)
        ax_B.set_title(f"{igr}\n(Real-world)")
        ax_B.set_ylabel('RMSE')
        ax_B.set_xticks([])
        ax_B.set_xticklabels([])
        ax_B.set_xlabel(None)

        # Significância: B
        if igr in significant_comparisons_B:
            y_max_B = np.max(igr_data_B)
            h_B = y_max_B * 0.08
            offset_B = 0
            for (x1, x2, sig) in significant_comparisons_B[igr]:
                y = y_max_B + h_B + offset_B
                ax_B.plot([x1, x1, x2, x2],
                          [y, y+h_B*0.3, y+h_B*0.3, y],
                          lw=1.5, c='k')  # Removido linestyle='--'
                ax_B.text((x1+x2)/2,
                          y+h_B*0.4,
                          sig,
                          ha='center',
                          va='bottom',
                          fontsize=11)
                offset_B += h_B * 1.6
            y_lim_B = y_max_B + offset_B + h_B * 2
        else:
            y_max_B = np.max(igr_data_B)
            y_lim_B = y_max_B
            h_B = y_max_B * 0.08

        # Ajusta o eixo y para ambos os plots da mesma igr
        y_min = min(np.min(igr_data_A), np.min(igr_data_B))
        y_max_final = max(y_lim, y_lim_B)
        ax_A.set_ylim(bottom=y_min, top=y_max_final)
        ax_B.set_ylim(bottom=y_min, top=y_max_final)

    # Desativa eixos não usados
    total_plots = rows * cols
    for i in range(len(igrs), total_plots // 2):
        row = i // pairs_per_row
        pair = i % pairs_per_row
        axes[row, pair * 2].axis('off')
        axes[row, pair * 2 + 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, top=0.95)

    # Legenda global - canto superior direito
    handles = [
        plt.Line2D([0], [0], color=f"C{j}", lw=8, label=classe, alpha=0.6)
        for j, classe in enumerate(classes)
    ]
    fig.legend(
        handles=handles,
        title="Legend",
        loc='lower center',
        ncol=len(classes),
        bbox_to_anchor=(0.5, 1.02)
    )

    filename = 'boxplot_rmse_triplicata'
    if supplement:
        filename += '_supplement'
    filename += '.tif'

    # plt.savefig(filename, 
    #         dpi=100, 
    #         bbox_inches='tight', 
    #         format='tiff')

    return plt.show()

# COMMAND ----------

igrs = [
    "Alegre",
    "Belo Horizonte",
    "Distrito Federal",
    "Maringá",
    "Marília",
    "Oliveira",
    "Passo Fundo",
    "Rio de Janeiro",
    "Santa Cruz do Sul",
    "Santa Maria",
    "São Paulo",
    "Uberlândia",
]

create_boxplots_two_groups(
    pivoted_A = pivoted_result,
    stacked_A = stacked_without_delay, 
    significant_comparisons_A = significant_comparisons_without_delay,
    pivoted_B = pivoted_result, 
    stacked_B = stacked_with_delay, 
    significant_comparisons_B = significant_comparisons_with_delay,
    igrs_to_plot = igrs
)

# COMMAND ----------

igrs_supplement = [
    'Campina Grande',
    'Campos dos Goytacazes',
    'Caratinga',
    'Catalão',
    'Cruz Alta',
    'Frederico Westphalen',
    'Ijuí',
    'Juiz de Fora',
    'Linhares',
    'Passos',
    'Pirapora',
    'Porto Alegre',
    'Ribeirão Preto',
    'Salvador',
    'São Miguel do Oeste',
    'Uberaba',
]

create_boxplots_two_groups(
    pivoted_A = pivoted_result,
    stacked_A = stacked_without_delay, 
    significant_comparisons_A = significant_comparisons_without_delay,
    pivoted_B = pivoted_result, 
    stacked_B = stacked_with_delay, 
    significant_comparisons_B = significant_comparisons_with_delay,
    igrs_to_plot = igrs_supplement,
    supplement=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SARIMAX

# COMMAND ----------

def load_pivot_sarimax(path, seed):
    """
    - Loads the result table by seed
    - Renames network architectures to article standard
    - Removes Caratinga
    - Selects the best model per microregion
    - Pivots the table
    - Sorts the columns
    - Returns the pivoted table
    """
    try:
        df = pd.read_excel(f"{path}{seed}.xlsx")
    except Exception as e:
        print(f"Error loading file: {e}")

    # Removing Caratinga
    df = df[df['igr'] != 'Caratinga']

    best_models = df.loc[
        df.groupby(['igr', 'model'])['RMSE'].idxmin()
    ]

    valid_stations = df['igr'].unique()

    pivoted = (
        best_models.pivot(index="igr", columns="model", values="RMSE")
        .loc[valid_stations]
    )

    
    return pivoted

# COMMAND ----------

path_without_delay = '../4_Model_Training/Sarimax_Without_Delay/best_sarimax_without_delay_'
path_with_delay = '../4_Model_Training/Sarimax_With_Delay/best_sarimax_with_delay_'

# List of seeds to analyze
seeds_without_delay = ['1045', '5046', '7380']
seeds_with_delay = ['2780', '4512', '7317']

pivoted_result = load_pivot_sarimax(path_without_delay, '1045')

# Load and stack all tables
pivoted_without_delay = [load_pivot_sarimax(path_without_delay, seed) for seed in seeds_without_delay]
pivoted_with_delay = [load_pivot_sarimax(path_with_delay, seed) for seed in seeds_with_delay]

# COMMAND ----------

# Stack into a 3D array: (experiments x rows x columns)
stacked_without_delay = np.stack([p.values for p in pivoted_without_delay])
stacked_with_delay = np.stack([p.values for p in pivoted_with_delay])
print(stacked_without_delay.shape)
print(stacked_with_delay.shape)

# COMMAND ----------

df_resultado_without_delay = calculate_means_and_stddevs(
    stacked = stacked_without_delay,
    pivoted = pivoted_result
    )
df_resultado_without_delay

# COMMAND ----------

df_resultado_without_delay.to_excel(f"result_sarimax_without_delay.xlsx")

# COMMAND ----------

df_resultado_with_delay = calculate_means_and_stddevs(
    stacked = stacked_with_delay,
    pivoted = pivoted_result
    )
df_resultado_with_delay

# COMMAND ----------

df_resultado_with_delay.to_excel(f"result_sarimax_with_delay.xlsx")