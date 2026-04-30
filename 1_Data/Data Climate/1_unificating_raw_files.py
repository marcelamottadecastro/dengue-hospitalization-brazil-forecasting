# Databricks notebook source
# MAGIC %md
# MAGIC ### Steps of this notebook:
# MAGIC
# MAGIC - Concatenation of source tables
# MAGIC - Filtering stations by measurement start and end dates

# COMMAND ----------

import os
import pandas as pd

# COMMAND ----------

# Path where climate data is located
path_climate_data = '/Workspace/Users/marcela.motta@afya.com.br/clima_saude/Data/Data Climate/Data Climate/Data Climate Raw'

file_climate_data = os.listdir(path_climate_data)

# COMMAND ----------

def extrair_station_code(csv_file):
    station_code = os.path.splitext(os.path.basename(csv_file))[0]
    return station_code

station_codes = []

for file in file_climate_data:
    if file.endswith('.csv'):
        full_path = os.path.join(path_climate_data, file)
        station_code = extrair_station_code(full_path)
        station_codes.append(station_code)


# COMMAND ----------

station_codes

# COMMAND ----------

len(station_codes)

# COMMAND ----------

len(set(station_codes))

# COMMAND ----------

def process_station_data(csv_file):
    """Function to load and process data from a weather station"""
    try:
        # Extract the station code from the file name
        station_code = os.path.splitext(os.path.basename(csv_file))[0]

        # Read the first lines to capture metadata
        with open(csv_file, 'r') as f:
            station_name = None
            latitude = None
            longitude = None
            start_date = None
            end_date = None

            for line in f:
                line_lower = line.lower()
                if 'nome:' in line_lower:
                    station_name = line.split(':')[1].strip()
                elif 'latitude' in line_lower:
                    latitude = float(line.split(':')[1].strip())
                elif 'longitude' in line_lower:
                    longitude = float(line.split(':')[1].strip())
                elif 'data inicial' in line_lower:
                    start_date = line.split(':')[1].strip()
                elif 'data final' in line_lower:
                    end_date = line.split(':')[1].strip()

                if station_name and latitude is not None and longitude is not None and start_date and end_date:
                    break

            if not station_name:
                print(f"Station name not found in file {csv_file}")
                return None, None

        # Load station data
        df = pd.read_csv(csv_file, skiprows=10, delimiter=';', on_bad_lines='skip')

        # Identify columns of interest
        columns_of_interest = [
            'Data Medicao',
            'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)',
            'TEMPERATURA MAXIMA, DIARIA (AUT)(°C)',
            'TEMPERATURA MEDIA, DIARIA (AUT)(°C)',
            'TEMPERATURA MINIMA, DIARIA (AUT)(°C)',
            'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'
        ]
        existing_columns = [col for col in df.columns if col in columns_of_interest]

        if not existing_columns:
            print(f"Columns of interest not found in {csv_file}")
            return None, None

        df = df[existing_columns]

        # Convert date column
        if 'Data Medicao' in df.columns:
            df['Data Medicao'] = pd.to_datetime(df['Data Medicao'], format='%Y-%m-%d', errors='coerce')

        # Filter data between 2021 and 2024
        df = df[(df['Data Medicao'] >= '2021-01-01') & (df['Data Medicao'] <= '2024-12-31')]

        # Add metadata to the DataFrame
        df['Station'] = station_name
        df['Station Code'] = station_code
        df['Latitude'] = latitude
        df['Longitude'] = longitude
        df['Start Date'] = start_date
        df['End Date'] = end_date

        return df, station_name

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None, None

# COMMAND ----------

# Create the final DataFrame
df_final = pd.DataFrame()

# Process each CSV file
for file in file_climate_data:
    if file.endswith('.csv'):
        file_path = os.path.join(path_climate_data, file)
        df_station, station = process_station_data(file_path)
        if df_station is not None:
            df_final = pd.concat([df_final, df_station], ignore_index=True)

# COMMAND ----------

# Save final CSV
if not df_final.empty:
    output_path = '/Workspace/Users/marcela.motta@afya.com.br/clima_saude/Data/Data Climate/Data Climate/Data Climate Processed/processed_all_stations_with_measurements_A2000_2024.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Final CSV saved to: {output_path}")
else:
    print("No data was processed.")

# COMMAND ----------

df_final

# COMMAND ----------

unique_station = df_final['Station Code'].nunique()
print(f"Number of unique stations with any data from 2021 to 2024: {unique_station}")