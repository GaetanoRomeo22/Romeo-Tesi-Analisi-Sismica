import csv
import os
from math import modf
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
from tabulate import tabulate
from scipy.signal import medfilt


def rtl_to_csv(folder_path: str, output_file: str):  # Converto il contenuto dei file rtl in un unico csv
    with open(output_file, mode='w', newline='') as csv_file:  # Apro il file di output in scrittura
        csv_writer = csv.writer(csv_file, delimiter=',')  # csv_writer per scrivere sul file
        csv_writer.writerow(['date', 'east', 'north', 'height'])  # Scrivo i nomi dei campi da salvare
        for file in os.listdir(folder_path):  # Elenco tutti i file della directory
            file_path = os.path.join(folder_path, file)  # Ottengo il path del file
            with open(file_path, mode='r') as rtl_file:  # Apro il file rtl in lettura
                for row in rtl_file:  # Per ogni riga del file
                    file_data = row.split(',')  # Divido il contenuto del rigo per la virgola
                    date = pd.to_datetime(file_data[2], format='%m%d%y')  # Estraggo la data in formato MM-DD-YYYY
                    if date.month == 3 and date.year == 2022 or date.month == 1 and date.year == 2023:  # Salta marzo 2022 e gennaio 2023 perchè ci sono pochi dati
                        continue
                    east = modf(float(file_data[3]))[0] # Estraggo la parte frazionaria del campo East
                    north = modf(float(file_data[5]))[0] # Estraggo la parte frazionaria del campo North
                    height = modf(float(file_data[10]))[0] # Estraggo la parte frazionaria del campo Height
                    csv_writer.writerow([date.strftime('%d-%m-%Y'), east, north, height]) # Scrivo i dati su file

def prepare_dataset(file_path: str) -> DataFrame: # Leggo il contenuto del file csv
    rtl_to_csv("SOLO-LICO", "SOLO-LICO_Dataset.csv")  # Converto i file rtl in csv
    df = pd.read_csv(file_path)  # Creo un dataframe con pandas
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')  # Converto la data nel formato corretto
    df['date'] = df['date'].dt.date  # Tronco la data per evitare problemi di visualizzazione
    apply_median_filter(df=df, kernel_size=15)
    print(tabulate(df.head(), headers='keys', tablefmt='fancy_grid', showindex=False))  # Visualizzo il data frame
    show_rolling_statistics(df=df, window=12)
    plot_acf_pacf_for_all_coordinates(df=df, lags=30)
    return df

def apply_median_filter(df: DataFrame, kernel_size: int = 15):  # Applico un filtro mediano ai dati per ridurre il rumore
    assert kernel_size % 2 != 0, "Il kernel_size deve essere un numero dispari."
    df['east'] = medfilt(df['east'], kernel_size=kernel_size)
    df['north'] = medfilt(df['north'], kernel_size=kernel_size)
    df['height'] = medfilt(df['height'], kernel_size=kernel_size)

def show_rolling_statistics(df: DataFrame, window: int = 12):
    targets = ['east', 'north', 'height']
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), sharex=True)
    for i, target in enumerate(targets):
        rolling_mean = df[target].rolling(window=window, min_periods=1).mean()
        rolling_std = df[target].rolling(window=window, min_periods=1).std()
        axes[i].plot(df[target], color='cornflowerblue', label=target)
        axes[i].plot(rolling_mean, color='firebrick', label='Rolling Mean')
        axes[i].plot(rolling_std, color='limegreen', label='Rolling Std')
        axes[i].set_title(f'Rolling Statistics ({target})', size=14)
        axes[i].set_xlabel('Date', size=12)
        axes[i].set_ylabel(target, size=12)
        axes[i].legend(loc='best')
    plt.tight_layout()
    plt.show()
    for target in targets:
        test_stationary(df=df, target=target)

def test_stationary(df: DataFrame, target: str): # Controllo la stazionarità dei dati
    print('Results of Dickey Fuller Test: (' + target + ')')
    dftest = adfuller(df[target], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'No. of Lags used', 'Number of observations used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def show_dataset(df: DataFrame, title: str):
    # Creazione dei subplot
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)  # 3 righe, 1 colonna, condividi l'asse x

    # Nord
    axs[0].scatter(df.index, df['north'], color='blue', s=1, alpha=0.7)
    axs[0].set_ylabel("Nord (cm)")
    axs[0].grid(alpha=0.3)

    # Est
    axs[1].scatter(df.index, df['east'], color='red', s=1, alpha=0.7)
    axs[1].set_ylabel("Est (cm)")
    axs[1].grid(alpha=0.3)

    # Up
    axs[2].scatter(df.index, df['height'], color='black', s=1, alpha=0.7)
    axs[2].set_ylabel("Up (cm)")
    axs[2].grid(alpha=0.3)

    # Titolo generale
    fig.suptitle(title, fontsize=16)
    axs[2].set_xlabel("Data")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Regola margini per titolo
    plt.show()

def show_comparison_plot(original_df: DataFrame, filtered_df: DataFrame, title: str):
    # Creazione dei subplot
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)  # 3 righe, 1 colonna, condividi l'asse x

    targets = ['north', 'east', 'height']  # Dati da confrontare

    for i, target in enumerate(targets):
        axs[i].plot(original_df.index, original_df[target], label=f'{target.capitalize()} Originale', color='blue', alpha=0.6)
        axs[i].plot(filtered_df.index, filtered_df[target], label=f'{target.capitalize()} Filtrato', color='red', alpha=0.8)
        axs[i].set_ylabel(f"{target.capitalize()} (cm)")
        axs[i].grid(alpha=0.3)
        axs[i].legend(loc='best')

    # Titolo generale
    fig.suptitle(title, fontsize=16)
    axs[-1].set_xlabel("Data")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Regola margini per il titolo
    plt.show()

def plot_acf_pacf_for_all_coordinates(df: DataFrame, lags: int = 30):
    coordinates = ['east', 'north', 'height']  # Le coordinate da analizzare
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))  # 3 righe, 2 colonne

    # Ciclo per ogni coordinata e creazione dei grafici
    for i, coord in enumerate(coordinates):
        # ACF
        smt.graphics.plot_acf(df[coord], lags=lags, ax=axes[i, 0], alpha=0.5)
        axes[i, 0].set_title(f"ACF - {coord.capitalize()}")

        # PACF
        smt.graphics.plot_pacf(df[coord], lags=lags, ax=axes[i, 1], alpha=0.5)
        axes[i, 1].set_title(f"PACF - {coord.capitalize()}")

    # Titolo generale
    fig.suptitle("ACF e PACF per East, North e Height", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Aggiusta margini per il titolo
    plt.show()

if __name__ == '__main__':
    dataset = prepare_dataset("SOLO-LICO_Dataset.csv") # Leggo il file csv e carico il contenuto in un dataframe pandas