import csv
import pandas as pd
from math import modf
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def rtl_to_csv(folder_path, output_file):  # Converto il contenuto dei file rtl in un unico csv
    with open(output_file, mode='w', newline='') as csv_file:  # Apro il file di output in scrittura
        csv_writer = csv.writer(csv_file, delimiter=',')  # csv_writer per scrivere sul file
        csv_writer.writerow(['date', 'height'])  # Scrivo i nomi dei campi da salvare
        for file in os.listdir(folder_path):  # Elenco tutti i file della directory
            file_path = os.path.join(folder_path, file)  # Ottengo il path del file
            with open(file_path, mode='r') as rtl_file:  # Apro il file rtl in lettura
                for row in rtl_file:  # Per ogni riga del file
                    try:
                        file_data = row.split(',')  # Divido il contenuto del rigo per la virgola
                        date = pd.to_datetime(file_data[2], format='%m%d%y').strftime('%d-%m-%Y')  # Estraggo la data
                        target = modf(float(file_data[10]))[0]  # Estraggo il valore di Grid Northing (parte decimale)
                        csv_writer.writerow([date, target])
                    except IndexError:
                        print("Riga incompleta")

if __name__ == '__main__':
    # Caricamento e settaggio del dataset
    rtl_to_csv("SOLO-LICO", "Dataset_ARIMA.csv") # Converte i file rtl in csv
    dataset = pd.read_csv("Dataset_ARIMA.csv")  # Creo un dataframe con pandas
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y')
    dataset['month'] = dataset.date.dt.month.astype(str)
    dataset['month'] = pd.to_datetime(dataset['month'], format='%m')
    dataset.set_index('month', inplace=True)
    dataset.sort_index(inplace=True)
    print(dataset)

    '''
    rolling_mean = dataset.rolling(window=12).mean()
    rolling_std = dataset.rolling(window=12).std()
    plt.figure(figsize=(10, 6))
    plt.plot(dataset['height'], color='cornflowerblue', label='Original')
    plt.plot(rolling_mean, color='firebrick', label='Rolling Mean')
    plt.plot(rolling_std, color='limegreen', label='Rolling Std')
    plt.xlabel('Date', size=12)
    plt.ylabel('Height', size=12)
    plt.legend(loc='upper left')
    plt.title('Rolling Statistics', size=14)
    plt.show()

    plt.figure(figsize=(18, 9))
    plt.plot(dataset.index, dataset["height"], linestyle="-")
    plt.xlabel = 'Dates'
    plt.ylabel = 'Heights'
    plt.show()

    rcParams['figure.figsize'] = 12, 8
    a = seasonal_decompose(dataset["height"], model="add", period=24)
    a.plot()

    plt.figure(figsize=(25, 5))
    a = seasonal_decompose(dataset["height"], model="add", period=24)
    plt.subplot(1, 3, 1)
    plt.plot(a.seasonal)
    plt.subplot(1, 3, 2)
    plt.plot(a.trend)
    plt.subplot(1, 3, 3)
    plt.plot(a.resid)
    plt.show()
    
    plot_acf(train_data['height'], lags=50)
    plot_pacf(train_data['height'], lags=50)
    plt.show()
    
    result = adfuller(train_data['height'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    '''

    train_data = dataset[:len(dataset) - 720]
    test_data = dataset[len(dataset) - 720:]

    arima_model = SARIMAX(train_data['height'], order=(2, 1, 2), seasonal_order=(4, 0, 3, 12))
    arima_result = arima_model.fit()
    print(arima_result.summary())

    arima_pred = arima_result.predict(start=len(train_data), end=len(dataset) - 1, typ="levels").rename("ARIMA Predictions")
    print(arima_pred)
    test_data['height'].plot(figsize=(16, 5), legend=True)
    arima_pred.plot(legend=True)
    plt.show()

    arima_rmse_error = rmse(test_data['height'], arima_pred)
    arima_mse_error = arima_rmse_error ** 2
    mean_value = dataset['height'].mean()
    print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}')

    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['height'], label="True values", color="blue")
    plt.plot(test_data.index, arima_pred, label = "forecasts", color='orange')
    plt.title("ARIMA Model", size=14)
    plt.legend(loc='upper left')
    plt.show()

    test_data['ARIMA_Predictions'] = arima_pred.values
    print(test_data)
