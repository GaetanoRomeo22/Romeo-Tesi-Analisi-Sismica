import csv
from math import modf
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from matplotlib import pyplot as plt
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import os
import json
from estimator import TemporalFusionTransformerEstimator
from gluonts.torch.distributions import StudentTOutput


def rtl_to_csv(folder_path, output_file):  # Converto il contenuto dei file rtl in un unico csv
    with open(output_file, mode='w', newline='') as csv_file:  # Apro il file di output in scrittura
        csv_writer = csv.writer(csv_file, delimiter=',')  # csv_writer per scrivere sul file
        csv_writer.writerow(['Date', 'East'])  # Scrivo i nomi dei campi da salvare
        for file in os.listdir(folder_path):  # Elenco tutti i file della directory
            file_path = os.path.join(folder_path, file)  # Ottengo il path del file
            with open(file_path, mode='r') as rtl_file:  # Apro il file rtl in lettura
                for row in rtl_file:  # Per ogni riga del file
                    try:
                        file_data = row.split(',')  # Divido il contenuto del rigo per la virgola
                        date = pd.to_datetime(file_data[2], format='%m%d%y').strftime('%d-%m-%Y')  # Estraggo la data
                        north = modf(float(file_data[3]))[0]  # Estraggo il valore di Grid Northing (parte decimale)
                        csv_writer.writerow([date, north])
                    except IndexError:
                        print("Riga incompleta")

def plot_dataset(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['East'], marker='o', linestyle='-', color='b')
    plt.title('East Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('East (Decimal Part)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def to_deepar_format(dataframe):
    freq = pd.infer_freq(dataframe.index)
    start_index = dataframe.index.min()
    data = [{
                FieldName.START: start_index,
                FieldName.TARGET: dataframe['East'].values,  # Modificato per utilizzare solo la colonna 'North'
            }]
    return ListDataset(data, freq=freq)

def plot_prob_forecasts(ts_entry, forecast_entry, prediction_length):
    plot_length = prediction_length
    context_length = prediction_length
    prediction_intervals = [0.8, 0.95]
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-(context_length + plot_length):].plot(ax=ax)
    forecast_entry.plot(intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()

# rtl_to_csv("ACAE-LICO", "East_Dataset.csv") # Converte i file rtl in csv
dataset = pd.read_csv("East_Dataset.csv")  # Creo un dataframe con pandas
# print(dataset.head())  # Stampo alcuni dati
dataset = dataset.dropna()  # Rimuove i campi NULL
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d-%m-%Y')  # Converto la colonna 'Date' in un tipo datetime
dataset.set_index('Date', inplace=True)  # Imposto la colonna 'Date' come indice del DataFrame
dataset.sort_index(inplace=True)  # Ordino in base alla data
dataset = dataset.resample('D').sum()  # Campiono i dati mensilmente
# plot_dataset(dataset)
validation_data = dataset[-60:-30]  # 30 giorni per la validazione
train_data = dataset[:-60]  # I restanti dati saranno per l'addestramento
test_data = dataset  # Dati di test
train_data_lds = to_deepar_format(train_data)
validation_data_lds = to_deepar_format(validation_data)
test_data_lds = to_deepar_format(test_data)
prediction_length = 7 # Lunghezza della sequenza temporale predetta
context_length = 7 # Lunghezza della sequenza temporale che il modello utilizza per generare le previsioni
freq = "D"

estimator = TemporalFusionTransformerEstimator(
    freq=freq,
    prediction_length=prediction_length,
    context_length=context_length,
    distr_output=StudentTOutput()
)

predictor = estimator.train(training_data=train_data_lds, validation_data=validation_data_lds)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data_lds,
    predictor=predictor
)
tss = list(ts_it)
forecasts = list(forecast_it)

n_forecasts = min(len(tss), len(forecasts))
for i in range(n_forecasts):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry, prediction_length)

evaluator = Evaluator()
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data_lds))
print(json.dumps(agg_metrics, indent=4))
