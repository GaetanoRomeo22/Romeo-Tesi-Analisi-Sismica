import csv
from math import modf
import pandas as pd
import ujson
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import Evaluator
from matplotlib import pyplot as plt
import os
from gluonts.evaluation.backtest import make_evaluation_predictions
from sklearn.model_selection import train_test_split
from module import DeepARModel
from lightning_module import DeepARLightningModule
from estimator import DeepAREstimator
from scipy.signal import medfilt


__all__ = [
    "DeepARModel",
    "DeepARLightningModule",
    "DeepAREstimator"
]

def rtl_to_csv(folder_path, output_file):  # Converto il contenuto dei file rtl in un unico csv
    with open(output_file, mode='w', newline='') as csv_file:  # Apro il file di output in scrittura
        csv_writer = csv.writer(csv_file, delimiter=',')  # csv_writer per scrivere sul file
        csv_writer.writerow(['Time', 'Date', 'Height'])  # Scrivo i nomi dei campi da salvare
        for file in os.listdir(folder_path):  # Elenco tutti i file della directory
            file_path = os.path.join(folder_path, file)  # Ottengo il path del file
            with open(file_path, mode='r') as rtl_file:  # Apro il file rtl in lettura
                for row in rtl_file:  # Per ogni riga del file
                    try:
                        file_data = row.split(',')  # Divido il contenuto del rigo per la virgola
                        time_str = file_data[1].strip()  # Estraggo l'orario dalla colonna corretta e rimuovo spazi
                        time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"  # Converto il tempo a formato HH:MM:SS
                        date = pd.to_datetime(file_data[2], format='%m%d%y').strftime('%d-%m-%Y')  # Estraggo la data
                        height = modf(float(file_data[10]))[0]  # Estraggo il valore di Grid Northing (parte decimale)
                        csv_writer.writerow([time, date, height])
                    except IndexError:
                        print("Riga incompleta")

def plot_dataset(data): # Stampa del dataset a disposizione
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Height'], marker='o', linestyle='-', color='b')
    plt.title('Height Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Height (Decimal Part)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def to_deepar_format(dataframe): # Conversione in formato richiesto dal modello DeepAR
    start_index = dataframe.index.min()
    data = [{
        FieldName.START: start_index,
        FieldName.TARGET: dataframe['Height'].values  # Modificato per utilizzare solo la colonna 'North'
    }]
    # print(data[0])
    return ListDataset(data, freq=pd.infer_freq(dataframe.index) or "D") # Frequenza giornaliera impostata

def plot_prob_forecasts(ts, forecast): # Stampa delle previsioni
    plot_length = prediction_length
    legend = ["observations", "median prediction"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts[-plot_length:].plot(ax=ax)
    forecast.plot( color='orange')
    plt.legend(legend, loc="upper left")
    plt.show()

# Caricamento e settaggio del dataset
rtl_to_csv("SOLO-LICO", "Height_Dataset.csv") # Converte i file rtl in csv
dataset = pd.read_csv("Height_Dataset.csv")  # Creo un dataframe con pandas
dataset["Date"] = pd.to_datetime(dataset["Date"], format='%d-%m-%Y')
dataset.set_index('Date', inplace=True)  # Imposto la colonna 'Date' come indice del DataFrame
dataset.sort_index(inplace=True)  # Ordino in base alla data
dataset = dataset.sort_values(by=['Date', 'Time'])
dataset['Height'] = medfilt(dataset['Height'], kernel_size=15)
# print(dataset.head())  # Stampo alcuni dati
# dataset = dataset.resample('D').sum()  # Campiono i dati quotidianamente
# plot_dataset(dataset) # Visualizzo il dataset
# Divido il dataset in train, validation e test (Divisione in percentuale)
train_data, validation_data = train_test_split(dataset, test_size=0.3, random_state=42)
# validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
test_data = dataset

# Divido il dataset in train, validation e test (Divisione fissa)
'''
validation_data = dataset[-240:-120]
train_data = dataset[:-120]
test_data = dataset
'''

# Converto i dati nel formato richiesto dal modello DeepAR
train_data_lds = to_deepar_format(train_data)
validation_data_lds = to_deepar_format(validation_data)
test_data_lds = to_deepar_format(test_data)

# Dati del modello di previsione
freq = "H" # Frequenza oraria
prediction_length = 7 if freq == 'D' else 168 if freq == 'H' else None # Lunghezza della sequenza temporale predetta
context_length = 30 if freq == 'D' else 504 if freq == 'H' else None # Lunghezza della sequenza temporale che il modello utilizza per generare le previsioni
epochs = 30 # Numero di epoche da usare
num_layers = 2
batch_size = 64
num_batches_per_epoch = 50

# Fase di allenamento e predizione
estimator = DeepAREstimator(
    freq = freq,
    context_length = context_length,
    prediction_length = prediction_length,
    cardinality = [1],
    num_layers = num_layers,
    batch_size = batch_size,
    num_batches_per_epoch = num_batches_per_epoch,
    lr = 0.05,
    trainer_kwargs = {
        "max_epochs": epochs
    },
    patience=5,
    dropout_rate=0.3
)
predictor = estimator.train(training_data=train_data_lds, validation_data=validation_data_lds)
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data_lds,
    predictor=predictor,
    num_samples=len(test_data_lds)
)
tss = list(ts_it)
forecasts = list(forecast_it)

# Stampa dei grafici delle previsioni
for i in range(1):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry)

'''
# Stampa delle metriche di valutazione
evaluator = Evaluator()
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data_lds))
print(ujson.dumps(agg_metrics, indent=4))
print(item_metrics)
'''