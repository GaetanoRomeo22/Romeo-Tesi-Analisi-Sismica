"""-------------------------------------------------------------------------------------------------
Il seguente codice ha come obiettivo quello di cercare di predire, a partire da una serie di
osservazioni dei Campi Flegrei di Napoli ad opera di varie stazioni, il valore di una delle tre
coordinate nel sistema ENU (East, North, Up).
Le osservazioni sono scandite ad intervalli orari e il modello di apprendimento utilizzato
è il Temporal Fusion Transformer.
Autore: Gaetano Romeo
-------------------------------------------------------------------------------------------------"""
import csv
import os
from math import modf
from pandas import DataFrame
from tabulate import tabulate
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import optuna
from pytorch_forecasting.metrics import QuantileLoss
import matplotlib.pyplot as plt
from lightning.pytorch.tuner import Tuner
import torch
from scipy.signal import medfilt
from statsmodels.tsa.stattools import adfuller

# Logger per wandb (richiede wandb e WandbLogger tra gli import)
# wandb.init(project="Romeo_Temporal_Fusion_Transformer")
# logger = WandbLogger()

# Logger per TensorBoard (richiede TensorBoard tra gli import)
# logger = TensorBoardLogger("lightning_logs")

"""-------------------------------------------------------------------------------------------------
Per una più comoda gestione del dataset, ho convertito tutti i file .rtl in un unico file .csv.
Gli unici campi memorizzati sono l'ora, la data e il valore da predire (East, North o Height).
Siccome è stato notato che il valore intero per le osservazioni è costante, per il campo target
è stata estratta solo la parte decimale.
Avendo a disposizione dati compresi tra i valori 0.1 e 0.9, è stato ritenuto opportuno non adottare
alcuna normalizzazione.
Le osservazioni inerenti a marzo 2022 e gennaio 2023 sono state ignorate in quanto non sufficienti.
-------------------------------------------------------------------------------------------------"""
def rtl_to_csv(folder_path: str, output_file: str):  # Converto il contenuto dei file rtl in un unico csv
    with open(output_file, mode='w', newline='') as csv_file:  # Apro il file di output in scrittura
        csv_writer = csv.writer(csv_file, delimiter=',')  # csv_writer per scrivere sul file
        csv_writer.writerow(['ID', 'date', 'east', 'north', 'height'])  # Scrivo i nomi dei campi da salvare
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
                    csv_writer.writerow(['$GPLLQ', date.strftime('%d-%m-%Y'), east, north, height]) # Scrivo i dati su file

def prepare_dataset(file_path: str) -> DataFrame: # Leggo il contenuto del file csv
    rtl_to_csv("SOLO-LICO", "SOLO-LICO_Dataset.csv")  # Converto i file rtl in csv
    df = pd.read_csv(file_path)  # Creo un dataframe con pandas
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')  # Converto la data nel formato corretto
    df['day'] = df.date.dt.day.astype(str)  # Aggiungo l'informazione relativa al giorno
    df['month'] = df.date.dt.month.astype(str)  # Aggiungo l'informazione relativa al mese
    df['year'] = df.date.dt.year.astype(str)  # Aggiungo l'informazione relativa all'anno
    df['time_idx'] = df.groupby(['month']).cumcount()  # Aggiungo una colonna per il time index per il TFT raggruppando per il mese
    df['date'] = df['date'].dt.date  # Tronco la data per evitare problemi di visualizzazione
    df.set_index(['date'], inplace=True)
    df.sort_index(inplace=True)
    print(df)
    return df

def show_dataset(df: DataFrame, target: str, title: str): # Visualizzo il dataset
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset['date'], df[target], color='blue', alpha=0.5, label=target)
    plt.title(title, fontsize=16)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel(target, fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

def apply_median_filter(df: DataFrame, kernel_size: int = 15): # Applico un filtro mediano ai dati per ridurre il rumore
    assert kernel_size % 2 != 0, "Il kernel_size deve essere un numero dispari."
    df['east'] = medfilt(df['east'], kernel_size=kernel_size)
    df['north'] = medfilt(df['north'], kernel_size=kernel_size)
    df['height'] = medfilt(df['height'], kernel_size=kernel_size)

def show_rolling_statistics(df: DataFrame, target: str, window: int = 12): # Mostro le statistiche dei dati
    rolling_mean = df[target].rolling(window=window, min_periods=1).mean()
    rolling_std = df[target].rolling(window=window, min_periods=1).std()
    plt.figure(figsize=(10, 6))
    plt.plot(df[target], color='cornflowerblue', label=target)
    plt.plot(rolling_mean, color='firebrick', label='Rolling Mean')
    plt.plot(rolling_std, color='limegreen', label='Rolling Std')
    plt.xlabel('Date', size=12)
    plt.ylabel(target, size=12)
    plt.legend(loc='best')
    plt.title('Rolling Statistics (' + target + ')', size=14)
    plt.show()
    test_stationary(df=df, target=target, window=window)

def test_stationary(df: DataFrame, target: str, window: int = 12): # Controllo la stazionarità dei dati
    movingAverage = df[target].rolling(window=window, min_periods=1).mean()
    movingSTD = df[target].rolling(window=window, min_periods=1).std()
    plt.figure(figsize=(10, 6))
    plt.plot(df[target], color='blue', label=target)
    plt.plot(movingAverage, color='red', label='Rolling Mean')
    plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation (' + target + ')', size=14)
    plt.show(block=False)
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(df[target], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'No. of Lags used', 'Number of observations used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def save_results(model, results, target: str): # Salvo le predizioni e il confronto con i valori effettivi su file
    results_vs_actuals = model.calculate_prediction_actual_by_variable(results.x, results.output)
    values_actual = results_vs_actuals["average"]["actual"][target].cpu()
    values_prediction = results_vs_actuals["average"]["prediction"][target].cpu()
    values_actual = values_actual[values_actual != 0]
    values_prediction = values_prediction[values_prediction != 0]
    predictions_dataframe = pd.DataFrame({
        "Valori Effettivi " + "(" + target + ")": values_actual,
        "Valori Predetti " + "(" + target + ")": values_prediction
    })
    predictions_dataframe.to_csv("Predizioni_" + target + ".csv", index=True)
    print(tabulate(predictions_dataframe, headers='keys', tablefmt='fancy_grid'))

if __name__ == '__main__':
    """-------------------------------------------------------------------------------------------------
    Come prima cosa, utilizzo la funzione definita per convertire i file rtl in un unico file csv
    contenente i campi ID, ora, data e target.
    Fatto ciò, utilizzo pandas per salvare tutti i dati in un dataframe.
    Successivamente costruisco un campo descrittivo per indicare il mese di ogni osservazione da
    utilizzare per raggruppare i dati.
    Il campo "time_idx", che consiste in un identificativo progressivo per ogni osservazione,
    fa riferimento al mese, il che significa che ogni mese di osservazioni sarà una serie temporale.
    Il dataset è diviso nella parte di dati utilizzata per l'addestramento (80%) e in quella
    utilizzata per le predizioni (20%).
    I campi max_prediction_length e max_encoder_length indicano, rispettivamente, quante osservazioni
    il modello deve predire e quante deve osservarne per generare le predizioni.
    In questo caso sono state impostate a 24 e 168, il che significa che l'obiettivo è predire una
    settimana di osservazioni a partire da tre settimane.
    -------------------------------------------------------------------------------------------------"""
    # Caricamento e settaggio del dataset
    dataset = prepare_dataset("SOLO-LICO_Dataset.csv") # Leggo il file csv e carico il contenuto in un dataframe pandas

    # Dataset originale
    show_dataset(dataset, 'east', 'Dataset') # Mostro un grafico per visualizzare il dataset originale
    show_dataset(dataset, 'north', 'Dataset') # Mostro un grafico per visualizzare il dataset originale
    show_dataset(dataset, 'height', 'Dataset') # Mostro un grafico per visualizzare il dataset originale

    # Dataset post filtragio mediano
    apply_median_filter(dataset, kernel_size=15)  # Filtro i dati con un filtro mediano per ridurre il rumore
    show_dataset(dataset, 'east', 'Dataset filtrato') # Mostro un grafico per visualizzare il dataset post filtraggio
    show_dataset(dataset, 'north', 'Dataset filtrato') # Mostro un grafico per visualizzare il dataset post filtraggio
    show_dataset(dataset, 'height', 'Dataset filtrato') # Mostro un grafico per visualizzare il dataset post filtraggio

    # Controllo della stazionarità
    show_rolling_statistics(df=dataset, target='east', window=12)
    show_rolling_statistics(df=dataset, target='north', window=12)
    show_rolling_statistics(df=dataset, target='height', window=12)

    # Divisione dataset in train e validation
    train_cnt = int(len(dataset) * .8)  # Divido il dataset in 80% train e 20% test
    train = dataset.iloc[:train_cnt] # Dati di train
    test = dataset.iloc[train_cnt:] # Dati di test
    max_prediction_length = 168 # Numero di osservazioni da predire
    max_encoder_length = 504 # Numero di osservazioni da analizzare per le predizioni
    batch_size = 64
    epochs = 30 # Numero di epoche

    # Costruzione del dataset nel formato richiesto dal Temporal Fusion Transformer
    training = TimeSeriesDataSet( # Converto il dataset nel formato richiesto dal TFT
        train, # Dati di addestramento
        time_idx='time_idx', # Indice temporale per le serie temporali
        target='north', # Valore da predire
        group_ids=['month'], # Campo utilizzato per identificare univocamente le serie temporali
        min_encoder_length=max_encoder_length // 2, # Numero minimo di osservazioni da analizzare per le predizioni
        max_encoder_length=max_encoder_length, # Numero di osservazioni da analizzare per le predizioni
        min_prediction_length=1, # Numero minimo di osservazioni da predire
        max_prediction_length=max_prediction_length, # Numero di osservazioni da predire
        static_categoricals=['ID'], # Parametri categorici statici
        time_varying_known_reals=['time_idx', 'east', 'height', 'day', 'year'], # Parametri che variano nel tempo e di cui si conosce il valore futuro
        time_varying_unknown_reals=['north'], # Parametri che variano nel tempo e di cui non si conosce il valore futuro
        target_normalizer=GroupNormalizer( # Normalizzazione dei parametri
            groups=['month'],
            transformation="softplus"
        ),
        add_relative_time_idx=True, # Aggiunge il time_idx alle features
        add_target_scales=True, # Aggiunge la media al target
        allow_missing_timesteps=True, # Consente serie temporali interrotte
        add_encoder_length=True
    )

    # Creazione dataloader di train e validation (l'aggiunta dei workers consente il lavoro in parallelo)
    validation = TimeSeriesDataSet.from_dataset(training, dataset, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=8, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=8, persistent_workers=True)

    """-------------------------------------------------------------------------------------------------
    Le seguenti istruzioni hanno l'obiettivo di analizzare il dataset e il modello da utilizzare
    per determinare il valore di learning rate ottimale da utilizzare per la fase di training.
    Una volta trovato il valore, lo si memorizza ed è possibile visualizzarne il grafico.
    Il valore è cercato nell'intervallo [0.01, 0.0001] consigliato nel paper di riferimento del TFT.
    -------------------------------------------------------------------------------------------------"""
    # Fase di ricerca del miglior learning rate con Tuner
    pl.seed_everything(42)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        gradient_clip_val=0.1
    )
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=40,
        attention_head_size=1,
        dropout=0.2,
        hidden_continuous_size=24,
        loss=QuantileLoss(),
        optimizer="Ranger"
    )
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=0.01,
        min_lr=0.0001
    )

    # Visualizzazione del grafico del learning rate ottimale
    # print("Learning Rate ottimale: ", res.suggestions())
    # fig = res.plot(show=True, suggest=True)
    # fig.show()

    """-------------------------------------------------------------------------------------------------
    La seguente funzione ha l'obiettivo di trovare i pesi ottimali di addestramento per il modello.
    Essa effettua un numero specificato dall'utente di tentativi di addestramento con l'obiettivo di
    minimizzare il valore della validation loss.
    L'intervallo di valori da provare per ogni peso è consigliato nel paper di riferimento del TFT.
    E' presente un monitor che ferma l'addestramento nel caso in cui si nota che il modello in più
    iterazioni consecutive non migliora le proprie prestazioni.
    L'utilizzo di Optuna permette di generare un file .db dal quale è possibile osservare informazioni
    circa l'addestramento mediante la dashboard da terminale.
    Una volta trovati i migliori parametri di addestramento, si salva il miglior modello trovato
    per poi ricaricarlo una volta che si vuole iniziare la fase di validazione.
    -------------------------------------------------------------------------------------------------"""
    def objective(trial): # Funzione per trovare i migliori pesi di addestramento
        # Dominio dei valori da provare per ogni peso
        hidden_size = trial.suggest_int("hidden_size", 8, 64, step=8)
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 64, step=8)
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
        attention_head_size = trial.suggest_int("attention_head_size", 1, 4)

        tft = TemporalFusionTransformer.from_dataset( # Creo il TFT con i migliori pesi
            training,
            learning_rate=res.suggestion(),
            hidden_size=hidden_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            attention_head_size=attention_head_size,
            loss=QuantileLoss(),
            optimizer="Ranger",
            reduce_on_plateau_patience=5
        )

        # Monitoro il learning rate per stoppare l'algoritmo di addestramento quando non apprende ulteriormente
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.004, patience=3, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()

        trainer = pl.Trainer( # Criteri di addestramento
            max_epochs=epochs,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[lr_logger, early_stop_callback],
            enable_checkpointing=True,
            # logger=logger
        )

        trainer.fit( # Fase di addestramento
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        # Valutazione del modello in termini di validation loss
        return trainer.callback_metrics['val_loss'].item()

    # Creazione di un nuovo studio optuna con l'obiettivo di minimizzare la validation loss
    study = optuna.create_study(direction="minimize", storage="sqlite:///Romeo_Temporal_Fusion_Transformer.db")

    # Tentativi di ottimizzazione
    study.optimize(objective, n_trials=10)

    # Monitoro il training per interromperlo quando la validation loss raggiunge un valore delta minimo
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.004, patience=3, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer( # Definizione dei parametri di addestramento del TFT
        max_epochs=epochs,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        # logger=logger
    )

    tft = TemporalFusionTransformer.from_dataset( # Definizione del TFT con i migliori parametri
        training,
        learning_rate=res.suggestion(),
        hidden_size=study.best_params['hidden_size'],
        dropout=study.best_params['dropout'],
        hidden_continuous_size=study.best_params['hidden_continuous_size'],
        attention_head_size=study.best_params['attention_head_size'],
        loss=QuantileLoss(),
        optimizer="Ranger",
        reduce_on_plateau_patience=5
    )

    trainer.fit( # Fase di addestramento
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Stampa della validation loss
    # print("Validation loss finale:", trainer.callback_metrics['val_loss'].item())

    # Carico il TFT con i pesi ottimali
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Salvo il modello ottimale
    torch.save(best_tft, "Migliori_Modelli/Romeo_Best_TFT_North.pth")

    # Carico il miglior modello e lo metto in fase di validazione
    best_tft = torch.load("Migliori_Modelli/Romeo_Best_TFT_North.pth")
    best_tft.eval()
    """-------------------------------------------------------------------------------------------------
    La seguente parte di codice mette in evidenza la fase di validazione del modello.
    Come prima cosa si ricavano le predizioni del modello e le si visualizzano graficamente.
    Successivamente si visualizzano le interpretazioni del modello in termini di attenzione e importanza
    di ogni parametro del dataset.
    Infine è visualizzato un grafico che mette a confronto i valori predetti e quelli effettivi.
    -------------------------------------------------------------------------------------------------"""
    # Ottengo e stampo le predizioni del modello
    predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    for idx in range(len(predictions)):
        best_tft.plot_prediction(predictions.x, predictions.output, idx=idx, add_loss_to_title=True)
    plt.show()

    # Stampa delle interpretazioni (importanza e attenzione)
    interpretation = best_tft.interpret_output(predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)
    plt.show()

    # Stampa del confronto predizioni-valori effettivi
    predictions = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output, normalize=False)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    plt.show()

    # Salvataggio dei risultati su file csv
    predictions = best_tft.predict(val_dataloader, return_x=True)
    save_results(best_tft, predictions, "north")