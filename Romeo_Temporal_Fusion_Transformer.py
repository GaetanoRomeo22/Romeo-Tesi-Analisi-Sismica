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
from typing import Dict, Union, Any
import numpy as np
from tabulate import tabulate
from math import modf
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, to_list
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import optuna
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, Metric, MASE
import matplotlib.pyplot as plt
from lightning.pytorch.tuner import Tuner
import torch

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
def rtl_to_csv(folder_path, output_file):  # Converto il contenuto dei file rtl in un unico csv
    with open(output_file, mode='w', newline='') as csv_file:  # Apro il file di output in scrittura
        csv_writer = csv.writer(csv_file, delimiter=',')  # csv_writer per scrivere sul file
        csv_writer.writerow(['time', 'date', 'height'])  # Scrivo i nomi dei campi da salvare
        for file in os.listdir(folder_path):  # Elenco tutti i file della directory
            file_path = os.path.join(folder_path, file)  # Ottengo il path del file
            with open(file_path, mode='r') as rtl_file:  # Apro il file rtl in lettura
                for row in rtl_file:  # Per ogni riga del file
                    file_data = row.split(',')  # Divido il contenuto del rigo per la virgola
                    time_str = file_data[1].strip()  # Estraggo l'orario dalla colonna corretta e rimuovo spazi
                    time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}" # Converto il tempo in formato HH:MM:SS
                    date = pd.to_datetime(file_data[2], format='%m%d%y')  # Estraggo la data in formato MM-DD-YYYY
                    if date.month == 3 and date.year == 2022 or date.month == 1 and date.year == 2023:  # Salta marzo 2022 e gennaio 2023 perchè ci sono pochi dati
                        continue
                    target = modf((float(file_data[10])))[0]  # Estraggo il valore di Grid Northing (parte decimale)
                    csv_writer.writerow([time, date.strftime('%d-%m-%Y'), target]) # Scrivo i dati su file


def plot_prediction(
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        loss: Metric = None
):
    """
    Plot prediction vs actuals: observed curve, predicted curve, and quantiles.
    Args:
        x: network input
        out: network output
        idx: index of prediction to plot
    Returns:
        matplotlib figure
    """
    # all true values for y of the first sample in batch
    encoder_targets = to_list(x["encoder_target"])
    decoder_targets = to_list(x["decoder_target"])

    y_hats = to_list(to_prediction(out, loss))
    y_quantiles = to_list(to_quantiles(out, loss))

    # for each target, plot
    figs = []
    for y_hat, y_quantile, encoder_target, decoder_target in zip(
            y_hats, y_quantiles, encoder_targets, decoder_targets
    ):
        y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
        max_encoder_length = x["encoder_lengths"].max()
        y = torch.cat(
            (
                y_all[: x["encoder_lengths"][idx]],
                y_all[max_encoder_length: (max_encoder_length + x["decoder_lengths"][idx])],
            ),
        )
        # move predictions to cpu
        y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
        y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]

        # move to cpu
        y = y.detach().cpu()
        # create figure
        fig, ax = plt.subplots()
        n_pred = y_hat.shape[0]
        x_pred = np.arange(n_pred)
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        pred_color = next(prop_cycle)["color"]
        if len(x_pred) > 1:
            plotter = ax.plot
        else:
            plotter = ax.scatter

        # plot observed prediction (actual values)
        plotter(x_pred, y[-n_pred:], label="Observed", c=obs_color)

        # plot predicted values (mean prediction)
        plotter(x_pred, y_hat, label="Predicted", c=pred_color)

        ax.set_xlabel("Time index")
        fig.legend()
        figs.append(fig)

    # Return multiple figures if there are multiple targets, else return one
    if isinstance(x["encoder_target"], (tuple, list)):
        return figs
    else:
        return figs[0]

def to_prediction(out: Dict[str, Any], loss):
    """
    Convert output to prediction using the loss metric.
    Args:
        out (Dict[str, Any]): output of network where "prediction" has been
            transformed with :py:meth:`~transform_output`
    Returns:
        torch.Tensor: predictions of shape batch_size x timesteps
    """
    y_pred = out["prediction"]
    if y_pred.shape[-1] > 1:
        y_pred = y_pred[:, :, 0]
    y_pred = loss.to_prediction(y_pred)
    return y_pred

def to_quantiles(out: Dict[str, Any], loss):
    """
    Convert output to quantiles using the loss metric.
    Args:
        out (Dict[str, Any]): output of network where "prediction" has been
            transformed with :py:meth:`~transform_output`
    Returns:
        torch.Tensor: quantiles of shape batch_size x timesteps x n_quantiles
    """
    out = loss.to_quantiles(out["prediction"])
    return out

if __name__ == '__main__':
    """-------------------------------------------------------------------------------------------------
    Come prima cosa, utilizzo la funzione definita per convertire i file rtl in un unico file csv
    contenente i campi ID, ora, data e target.
    Fatto ciò, utilizzo pandas per salvare tutti i dati in un dataframe e uso dropna per eliminare
    eventuali record incompleti.
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
    # rtl_to_csv("STRZ-LICO", "STRZ-LICO_Dataset.csv") # Converto i file rtl in csv
    dataset = pd.read_csv("STRZ-LICO_Dataset.csv")  # Creo un dataframe con pandas
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y') # Converto la data nel formato corretto
    dataset['month'] = dataset.date.dt.month.astype(str) # Aggiungo l'informazione relativa al mese
    dataset['time_idx'] = dataset.groupby(['month']).cumcount()  # Aggiungo una colonna per il time index per il TFT raggruppando per il mese
    dataset['date'] = dataset['date'].dt.date
    # print(dataset.head())
    train_cnt = int(len(dataset) * .8)  # Divido il dataset in 80% train e 20% test
    train = dataset.iloc[:train_cnt] # Dati di train
    test = dataset.iloc[train_cnt:] # Dati di test
    max_prediction_length = 168 # Numero di osservazioni da predire
    max_encoder_length = 504 # Numero di osservazioni da analizzare per le predizioni
    batch_size = 64
    epochs = 30 # Numero di epoche

    training = TimeSeriesDataSet( # Converto il dataset nel formato richiesto dal TFT
        train, # Dati di addestramento
        time_idx='time_idx', # Indice temporale per le serie temporali
        target='height', # Valore da predire
        group_ids=['month'], # Campo utilizzato per identificare univocamente le serie temporali
        min_encoder_length=max_encoder_length // 2, # Numero minimo di osservazioni da analizzare per le predizioni
        max_encoder_length=max_encoder_length, # Numero di osservazioni da analizzare per le predizioni
        min_prediction_length=1, # Numero minimo di osservazioni da predire
        max_prediction_length=max_prediction_length, # Numero di osservazioni da predire
        time_varying_known_reals=['time_idx'], # Parametri che cambiano nel tempo e di cui si conosce il valore futuro
        time_varying_unknown_reals=['height'], # Parametri che cambiano nel tempo e di cui non si conosce il valore futuro
        target_normalizer=GroupNormalizer(
            groups=['month'],
            transformation="softplus"
        ),
        add_relative_time_idx=True, # Aggiunge il time_idx alle features
        add_target_scales=True, # Aggiunge la media al target
        allow_missing_timesteps=True # Consente serie temporali interrotte
    )

    # Divisione del dataset in training e validation (l'aggiunta dei workers consente il lavoro in parallelo)
    validation = TimeSeriesDataSet.from_dataset(training, dataset, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=10, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=10, persistent_workers=True)

    """-------------------------------------------------------------------------------------------------
    Le seguenti istruzioni hanno l'obiettivo di analizzare il dataset e il modello da utilizzare
    per determinare il valore di learning rate ottimale da utilizzare per la fase di training.
    Una volta trovato il valore, lo si memorizza ed è possibile visualizzarne il grafico.
    Il valore è cercato nell'intervallo [0.01, 0.0001] consigliato nel paper di riferimento del TFT.
    -------------------------------------------------------------------------------------------------"""
    '''
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
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
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
    study.optimize(objective, n_trials=5)

    # Monitoro il training per interromperlo quando la validation loss raggiunge un valore delta minimo
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
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
    print("Validation loss finale:", trainer.callback_metrics['val_loss'].item())

    # Carico il TFT con i pesi ottimali
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Salvo il modello ottimale
    torch.save(best_tft, "Migliori_Modelli/Romeo_Best_TFT_Height.pth")
    '''
    # Carico il miglior modello e lo metto in fase di validazione
    best_tft = torch.load("Migliori_Modelli/Romeo_Best_TFT_Height.pth")
    best_tft.eval()
    """-------------------------------------------------------------------------------------------------
    La seguente parte di codice mette in evidenza la fase di validazione del modello.
    Come prima cosa si ricavano le predizioni del modello e le si visualizzano graficamente.
    Successivamente si visualizzano le interpretazioni del modello in termini di attenzione e importanza
    di ogni parametro del dataset.
    Infine è visualizzato un grafico che mette a confronto i valori predetti e quelli effettivi.
    -------------------------------------------------------------------------------------------------"""
    # Stampa delle predizioni
    predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    best_tft.plot_prediction(predictions.x, predictions.output, idx=0, add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles))
    plt.show()

    '''
    best_tft.plot_prediction = plot_prediction
    for idx in range(len(predictions.output)):
        best_tft.plot_prediction(predictions.x, predictions.output, idx=0, loss=best_tft.loss)
    plt.show()
    '''

    # Stampa delle interpretazioni (importanza e attenzione)
    interpretation = best_tft.interpret_output(predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)
    plt.show()

    # Stampa del confronto predizioni-valori effettivi
    predictions = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output, normalize=False)
    values_actual = predictions_vs_actuals["average"]["actual"]["height"].cpu()
    values_prediction = predictions_vs_actuals["average"]["prediction"]["height"].cpu()
    values_actual = values_actual[values_actual != 0]
    values_prediction = values_prediction[values_prediction != 0]
    plt.figure(figsize=(10, 6))
    plt.plot(values_actual, label="Valori Effettivi", color="blue")
    plt.plot(values_prediction, label="Predizioni", color="red")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("Confronto tra Valori Effettivi e Predizioni (non normalizzati)", fontweight="bold")
    plt.xlabel("time_idx")
    plt.ylabel("height")
    plt.legend()
    plt.show()

    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals, name="height")
    plt.show()

    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
    values_actual = predictions_vs_actuals["average"]["actual"]["height"].cpu()
    values_prediction = predictions_vs_actuals["average"]["prediction"]["height"].cpu()
    values_actual = values_actual[values_actual != 0]
    values_prediction = values_prediction[values_prediction != 0]
    plt.figure(figsize=(10, 6))
    plt.plot(values_actual, label="Valori Effettivi", color="blue")
    plt.plot(values_prediction, label="Predizioni", color="red")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("Confronto tra Valori Effettivi e Predizioni (normalizzati)", fontweight="bold")
    plt.xlabel("time_idx")
    plt.ylabel("height")
    plt.legend()
    plt.show()
    predictions_dataframe = pd.DataFrame({
        "Valori Effettivi (Height)": values_actual,
        "Predizioni (Height)": values_prediction
    })
    predictions_dataframe.to_csv("Predizioni_Height.csv", index=True)
    print(tabulate(predictions_dataframe, headers='keys', tablefmt='fancy_grid'))
