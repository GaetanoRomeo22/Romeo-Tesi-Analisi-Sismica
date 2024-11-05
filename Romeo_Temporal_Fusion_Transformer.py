import csv
import os
from math import modf
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import optuna
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
import matplotlib.pyplot as plt
from lightning.pytorch.tuner import Tuner

# Logger per wandb (richiede wandb e WandbLogger tra gli import)
# wandb.init(project="Romeo_Temporal_Fusion_Transformer")
# logger = WandbLogger()

# Logger per TensorBoard (richiede TensorBoard tra gli import)
# logger = TensorBoardLogger("lightning_logs")

def rtl_to_csv(folder_path, output_file):  # Converto il contenuto dei file rtl in un unico csv
    with open(output_file, mode='w', newline='') as csv_file:  # Apro il file di output in scrittura
        csv_writer = csv.writer(csv_file, delimiter=',')  # csv_writer per scrivere sul file
        csv_writer.writerow(['ID', 'time', 'date', 'height'])  # Scrivo i nomi dei campi da salvare
        for file in os.listdir(folder_path):  # Elenco tutti i file della directory
            file_path = os.path.join(folder_path, file)  # Ottengo il path del file
            with open(file_path, mode='r') as rtl_file:  # Apro il file rtl in lettura
                for row in rtl_file:  # Per ogni riga del file
                    try:
                        file_data = row.split(',')  # Divido il contenuto del rigo per la virgola
                        time_str = file_data[1].strip()  # Estraggo l'orario dalla colonna corretta e rimuovo spazi
                        time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}" # Converto il tempo in formato HH:MM:SS
                        date = pd.to_datetime(file_data[2], format='%m%d%y')  # Estraggo la data
                        if date.month == 3 and date.year == 2022 or date.month == 1 and date.year == 2023:  # Salta marzo 2022 e gennaio 2023
                            continue
                        target = modf(float(file_data[10]))[0]  # Estraggo il valore di Grid Northing (parte decimale)
                        csv_writer.writerow(['$GPLLQ', time, date.strftime('%d-%m-%Y'), target]) # Scrivo i dati su file
                    except IndexError:
                        print("Riga incompleta")

if __name__ == '__main__':
    # Caricamento e settaggio del dataset
    rtl_to_csv("SOLO-LICO", "Dataset.csv") # Converto i file rtl in csv
    dataset = pd.read_csv("Dataset.csv")  # Creo un dataframe con pandas
    dataset = dataset.dropna()  # Rimuovo eventuali record incompleti
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y') # Converto la data nel formato corretto
    dataset['month'] = dataset.date.dt.month.astype(str) # Aggiungo l'informazione relativa al mese
    dataset['year'] = dataset.date.dt.year.astype(str) # Aggiungo l'informazione relativa all'anno
    dataset['time_idx'] = dataset.groupby(['month']).cumcount()  # Aggiungo una colonna per il time index per il TFT
    print(dataset.head())  # Stampo alcuni dati
    train_cnt = int(len(dataset) * .8)  # 80% train e 20% test
    train = dataset.iloc[:train_cnt] # Dati di train
    test = dataset.iloc[train_cnt:] # Dati di test
    max_prediction_length = 24 # Numero di osservazioni da predire
    max_encoder_length = 168 # Numero di osservazioni da analizzare per le predizioni
    batch_size = 64
    epochs = 50 # Numero di epoche

    training = TimeSeriesDataSet( # Converto il dataset nel formato richiesto dal TFT
        train, # Dati di addestramento
        time_idx='time_idx', # Indice temporale per le serie temporali
        target='height', # Valore da predire
        group_ids=['month'], # Campo utilizzato per identificare univocamente le serie temporali (da modificare!!!!)
        min_encoder_length=max_encoder_length // 2, # Numero minimo di osservazioni da analizzare per le predizioni
        max_encoder_length=max_encoder_length, # Numero di osservazioni da analizzare per le predizioni
        min_prediction_length=1, # Numero minimo di osservazioni da predire
        max_prediction_length=max_prediction_length, # Numero di osservazioni da predire
        static_categoricals=['ID'], # Variabili statiche categoriche che non cambiano nel tempo
        time_varying_known_reals=['time_idx'], # Parametri che cambiano nel tempo e di cui si conosce il valore futuro
        time_varying_unknown_reals=['height'], # Parametri che cambiano nel tempo e di cui non si conosce il valore futuro
        target_normalizer=GroupNormalizer(
            groups=['month'],
            transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    # Divisione del dataset in training e validation
    validation = TimeSeriesDataSet.from_dataset(training, dataset, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=8, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=8, persistent_workers=True)

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
    # fig = res.plot(show=True, suggest=True)
    # fig.show()

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

    # Stampa delle predizioni
    predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True, return_y=True)
    best_tft.plot_prediction(predictions.x, predictions.output, idx=0, add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles))
    plt.show()

    # Stampa delle interpretazioni (importanza e attenzione)
    interpretation = best_tft.interpret_output(predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)
    plt.show()

    # Stampa del confronto predizioni-valori effettivi
    predictions = best_tft.predict(val_dataloader, return_x=True, return_y=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output, normalize=False)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    plt.show()
