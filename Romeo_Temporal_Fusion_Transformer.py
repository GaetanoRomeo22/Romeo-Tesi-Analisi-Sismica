import csv
import os
from math import modf
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import optuna
from pytorch_forecasting.metrics import QuantileLoss
import matplotlib.pyplot as plt

'''
wandb.init(project="Romeo_Temporal_Fusion_Transformer")
logger = WandbLogger()
logger = TensorBoardLogger("lightning_logs")
'''

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
                        date = pd.to_datetime(file_data[2], format='%m%d%y').strftime('%d-%m-%Y')  # Estraggo la data
                        date_obj = pd.to_datetime(file_data[2], format='%m%d%y')  # Converto la stringa in oggetto datetime
                        if date_obj.month == 3 and date_obj.year == 2022 or date_obj.month == 1 and date_obj.year == 2023:  # Condizione per saltare marzo 2022
                            continue  # Salta questa riga se è marzo 2022
                        target = modf(float(file_data[10]))[0]  # Estraggo il valore di Grid Northing (parte decimale)
                        csv_writer.writerow(['$GPLLQ', time, date, target]) # Scrivo i dati su file
                    except IndexError:
                        print("Riga incompleta")

if __name__ == '__main__':
    # Caricamento e settaggio del dataset
    # rtl_to_csv("SOLO-LICO", "Dataset.csv") # Converto i file rtl in csv
    dataset = pd.read_csv("Dataset.csv")  # Creo un dataframe con pandas
    dataset = dataset.dropna()  # Rimuovo i campi NULL
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y') # Converto la data nel formato corretto
    # dataset = dataset.sort_values(by=['date', 'time'], ignore_index=True) # Ordino i dati in ordine cronologico
    dataset['month'] = dataset.date.dt.month.astype(str) # Aggiungo l'informazione relativa al mese
    dataset['year'] = dataset.date.dt.year.astype(str) # Aggiungo l'informazione relativa all'anno
    dataset['time_idx'] = dataset.groupby(['month']).cumcount()  # Aggiungo una colonna per il time index per il TFT
    print(dataset)  # Stampo alcuni dati
    train_cnt = int(len(dataset) * .8)  # 80% train e 20% test
    train = dataset.iloc[:train_cnt] # Dati di train
    test = dataset.iloc[train_cnt:] # Dati di test
    max_prediction_length = 168 # Numero di osservazioni da predire
    max_encoder_length = 720 # Numero di osservazioni da analizzare per le predizioni
    batch_size = 64

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

    def objective(trial): # Funzione per trovare i migliori pesi di addestramento
        hidden_size = trial.suggest_int("hidden_size", 8, 64, step=8)
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 64, step=8)
        dropout = trial.suggest_float("dropout", 0.3, 0.7)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        attention_head_size = trial.suggest_int("attention_head_size", 1, 4)

        tft = TemporalFusionTransformer.from_dataset( # Creo il TFT con i migliori pesi
            training,
            learning_rate=learning_rate,
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
            max_epochs=30,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[lr_logger, early_stop_callback],
            enable_checkpointing=True
        )

        trainer.fit( # Fase di addestramento
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        # Valutazione del modello in termini di validaiton loss
        val_loss = trainer.callback_metrics['val_loss'].item()
        return val_loss

    # Create a Study and specify the direction of optimization
    study = optuna.create_study(direction="minimize", storage="sqlite:///Romeo_Temporal_Fusion_Transformer.db")

    # Optimize the objective function
    study.optimize(objective, n_trials=3)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate

    trainer = pl.Trainer( # Definizione dei parametri di addestramento del TFT
        max_epochs=50,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback]
    )

    tft = TemporalFusionTransformer.from_dataset( # Definizione del TFT con i migliori parametri
        training,
        learning_rate=study.best_params['learning_rate'],
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

    best_model_path = trainer.checkpoint_callback.best_model_path
    # print("Best model path: ", best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Stampa delle predizioni
    predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    best_tft.plot_prediction(predictions.x, predictions.output, idx=0, add_loss_to_title=True)
    plt.show()

    # Stampa delle interpretazioni
    interpretation = best_tft.interpret_output(predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)
    plt.show()

    # Stampa del confronto predizioni-valori effettivi
    predictions = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    plt.show()

    '''
    torch.save(best_tft.state_dict(), 'Best_tft_model.pth') # Salvataggio del miglior modello

    new_tft = TemporalFusionTransformer.from_dataset( # Definizione del TFT
        training,
        learning_rate=0.004,
        hidden_size=5,
        attention_head_size=2,
        hidden_continuous_size=1,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="AdamW",
        reduce_on_plateau_patience=1,
        lstm_layers=3,
        dropout=0.5
    )
    new_tft.load_state_dict(torch.load('Best_tft_model.pth'), strict=False) # Ricarico il miglior modello
    new_tft.eval() # Metto il TFT in fase di validazione

    # Stampa delle predizioni
    predictions = new_tft.predict(val_dataloader, return_x=True)
    print("Prediction keys: ", predictions.keys())
    new_tft.plot_prediction(predictions.y, predictions.output, idx=0, add_loss_to_title=True)
    plt.show()

    # Stampa del confronto predizioni-valori effettivi
    predictions_vs_actuals = new_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
    new_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    plt.show()
    print("Prediction output: ", predictions.output)
    print(predictions.x['groups'])
    '''
