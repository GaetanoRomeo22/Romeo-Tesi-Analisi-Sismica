import pandas as pd
from tabulate import tabulate

# Apertura e lettura del file
file_name = input("Inserire il nome del file con l'estensione .rtl: ")
with open(file_name, "r") as file:
    lines = file.readlines()

# Estrapolazione dei dati dal file
info = ["ID", "Time", "Date", "Grid Easting", "Meters", "Grid Northing", "Meters", "GPS Quality",
        "Number of Satellities", "Position Quality", "Height", "Meters"]  # Etichette delle informazioni
data = []
for line in lines:
    field = line.strip().split(",") # Rimuovo gli spazi all'inizio e alla fine e divido per le virgole
    if len(field) == len(info): # Controllo che la riga abbia un numero valido di informazioni
        data.append(field)

# Creo un dataframe
data_frame = pd.DataFrame(data, columns=info)

# Converto la colonna 'time' nel formato HH:MM:SS
data_frame['Time'] = pd.to_datetime(data_frame['Time'], format='%H%M%S.%f', errors='coerce').dt.strftime('%H:%M:%S')

# Converto la colonna 'date' nel formato DD-MM-YYYY
data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%m%d%y', errors='coerce').dt.strftime('%d-%m-%Y')

# Rimuovo le righe contenenti valori invalidi dal file
data_frame.dropna(subset=['Time', 'Date'], inplace=True)

# Modifico l'estensione del file in .csv
file_name_csv = file_name.replace(".rtl", ".csv")

# Salvo il dataframe in formato .csv
data_frame.to_csv(file_name_csv, index=False)

# Visualizzo il dataframe creato
print(tabulate(data_frame, headers='keys', tablefmt='fancy_grid'))
