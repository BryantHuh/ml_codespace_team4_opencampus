import pandas as pd

# Daten laden
df_umsatz = pd.read_csv("data/umsatzdaten_gekuerzt.csv")
df_wetter = pd.read_csv("data/wetter.csv")
df_kiwo = pd.read_csv("data/kiwo.csv")

# Überblick über jede Tabelle
for name, df in zip(['Umsatz', 'Wetter', 'KiWo'], [df_umsatz, df_wetter, df_kiwo]):
    print(f"\n=== {name} ===")
    print(df.info())
    print(df.head())