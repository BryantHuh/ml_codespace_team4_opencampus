import pandas as pd
import numpy as np
from datetime import datetime
import os

# Ursprüngliche CSVs laden
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
umsatz = pd.read_csv(os.path.join(project_root, "data", "umsatzdaten_gekuerzt.csv"), parse_dates=["Datum"])
wetter = pd.read_csv(os.path.join(project_root, "data", "wetter.csv"), parse_dates=["Datum"])
kiwo = pd.read_csv(os.path.join(project_root, "data", "kiwo.csv"), parse_dates=["Datum"])

# Feiertage mit workalendar bestimmen
from workalendar.europe import Germany
cal = Germany(prov="SH")

startjahr = umsatz["Datum"].dt.year.min()
endjahr = umsatz["Datum"].dt.year.max()

feiertage_liste = []
for jahr in range(startjahr, endjahr + 1):
    feiertage_liste.extend(cal.holidays(jahr))
feiertage_df = pd.DataFrame(feiertage_liste, columns=["Datum", "FeiertagName"])
feiertage_df["Feiertag"] = 1
ferien = pd.read_csv(os.path.join(project_root, "data", "ferien_sh.csv"), parse_dates=["Startdatum", "Enddatum"])

# Ferien-Tage extrahieren
ferien_tage = set()
for _, row in ferien.iterrows():
    ferien_tage.update(pd.date_range(start=row["Startdatum"], end=row["Enddatum"]))
ferien_df = pd.DataFrame({"Datum": sorted(ferien_tage), "Ferienzeit": 1})

# Ensure 'Datum' in feiertage_df is datetime
feiertage_df["Datum"] = pd.to_datetime(feiertage_df["Datum"])

# Merge
df = pd.merge(umsatz, wetter, on="Datum", how="left")
df = pd.merge(df, kiwo, on="Datum", how="left")
df = pd.merge(df, feiertage_df[["Datum", "Feiertag"]], on="Datum", how="left")
df = pd.merge(df, ferien_df, on="Datum", how="left")
df["KielerWoche"] = df["KielerWoche"].fillna(0)
df["Feiertag"] = df["Feiertag"].fillna(0)
df["Ferienzeit"] = df["Ferienzeit"].fillna(0)

# Zeitfeatures
df["Datum"] = pd.to_datetime(df["Datum"])
df["Wochentag"] = df["Datum"].dt.weekday
df["Monat"] = df["Datum"].dt.month
df["IstWochenende"] = (df["Wochentag"] >= 5).astype(int)

# === Zeitbasierte Features ===
df["KW"] = df["Datum"].dt.isocalendar().week
df["TagSeitWochenstart"] = df["Datum"].dt.weekday
df["Sin_Monat"] = np.sin(2 * np.pi * df["Monat"] / 12)
df["Cos_Monat"] = np.cos(2 * np.pi * df["Monat"] / 12)

# === Wetterkombinationen & Indikatoren ===
df["Wetter_extrem"] = ((df["Temperatur"] < 0) | (df["Temperatur"] > 30)).astype(int)
df["Temp_Step"] = pd.cut(df["Temperatur"], bins=[-10, 5, 20, 35], labels=["kalt", "mild", "heiß"])
df["Temp_Step"] = df["Temp_Step"].astype("category").cat.codes

# === Fehlende Werte behandeln ===
df["Bewoelkung"] = df["Bewoelkung"].interpolate(limit_direction="both")
df["Temperatur"] = df["Temperatur"].interpolate(limit_direction="both")
df["Windgeschwindigkeit"] = df["Windgeschwindigkeit"].fillna(df["Windgeschwindigkeit"].median())
df["Wettercode"] = df.groupby(df["Monat"])["Wettercode"].transform(lambda x: x.fillna(x.median()))
df["Temp_Wind"] = df["Temperatur"] * df["Windgeschwindigkeit"]

# Optional: Feature falls Wetterwerte gefehlt haben
df["WetterFehlend"] = df[["Bewoelkung", "Temperatur", "Windgeschwindigkeit", "Wettercode"]].isna().any(axis=1).astype(int)

# === Speichern ===
output_path = os.path.join(project_root, "data", "eng_data.pkl")
df.to_pickle(output_path)

print("Noch fehlende Werte pro Spalte:")
print(df.isnull().sum()[df.isnull().sum() > 0])

assert not df.isnull().any().any(), "❌ DataFrame enthält noch fehlende Werte!"

# Pickle-Daten für Training/Validierung/Test
from sklearn.model_selection import train_test_split
os.makedirs(os.path.join(project_root, "data", "eng_pickle_data"), exist_ok=True)

features = df.drop(columns=["Umsatz", "id"])
labels = df["Umsatz"]

X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

X_train.to_pickle(os.path.join(project_root, "data", "eng_pickle_data", "training_features.pkl"))
y_train.to_pickle(os.path.join(project_root, "data", "eng_pickle_data", "training_labels.pkl"))
X_val.to_pickle(os.path.join(project_root, "data", "eng_pickle_data", "validation_features.pkl"))
y_val.to_pickle(os.path.join(project_root, "data", "eng_pickle_data", "validation_labels.pkl"))
X_test.to_pickle(os.path.join(project_root, "data", "eng_pickle_data", "test_features.pkl"))
y_test.to_pickle(os.path.join(project_root, "data", "eng_pickle_data", "test_labels.pkl"))

print("✅ Feature Engineering abgeschlossen – Datei gespeichert unter:", output_path)
