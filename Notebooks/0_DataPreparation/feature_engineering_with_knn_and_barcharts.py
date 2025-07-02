import pandas as pd
import numpy as np
import os
from workalendar.europe import Germany
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
data_dir = os.path.join(project_root, 'data')
impute_dir = os.path.join(data_dir, 'imputated_pickle')
os.makedirs(impute_dir, exist_ok=True)

umsatz = pd.read_csv(os.path.join(data_dir, 'umsatzdaten_gekuerzt.csv'), parse_dates=['Datum'])
wetter = pd.read_csv(os.path.join(data_dir, 'wetter.csv'), parse_dates=['Datum'])
kiwo = pd.read_csv(os.path.join(data_dir, 'kiwo.csv'), parse_dates=['Datum'])
ferien = pd.read_csv(os.path.join(data_dir, 'ferien_sh.csv'), parse_dates=['Startdatum', 'Enddatum'])

# Ensure Datum columns are datetime for all merges
umsatz['Datum'] = pd.to_datetime(umsatz['Datum'])
wetter['Datum'] = pd.to_datetime(wetter['Datum'])
kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])

# --- 2. Feature Engineering ---
cal = Germany(prov='SH')
startjahr = umsatz['Datum'].dt.year.min()
endjahr = umsatz['Datum'].dt.year.max()
feiertage_liste = []
for jahr in range(startjahr, endjahr + 1):
    feiertage_liste.extend(cal.holidays(jahr))
feiertage_df = pd.DataFrame(feiertage_liste, columns=['Datum', 'FeiertagName'])
feiertage_df['Feiertag'] = 1
feiertage_df['Datum'] = pd.to_datetime(feiertage_df['Datum'])

ferien_tage = set()
for _, row in ferien.iterrows():
    ferien_tage.update(pd.date_range(start=row['Startdatum'], end=row['Enddatum']))
ferien_df = pd.DataFrame({'Datum': sorted(ferien_tage), 'Ferienzeit': 1})
ferien_df['Datum'] = pd.to_datetime(ferien_df['Datum'])

# --- Ferienzeit: Add FerienName feature (after df is defined) ---
ferien_name_map = {}
for _, row in ferien.iterrows():
    for date in pd.date_range(start=row['Startdatum'], end=row['Enddatum']):
        ferien_name_map[date] = row['Ferientyp']

# Merge all data sources into df
df = pd.merge(umsatz, wetter, on='Datum', how='left')
df = pd.merge(df, kiwo, on='Datum', how='left')
df = pd.merge(df, feiertage_df[['Datum', 'Feiertag']], on='Datum', how='left')
df = pd.merge(df, ferien_df, on='Datum', how='left')
df['KielerWoche'] = df['KielerWoche'].fillna(0)
df['Feiertag'] = df['Feiertag'].fillna(0)
df['Ferienzeit'] = df['Ferienzeit'].fillna(0)

# Ensure Datum is datetime for mapping
df['Datum'] = pd.to_datetime(df['Datum'])

# --- Ferienzeit: Add FerienName feature (after df is defined) ---
ferien_name_series = df['Datum'].map(lambda d: ferien_name_map.get(d, 'None'))
df['FerienName'] = ferien_name_series
ferien_name_cat = pd.Categorical(df['FerienName'])
df['FerienName_Code'] = ferien_name_cat.codes

df['Wochentag'] = df['Datum'].dt.weekday
df['Monat'] = df['Datum'].dt.month
df['IstWochenende'] = (df['Wochentag'] >= 5).astype(int)
df['KW'] = df['Datum'].dt.isocalendar().week
df['TagSeitWochenstart'] = df['Datum'].dt.weekday
df['Sin_Monat'] = np.sin(2 * np.pi * df['Monat'] / 12)
df['Cos_Monat'] = np.cos(2 * np.pi * df['Monat'] / 12)
df['Wetter_extrem'] = ((df['Temperatur'] < 0) | (df['Temperatur'] > 30)).astype(int)
df['Temp_Step'] = pd.cut(df['Temperatur'], bins=[-10, 5, 20, 35], labels=['kalt', 'mild', 'heiß'])
df['Temp_Step'] = df['Temp_Step'].astype('category').cat.codes
df['Temp_Wind'] = df['Temperatur'] * df['Windgeschwindigkeit']

# --- 3. KNN Imputation for Wettercode and Bewoelkung ---
df['Wettercode'] = df['Wettercode'].replace(0.0, np.nan)
knn_features = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Monat']
df_knn = df[['Wettercode'] + knn_features]
imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = imputer.fit_transform(df_knn)
df['Wettercode'] = np.round(df_knn_imputed[:, 0]).astype(int)
# Bewoelkung
# KNN imputation for Bewoelkung
bewoelkung_features = ['Bewoelkung', 'Temperatur', 'Wettercode', 'Monat', 'Windgeschwindigkeit']
df_bew = df[bewoelkung_features]
df['Bewoelkung'] = KNNImputer(n_neighbors=5).fit_transform(df_bew)[:, 0]

# --- 4. Imputation for Other Variables ---
df['Temperatur'] = df['Temperatur'].interpolate(limit_direction='both')
df['Windgeschwindigkeit'] = df['Windgeschwindigkeit'].fillna(df['Windgeschwindigkeit'].median())
df['Temp_Wind'] = df['Temperatur'] * df['Windgeschwindigkeit']

# --- 5. Save Pickled Data for Modeling ---
# Drop 'id' and 'Umsatz' if they exist
feature_cols = [col for col in df.columns if col not in ['id', 'Umsatz']]
features = df[feature_cols]
labels = df['Umsatz']

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Save as pickles
X_train.to_pickle(os.path.join(impute_dir, 'training_features.pkl'))
y_train.to_pickle(os.path.join(impute_dir, 'training_labels.pkl'))
X_val.to_pickle(os.path.join(impute_dir, 'validation_features.pkl'))
y_val.to_pickle(os.path.join(impute_dir, 'validation_labels.pkl'))
X_test.to_pickle(os.path.join(impute_dir, 'test_features.pkl'))
y_test.to_pickle(os.path.join(impute_dir, 'test_labels.pkl'))

# --- 6. Bar Charts for Engineered Variables ---
plt.figure(figsize=(6,4))
sns.barplot(data=df, x='IstWochenende', y='Umsatz', errorbar=('ci', 95), palette='Blues', hue='IstWochenende', legend=False)
plt.title('Average Sales by Weekend/Weekday')
plt.xlabel('IstWochenende (0=Weekday, 1=Weekend)')
plt.ylabel('Average Sales')
plt.tight_layout()
plt.savefig(os.path.join(impute_dir, 'barchart_IstWochenende.png'))
plt.close()

plt.figure(figsize=(6,4))
sns.barplot(data=df, x='Wetter_extrem', y='Umsatz', errorbar=('ci', 95), palette='Oranges', hue='Wetter_extrem', legend=False)
plt.title('Average Sales by Extreme Weather')
plt.xlabel('Wetter_extrem (0=Normal, 1=Extreme)')
plt.ylabel('Average Sales')
plt.tight_layout()
plt.savefig(os.path.join(impute_dir, 'barchart_Wetter_extrem.png'))
plt.close()

print('Feature engineering, KNN imputation, and bar chart generation complete. Pickles and plots saved in:', impute_dir)