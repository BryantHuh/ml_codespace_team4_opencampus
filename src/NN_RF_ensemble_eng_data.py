import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# === Dateipfade ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
pickle_dir = os.path.join(project_root, "data", "eng_pickle_data")
nn_model_path = os.path.join(project_root, "models", "best_nn_model.keras")

# === Pickles laden ===
X_test = pd.read_pickle(os.path.join(pickle_dir, "test_features.pkl"))
y_test = pd.read_pickle(os.path.join(pickle_dir, "test_labels.pkl"))
X_train = pd.read_pickle(os.path.join(pickle_dir, "training_features.pkl"))
y_train = pd.read_pickle(os.path.join(pickle_dir, "training_labels.pkl"))

# Entferne datetime-Spalten (z. B. "Datum") aus den Features
for df in [X_train, X_test]:
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns
    df.drop(columns=datetime_cols, inplace=True)

# One-Hot-Encoding für kategoriale Spalten (z. B. Temp_Step)
for df in [X_train, X_test]:
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].astype("category")
    df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)

# === Modelle laden / trainieren ===

## RF trainieren
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

## NN laden und vorhersagen
nn = load_model(nn_model_path)
nn_preds = nn.predict(X_test).flatten()

# === Ensemble (Mittelwert) ===
ensemble_preds = (rf_preds + nn_preds) / 2

# === Bewertung ===
rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
r2 = r2_score(y_test, ensemble_preds)

print(f"Ensemble RMSE: {rmse:.2f}")
print(f"Ensemble R²: {r2:.3f}")