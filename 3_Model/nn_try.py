import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Verzeichnis mit Pickle-Dateien
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
pickle_dir = os.path.join(project_root, "data", "eng_pickle_data")

# Check für erforderliche Pickle-Dateien
required_files = [
    "training_features.pkl",
    "training_labels.pkl",
    "validation_features.pkl",
    "validation_labels.pkl",
    "test_features.pkl",
    "test_labels.pkl"
]

missing = [f for f in required_files if not os.path.isfile(os.path.join(pickle_dir, f))]
if missing:
    raise FileNotFoundError(f"❌ Fehlende Pickle-Dateien in {pickle_dir}: {missing}\n"
                            f"Bitte stelle sicher, dass die Datenvorbereitung ausgeführt wurde und alle Dateien vorhanden sind.")
else:
    print("✅ Alle Pickle-Dateien vorhanden.")

# Daten laden
X_train = pd.read_pickle(os.path.join(pickle_dir, "training_features.pkl"))
y_train = pd.read_pickle(os.path.join(pickle_dir, "training_labels.pkl"))
X_val = pd.read_pickle(os.path.join(pickle_dir, "validation_features.pkl"))
y_val = pd.read_pickle(os.path.join(pickle_dir, "validation_labels.pkl"))
X_test = pd.read_pickle(os.path.join(pickle_dir, "test_features.pkl"))
y_test = pd.read_pickle(os.path.join(pickle_dir, "test_labels.pkl"))

# Optional: Objektspalten in numerische umwandeln
for df in [X_train, X_val, X_test]:
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].astype("category")
    df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)

# Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Modell definieren
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(96, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# Modell speichern
model_save_path = os.path.join(project_root, "models", "best_nn_model_eng.keras")
model.save(model_save_path)
print(f"✅ Modell gespeichert unter: {model_save_path}")
