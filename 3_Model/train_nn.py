

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Daten laden
pickle_dir = os.path.join(os.path.dirname(__file__), "../../data/eng_pickle_data")
X_train = pd.read_pickle(os.path.join(pickle_dir, "training_features.pkl"))
y_train = pd.read_pickle(os.path.join(pickle_dir, "training_labels.pkl"))
X_val = pd.read_pickle(os.path.join(pickle_dir, "validation_features.pkl"))
y_val = pd.read_pickle(os.path.join(pickle_dir, "validation_labels.pkl"))
X_test = pd.read_pickle(os.path.join(pickle_dir, "test_features.pkl"))
y_test = pd.read_pickle(os.path.join(pickle_dir, "test_labels.pkl"))

# Datetime-Spalten entfernen
for df in [X_train, X_val, X_test]:
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns
    df.drop(columns=datetime_cols, inplace=True)

# Objekt- und Kategoriespalten in numerische Werte umwandeln
for df in [X_train, X_val, X_test]:
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].astype("category")
    df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)

# Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Modell definieren (beste Tuner-Architektur)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(96, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Training
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation auf Testdaten
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test-RMSE: {rmse:.2f}")
print(f"Test-R²: {r2:.3f}")

# Modell speichern
model_dir = os.path.join(os.path.dirname(__file__), "../../models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "best_nn_final.keras"))