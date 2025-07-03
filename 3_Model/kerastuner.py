import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel, RandomSearch

# Daten laden
import os
# Verzeichnis mit Pickle-Dateien
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/pickle_data"))
pickle_dir = root_dir

X_train = pd.read_pickle(f"{pickle_dir}/training_features.pkl")
y_train = pd.read_pickle(f"{pickle_dir}/training_labels.pkl")
X_val = pd.read_pickle(f"{pickle_dir}/validation_features.pkl")
y_val = pd.read_pickle(f"{pickle_dir}/validation_labels.pkl")

# Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# HyperModel definieren
class RegressionHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32),
                        activation='relu', input_shape=(X_train_scaled.shape[1],)))
        if hp.Boolean("dropout_1"):
            model.add(Dropout(0.2))
        model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32),
                        activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='mse',
            metrics=['mae']
        )
        return model

# Tuner starten
tuner = RandomSearch(
    RegressionHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory='kerastuner_dir',
    project_name='bakery_sales'
)

# Suche durchführen
tuner.search(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=50, batch_size=16)

# Bestes Modell und Bewertung
best_model = tuner.get_best_models(num_models=1)[0]
val_loss, val_mae = best_model.evaluate(X_val_scaled, y_val)
print(f"Bestes Modell: Val Loss = {val_loss:.2f}, MAE = {val_mae:.2f}")
import os
model_dir = os.path.join(os.path.dirname(__file__), "./models")
os.makedirs(model_dir, exist_ok=True)
best_model.save(os.path.join(model_dir, "best_nn_model.keras"))
print("✅ Bestes Modell gespeichert als 'best_nn_model.keras'")
