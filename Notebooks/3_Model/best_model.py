

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# Pfade
import os
# Verzeichnis mit Pickle-Dateien
pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/pickle_data"))
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/best_nn_model.keras"))

# Daten laden
X_train = pd.read_pickle(f"{pickle_dir}/training_features.pkl")
X_test = pd.read_pickle(f"{pickle_dir}/test_features.pkl")
y_test = pd.read_pickle(f"{pickle_dir}/test_labels.pkl")

# Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bestes Modell laden
model = load_model(model_path)
model.summary()

for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} – {layer.__class__.__name__}")
    try:
        print("  Units:", layer.units)
        print("  Activation:", layer.activation.__name__)
    except AttributeError:
        pass
# Vorhersage
y_pred = model.predict(X_test_scaled).flatten()

# Bewertung
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Vorhersage abgeschlossen\nRMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")