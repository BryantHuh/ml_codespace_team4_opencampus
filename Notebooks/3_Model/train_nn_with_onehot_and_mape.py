import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# --- 1. Load Data ---
data_dir = os.path.join(os.path.dirname(__file__), '../../data/imputated_pickle')
X_train = pd.read_pickle(os.path.join(data_dir, 'training_features.pkl'))
y_train = pd.read_pickle(os.path.join(data_dir, 'training_labels.pkl'))
X_val = pd.read_pickle(os.path.join(data_dir, 'validation_features.pkl'))
y_val = pd.read_pickle(os.path.join(data_dir, 'validation_labels.pkl'))

# --- 2. One-Hot Encoding for Categorical Features ---
categorical_cols = []
# Detect categorical columns (object, category, or known)
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
        categorical_cols.append(col)
# Add known categorical features
for col in ['Warengruppe', 'FerienName_Code', 'Temp_Step']:
    if col in X_train.columns and col not in categorical_cols:
        categorical_cols.append(col)
# Add 'Datum' to columns to drop if present (to avoid datetime in features)
if 'Datum' in X_train.columns and 'Datum' not in categorical_cols:
    categorical_cols.append('Datum')

# Ensure all categorical columns are DataFrames for concat
X_train_cat_df = pd.DataFrame(X_train[categorical_cols])
X_val_cat_df = pd.DataFrame(X_val[categorical_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(pd.concat([X_train_cat_df, X_val_cat_df], axis=0))

X_train_cat = encoder.transform(X_train_cat_df)
X_val_cat = encoder.transform(X_val_cat_df)

# Ensure one-hot encoded arrays are numpy arrays (not sparse matrices)
if hasattr(X_train_cat, 'toarray'):
    X_train_cat = X_train_cat.toarray()
if hasattr(X_val_cat, 'toarray'):
    X_val_cat = X_val_cat.toarray()

# Keep a copy of the original validation DataFrame for group-wise MAPE
X_val_orig = X_val.copy()

# Drop categorical columns and concatenate one-hot encoded
X_train_num = X_train.drop(columns=categorical_cols)
X_val_num = X_val.drop(columns=categorical_cols)

# Ensure all arrays for np.hstack are numpy arrays
X_train_num_arr = np.asarray(X_train_num)
X_val_num_arr = np.asarray(X_val_num)
X_train_cat_arr = np.asarray(X_train_cat)
X_val_cat_arr = np.asarray(X_val_cat)

X_train_final = np.hstack([X_train_num_arr, X_train_cat_arr])
X_val_final = np.hstack([X_val_num_arr, X_val_cat_arr])

# Ensure all data is float for StandardScaler
print('X_train_final dtype before scaling:', X_train_final.dtype)
print('X_train_final shape:', X_train_final.shape)
X_train_final = X_train_final.astype(float)
X_val_final = X_val_final.astype(float)

# --- 3. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val_final)

# --- 4. Define and Train Neural Network ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print('Neural Network Architecture:')
model.summary()

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# --- 5. Plot Loss Curves ---
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Neural Network Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'nn_loss_curves.png'))
plt.show()

# --- 6. MAPE Calculation ---
# Predict on validation set
y_val_pred = model.predict(X_val_scaled).flatten()
# Overall MAPE
overall_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Overall Validation MAPE: {overall_mape:.2f}%')

# MAPE by product group (Warengruppe)
if 'Warengruppe' in X_val_orig.columns:
    # Ensure we use the pandas Series for unique()
    warengruppe_vals = pd.Series(X_val_orig['Warengruppe']).unique()
    print('\nMAPE by Product Group (Warengruppe):')
    for wg in sorted(warengruppe_vals):
        idx = X_val_orig['Warengruppe'] == wg
        mape_wg = mean_absolute_percentage_error(y_val[idx], y_val_pred[idx]) * 100
        print(f'  Warengruppe {wg}: {mape_wg:.2f}%')
else:
    print('Warengruppe column not found in validation set.')

# --- 7. Save Model ---
model.save(os.path.join(data_dir, 'nn_model_imputed_onehot.keras'))