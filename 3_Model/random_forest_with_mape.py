import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Load Data ---
data_dir = os.path.join(os.path.dirname(__file__), '../../data/imputated_pickle')
X_train = pd.read_pickle(os.path.join(data_dir, 'training_features.pkl'))
y_train = pd.read_pickle(os.path.join(data_dir, 'training_labels.pkl'))
X_val = pd.read_pickle(os.path.join(data_dir, 'validation_features.pkl'))
y_val = pd.read_pickle(os.path.join(data_dir, 'validation_labels.pkl'))

# --- 2. Drop non-numeric columns (e.g., Datum, FerienName) ---
drop_cols = []
for col in X_train.columns:
    dtype = X_train[col].dtype
    # Only call issubdtype if dtype is a numpy dtype
    if X_train[col].dtype == 'object' or (isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.datetime64)):
        drop_cols.append(col)
if 'FerienName' in X_train.columns:
    drop_cols.append('FerienName')
if drop_cols:
    print('Dropping columns:', drop_cols)
X_train_rf = X_train.drop(columns=drop_cols)
X_val_rf = X_val.drop(columns=drop_cols)

# --- 3. Feature Scaling (optional for RF, but can help) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rf)
X_val_scaled = scaler.transform(X_val_rf)

# --- 4. Train Random Forest ---
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

# --- 5. Predict and Evaluate ---
y_val_pred = rf.predict(X_val_scaled)

# Overall MAPE
overall_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Overall Validation MAPE: {overall_mape:.2f}%')

# MAPE by product group (Warengruppe)
if 'Warengruppe' in X_val.columns:
    print('\nMAPE by Product Group (Warengruppe):')
    warengruppe_vals = pd.Series(X_val['Warengruppe']).unique()
    for wg in sorted(warengruppe_vals):
        idx = X_val['Warengruppe'] == wg
        mape_wg = mean_absolute_percentage_error(y_val[idx], y_val_pred[idx]) * 100
        print(f'  Warengruppe {wg}: {mape_wg:.2f}%')
else:
    print('Warengruppe column not found in validation set.')

# --- 6. Feature Importance Plot (optional) ---
importances = rf.feature_importances_
feat_names = X_train_rf.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Random Forest)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [str(feat_names[i]) for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'rf_feature_importances.png'))
plt.show()