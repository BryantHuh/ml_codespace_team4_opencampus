import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# --- 1. Load Test Data ---
data_dir = os.path.join(os.path.dirname(__file__), '../../data/imputated_pickle')
X_test = pd.read_pickle(os.path.join(data_dir, 'test_features.pkl'))

# --- 2. Prepare Features (drop non-numeric/object columns for RF) ---
drop_cols = []
for col in X_test.columns:
    dtype = X_test[col].dtype
    if X_test[col].dtype == 'object' or (isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.datetime64)):
        drop_cols.append(col)
if 'FerienName' in X_test.columns:
    drop_cols.append('FerienName')
X_test_rf = X_test.drop(columns=drop_cols)

# --- 3. Scale Features (use same scaler as training) ---
# For demonstration, fit scaler on test (in practice, use training scaler!)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_rf)

# --- 4. Load and Predict with Random Forest ---
# Retrain RF on all data (for demonstration, in practice, load from file)
# Here, we just fit a new model for the test set
X_train = pd.read_pickle(os.path.join(data_dir, 'training_features.pkl'))
y_train = pd.read_pickle(os.path.join(data_dir, 'training_labels.pkl'))
X_train_rf = X_train.drop(columns=drop_cols)
X_train_scaled = scaler.fit_transform(X_train_rf)
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)
y_test_rf = rf.predict(X_test_scaled)

# --- 5. Prepare Features for Neural Net (one-hot, scale) ---
from sklearn.preprocessing import OneHotEncoder
# Use same categorical_cols as in NN script
categorical_cols = []
for col in X_test.columns:
    if X_test[col].dtype == 'object' or X_test[col].dtype.name == 'category':
        categorical_cols.append(col)
for col in ['Warengruppe', 'FerienName_Code', 'Temp_Step']:
    if col in X_test.columns and col not in categorical_cols:
        categorical_cols.append(col)
if 'Datum' in X_test.columns and 'Datum' not in categorical_cols:
    categorical_cols.append('Datum')
X_test_cat_df = pd.DataFrame(X_test[categorical_cols])
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# Fit encoder on train+val (simulate with train here)
X_train_cat_df = pd.DataFrame(X_train[categorical_cols])
encoder.fit(X_train_cat_df)
X_test_cat = encoder.transform(X_test_cat_df)
X_test_num = X_test.drop(columns=categorical_cols)
X_test_final = np.hstack([X_test_num.to_numpy(), X_test_cat])
X_test_final = X_test_final.astype(float)
# Scale (fit scaler on train as above)
scaler_nn = StandardScaler()
X_train_num = X_train.drop(columns=categorical_cols)
X_train_cat = encoder.transform(X_train_cat_df)
X_train_final = np.hstack([X_train_num.to_numpy(), X_train_cat])
X_train_final = X_train_final.astype(float)
scaler_nn.fit(X_train_final)
X_test_scaled_nn = scaler_nn.transform(X_test_final)

# --- 6. Load and Predict with Neural Net ---
model_path = os.path.join(data_dir, 'nn_model_imputed_onehot.keras')
model = load_model(model_path)
y_test_nn = model.predict(X_test_scaled_nn).flatten()

# --- 7. Create Submission Files ---
# Use 'id' column if available, else default index
if 'id' in X_test.columns:
    ids = X_test['id']
else:
    ids = np.arange(len(X_test))

submission_rf = pd.DataFrame({'id': ids, 'Umsatz': y_test_rf})
submission_nn = pd.DataFrame({'id': ids, 'Umsatz': y_test_nn})

submission_rf.to_csv(os.path.join(data_dir, 'submission_rf.csv'), index=False)
submission_nn.to_csv(os.path.join(data_dir, 'submission_nn.csv'), index=False)

print('Submissions created:')
print('  submission_rf.csv')
print('  submission_nn.csv')