import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
# Use keras directly for model loading to avoid import errors
from keras.models import load_model
import joblib

# --- 1. Load Test Data ---
data_dir = os.path.join(os.path.dirname(__file__), '../../data/imputated_pickle')
X_test = pd.read_pickle(os.path.join(data_dir, 'test_features.pkl'))

# --- Ensure correct id column in test set ---
X_test['Datum'] = pd.to_datetime(X_test['Datum'])
X_test['Warengruppe'] = X_test['Warengruppe'].astype(int)
X_test['id'] = X_test.apply(lambda row: int(row['Datum'].strftime('%y%m%d') + str(row['Warengruppe'])), axis=1)

# --- Check id coverage against sample_submission.csv ---
sample_sub_path = os.path.join(os.path.dirname(__file__), '../../data/sample_submission.csv')
sample_submission = pd.read_csv(sample_sub_path)
missing_ids = set(sample_submission['id']) - set(X_test['id'])
extra_ids = set(X_test['id']) - set(sample_submission['id'])
print(f'Missing IDs in test: {len(missing_ids)}')
if len(missing_ids) > 0:
    print(f'Example missing IDs: {list(missing_ids)[:10]}')
print(f'Extra IDs in test: {len(extra_ids)}')
if len(extra_ids) > 0:
    print(f'Example extra IDs: {list(extra_ids)[:10]}')

# --- 2. Prepare Features (drop non-numeric/object columns for RF) ---
drop_cols = []
for col in X_test.columns:
    dtype = X_test[col].dtype
    if X_test[col].dtype == 'object' or (isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.datetime64)):
        drop_cols.append(col)
if 'FerienName' in X_test.columns:
    drop_cols.append('FerienName')
if 'id' in X_test.columns:
    drop_cols.append('id')
X_test_rf = X_test.drop(columns=drop_cols, errors='ignore')

# --- 3. Scale Features (use same scaler as training) ---
X_train = pd.read_pickle(os.path.join(data_dir, 'training_features.pkl'))
y_train = pd.read_pickle(os.path.join(data_dir, 'training_labels.pkl'))
X_train['Datum'] = pd.to_datetime(X_train['Datum'])
X_train['Warengruppe'] = X_train['Warengruppe'].astype(int)
X_train['id'] = X_train.apply(lambda row: int(row['Datum'].strftime('%y%m%d') + str(row['Warengruppe'])), axis=1)
X_train_rf = X_train.drop(columns=drop_cols, errors='ignore')
# Ensure columns are in the same order
X_test_rf = X_test_rf[X_train_rf.columns]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rf)
X_test_scaled = scaler.transform(X_test_rf)

# --- 4. Load and Predict with Random Forest ---
# Retrain RF on all data (for demonstration, in practice, load from file)
# Here, we just fit a new model for the test set
X_train = pd.read_pickle(os.path.join(data_dir, 'training_features.pkl'))
y_train = pd.read_pickle(os.path.join(data_dir, 'training_labels.pkl'))
X_train_rf = X_train.drop(columns=drop_cols, errors='ignore')
X_train_scaled = scaler.fit_transform(X_train_rf)
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)
y_test_rf = rf.predict(X_test_scaled)

# --- 5. Prepare Features for Neural Net (one-hot, scale) ---
# Load encoder and scaler from disk (fitted during training)
encoder = joblib.load(os.path.join(data_dir, 'encoder_nn.joblib'))
scaler_nn = joblib.load(os.path.join(data_dir, 'scaler_nn.joblib'))
categorical_cols = joblib.load(os.path.join(data_dir, 'categorical_cols_nn.joblib'))
numeric_cols = joblib.load(os.path.join(data_dir, 'numeric_cols_nn.joblib'))
# Use the exact same columns and order as in training
X_test_cat_df = pd.DataFrame(X_test[categorical_cols])
# Ensure all categorical columns are string type and have no NaNs
for col in categorical_cols:
    if col in X_test_cat_df.columns:
        X_test_cat_df[col] = X_test_cat_df[col].astype(str).fillna('missing')
X_test_cat = encoder.transform(X_test_cat_df)
X_test_num = X_test[numeric_cols]
X_test_num_arr = np.asarray(X_test_num)
X_test_cat_arr = np.asarray(X_test_cat)
X_test_final = np.hstack([X_test_num_arr, X_test_cat_arr])
print('X_test_final shape:', X_test_final.shape)
print('Expected shape:', scaler_nn.mean_.shape)
X_test_final = X_test_final.astype(float)
X_test_scaled_nn = scaler_nn.transform(X_test_final)

# --- 6. Load and Predict with Neural Net ---
model_path = os.path.join(data_dir, 'nn_model_imputed_onehot.keras')
model = load_model(model_path)
if model is None:
    raise RuntimeError(f'Failed to load neural network model from {model_path}')
y_test_nn = model.predict(X_test_scaled_nn).flatten()

# --- 7. Create Submission Files ---
# Use 'id' column if available, else default index
if 'id' in X_test.columns:
    ids = X_test['id']
else:
    ids = np.arange(len(X_test))

submission_rf = pd.DataFrame({'id': ids, 'Umsatz': y_test_rf})
submission_nn = pd.DataFrame({'id': ids, 'Umsatz': y_test_nn})

# --- 8. Align with sample_submission.csv ---
sample_sub_path = os.path.join(os.path.dirname(__file__), '../../data/sample_submission.csv')
sample_submission = pd.read_csv(sample_sub_path)
# Merge predictions with sample_submission to ensure correct order and count
submission_rf = pd.DataFrame({'id': ids, 'Umsatz': y_test_rf})
submission_nn = pd.DataFrame({'id': ids, 'Umsatz': y_test_nn})
# Only keep rows with ids in sample_submission, and in the same order
submission_rf = sample_submission[['id']].merge(submission_rf, on='id', how='left')
submission_nn = sample_submission[['id']].merge(submission_nn, on='id', how='left')

submission_rf.to_csv(os.path.join(data_dir, 'submission_rf.csv'), index=False)
submission_nn.to_csv(os.path.join(data_dir, 'submission_nn.csv'), index=False)

print('Submissions created:')
print('  submission_rf.csv')
print('  submission_nn.csv')