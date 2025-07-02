# Project File Summary

## Root Directory
- **PROJECT_FILE_SUMMARY.md**: This file. Summarizes the purpose of all files in the project.
- **README.md**: Project overview and instructions.
- **.gitignore**: Specifies files and directories to be ignored by git.
- **.DS_Store**: macOS system file for folder view settings (not relevant to project).

## models/
- **best_nn_final.keras**: Saved Keras neural network model (final version).
- **best_nn_model.keras**: Saved Keras neural network model (best from tuning).

## data/
- **eng_data.pkl**: Pickled DataFrame with engineered features.
- **engineered_data.pkl**: Pickled DataFrame, likely similar to eng_data.pkl.
- **df_prepared.csv**: Prepared dataset in CSV format.
- **ferien_sh.csv**: CSV with school holiday dates for Schleswig-Holstein.
- **umsatzdaten_gekuerzt.csv**: Main sales data CSV.
- **wetter.csv**: Weather data CSV.
- **train.csv**: Training data CSV (possibly for Kaggle-style split).
- **kiwo.csv**: Kieler Woche event data CSV.
- **sample_submission.csv**: Example submission file (for competitions).
- **test.csv**: Test data CSV (possibly for Kaggle-style split).
- **eng_pickle_data/**: Contains pickled train/val/test features and labels.
- **pickle_data/**: Contains pickled train/val/test features and labels (possibly from a different pipeline).

### data/eng_pickle_data/
- **training_features.pkl, training_labels.pkl, validation_features.pkl, validation_labels.pkl, test_features.pkl, test_labels.pkl**: Pickled feature and label sets for model training, validation, and testing.

### data/pickle_data/
- **training_features.pkl, training_labels.pkl, validation_features.pkl, validation_labels.pkl, test_features.pkl, test_labels.pkl**: Another set of pickled feature and label sets.

## kerastuner_dir/
- **bakery_sales/**: Directory for Keras Tuner results.
  - **tuner0.json, oracle.json**: Keras Tuner configuration and search state.
  - **trial_00/ to trial_09/**: Each contains build config, checkpoint weights, and trial metadata for a specific hyperparameter search trial.

## docs/
- **Dokumentation Projektdatensatz.pdf**: PDF documentation for the project dataset.
- **.DS_Store**: macOS system file (not relevant).
- **ForestPlots/output.png**: Output plot for forest model results.
- **BaselinePlots/tats_vorh.png, coefficients.png**: Plots for baseline model analysis.

## Notebooks/
- **.DS_Store**: macOS system file (not relevant).
- **0_DataPreparation/**: Data loading, cleaning, and feature engineering scripts and notebooks.
- **1_DatasetCharacteristics/**: Notebooks for missing value analysis and dataset statistics.
- **2_BaselineModel/**: Baseline model notebook(s).
- **3_Model/**: Model training, tuning, and evaluation scripts and notebooks.
- **4_Presentation/**: (No files listed, likely for presentation material.)

### Notebooks/0_DataPreparation/
- **feat_eng.py**: Feature engineering script (see earlier summary).
- **base_data.py**: Loads and prints info for main CSVs.
- **try.py**: Linear regression on house pricing data (example).
- **week2.ipynb**: Intro data science task, data loading, and analysis.
- **week3.ipynb**: Data merging, mapping, and visualization.
- **weather_codes.ipynb**: Weather code analysis and visualization.
- **prepare_data_for_models.ipynb, prepare_data_for_models_pickles.ipynb**: Notebooks for preparing data for modeling and saving as pickles.
- **missing_value_imputation.ipynb**: Notebook for handling missing values in the data.
- **add_calendar_features.ipynb**: Adds calendar-based features to the dataset.
- **data_merge.ipynb**: Merges multiple data sources into a single DataFrame.

### Notebooks/1_DatasetCharacteristics/
- **missing_value_analysis.ipynb**: Analyzes missing values in the dataset.
- **statistics_and_structure.ipynb**: Explores dataset statistics and structure.

### Notebooks/2_BaselineModel/
- **linear_regression_baseline.ipynb**: Baseline linear regression model notebook.

### Notebooks/3_Model/
- **kerastuner.py**: Keras Tuner script for neural network hyperparameter search.
- **train_nn.py**: Trains a neural network on engineered features.
- **best_model.py**: Loads and evaluates the best neural network model.
- **nn_try.py**: Neural network training script with robust data loading.
- **neural_network_model.ipynb**: Notebook for neural network modeling.
- **random_forest_model.ipynb**: Notebook for random forest modeling.
- **subm_rand_forest.ipynb**: Notebook for random forest submission.
- **submission_randomforest.csv**: Submission file for random forest predictions.

## src/
- **NN_RF_ensemble_eng_data.py**: Ensemble script combining neural network and random forest predictions.
- **.gitkeep**: Placeholder file to keep the directory in git.

## weekly_tasks/
- **week5.ipynb, week5fail.ipynb, week7_neurolnet.ipynb, week7_prep.ipynb**: Weekly task notebooks (various analyses and experiments).
- **submission_statsmodels_multi.csv, submission_statsmodels_temp.csv, submission.csv, submission_multi_cleaned.csv**: Submission files for different models/experiments.
- **python_model.h5**: Saved Keras or TensorFlow model (HDF5 format).
- **splitdataset.ipynb**: Notebook for splitting the dataset.
- **desc_statistics.ipynb**: Descriptive statistics notebook.
- **README.md**: Documentation for weekly tasks.
- **pickle_data/**: Contains pickled feature and label sets for weekly tasks.

### weekly_tasks/pickle_data/
- **training_features.pkl, training_labels.pkl, validation_features.pkl, validation_labels.pkl, test_features.pkl, test_labels.pkl**: Pickled feature and label sets for weekly task experiments.

## example_notebooks/
- **neural_net_estimation.ipynb**: Example notebook for neural network estimation.
- **neural_net_data_preparation.ipynb**: Example notebook for data preparation for neural networks.
- **README.md**: Documentation for example notebooks.
- **pickle_data/**: Contains pickled feature and label sets for example notebooks.

### example_notebooks/pickle_data/
- **training_features.pkl, training_labels.pkl, validation_features.pkl, validation_labels.pkl, test_features.pkl, test_labels.pkl**: Pickled feature and label sets for example notebooks.

---

*If a file's purpose is unclear or it is a binary/model/data file, this is noted. For further details on any specific file, please ask!*