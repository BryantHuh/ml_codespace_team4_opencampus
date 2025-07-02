# Weekly Tasks Directory

This directory contains weekly assignments, experiments, and related data for the project.

## Contents
- **week5.ipynb, week5fail.ipynb, week7_neurolnet.ipynb, week7_prep.ipynb**: Notebooks for weekly tasks and experiments.
- **desc_statistics.ipynb**: Descriptive statistics notebook.
- **splitdataset.ipynb**: Dataset splitting notebook.
- **submission_*.csv**: Submission files for various models.
- **python_model.h5**: Saved model file.
- **pickle_data/**: Pickled feature and label sets for weekly tasks.
- **README.md**: This file.

See subdirectory README for more details.

# Task Checklist

## 📊 Data Preparation & Cleaning
- [ ] Aggregate relevant data (e.g., Holiday list)
- [ ] Cluster weather codes
- [ ] Check for zero values in the sales column
- [ ] Enrich dataset with additional variables that may be relevant for estimating revenue

## 🧪 Dataset Splitting
- [ ] Split dataset:
  - [ ] Training set: 01.07.2013 – 31.07.2017
  - [ ] Validation set: 01.08.2017 – 31.07.2018
  - [ ] Test set: 01.08.2018 – 31.07.2019

## 📈 Modeling
- [ ] Use a simple linear model to predict the test dataset
- [ ] Evaluate the model performance (aim for R² as close to 1 as possible)
- [ ] Maximize adjusted R² on the training set
- [ ] Document linear regression calculations in the "Baseline Model" directory

## 📤 Submission
- [ ] Create `submission.csv` file
- [ ] Ensure row count and IDs match the [sample_submission](https://www.kaggle.com/competitions/bakery-sales-prediction-summer-2025/data?select=sample_submission.csv)
- [ ] Upload and evaluate the prediction on [Kaggle](https://www.kaggle.com/competitions/bakery-sales-prediction-summer-2025)