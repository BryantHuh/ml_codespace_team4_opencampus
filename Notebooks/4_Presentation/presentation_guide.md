# Bakery Sales Forecasting – Presentation Guide

## 1. Self-Created Variables (Feature Engineering)

| Variable            | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| Wochentag           | Day of the week (0=Monday, ..., 6=Sunday)                          |
| Monat               | Month of the year (1–12)                                           |
| IstWochenende       | 1 if Saturday/Sunday, else 0                                       |
| KW                  | Calendar week number                                               |
| TagSeitWochenstart  | Day since start of week (0–6)                                      |
| Sin_Monat, Cos_Monat| Sine and cosine encoding of month (seasonality)                    |
| Wetter_extrem       | 1 if temperature <0°C or >30°C, else 0                             |
| Temp_Step           | Categorical binning of temperature (cold, mild, hot)               |
| Temp_Wind           | Product of temperature and wind speed                              |
| Ferienzeit          | 1 if school holiday, else 0                                        |
| FerienName, FerienName_Code | Name/code of the holiday period                            |
| Feiertag            | 1 if public holiday, else 0                                        |
| KielerWoche         | 1 if during Kieler Woche event, else 0                             |

---

## 2. Bar Charts with Confidence Intervals

- **IstWochenende (Weekend Effect):**
  - Shows average sales for weekends vs. weekdays.
  - ![Bar chart: IstWochenende](../data/imputated_pickle/barchart_IstWochenende.png)

- **Wetter_extrem (Extreme Weather):**
  - Shows average sales on days with extreme weather vs. normal weather.
  - ![Bar chart: Wetter_extrem](../data/imputated_pickle/barchart_Wetter_extrem.png)

---

## 3. Linear Model Optimization

- **Model Equation:**
  ```
  Umsatz = β₀ + β₁*Temperatur + β₂*Bewoelkung + β₃*Windgeschwindigkeit + β₄*KielerWoche + ... + ε
  ```
  *(Replace βs with actual coefficients from your linear regression output if available)*

- **Adjusted R²:**
  - Example: Adjusted R² = 0.72 (replace with your actual value)

---

## 4. Type of Missing Value Imputation Used

- **KNN Imputation:**
  - Used for `Wettercode` and `Bewoelkung` (KNNImputer with n_neighbors=5).
- **Median/Interpolation:**
  - `Temperatur`: Interpolated.
  - `Windgeschwindigkeit`: Filled with median.
- **Categorical:**
  - Categorical NaNs filled with mode or a default value after merging for the test set.

---

## 5. Neural Network Optimization

### a) Source Code Defining the Neural Network
```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.0005), input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.0005)),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
```

### b) Loss Function Plots for Training and Validation Sets
- ![NN Loss Curves](../data/imputated_pickle/nn_loss_curves.png)

### c) MAPE Scores for the Overall Validation Set and Each Product Group
- **Overall Validation MAPE:**
  - Example: 22.5% (replace with your actual value)
- **MAPE by Product Group (Warengruppe):**
  - Example:
    - Warengruppe 1: 18.2%
    - Warengruppe 2: 24.5%
    - Warengruppe 3: 21.0%
    - ...
  *(Replace with your actual output from the script)*

---

## 6. Presentation Tips
- Show the bar charts and explain the effect of the variables.
- Present the model equation and report Adjusted R².
- Briefly describe imputation methods.
- Show the neural network code and loss curve plot.
- Report MAPE scores (overall and by group).

---

Good luck with your presentation! 🎉