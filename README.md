## LSTM Forecasting with Seasonality Features: Air Passengers Dataset

This repository implements a time series forecasting pipeline using a stacked LSTM neural network with cyclical month encoding to capture seasonality. The goal is to predict monthly international airline passengers (1949–1960) with high accuracy (MAPE < 10%).

---

🚀 Features

- Log-transform of passenger data to stabilize variance
- Cyclical encoding of month features (sin & cos) for seasonality
- Stacked LSTM layers with Dropout for regularization
- EarlyStopping callback to prevent overfitting
- Evaluation metrics: RMSE & MAPE
- Visualization: Actual vs. Predicted passenger counts

---

📁 Repository Structure

```
AirPassengers.csv      # Raw dataset (1949-1960 monthly passenger counts)
lstm_airpassengers.py  # Main Python script with model pipeline
README.txt             # Project overview and instructions
```

---

🛠️ Prerequisites

- Python 3.7+
- pip

Install dependencies:
```
pip install pandas numpy matplotlib scikit-learn tensorflow
```

---

⚙️ Usage

1. Place `AirPassengers.csv` in the project root directory.
2. Run the script:
   ```
   python lstm_airpassengers.py
   ```
3. Review training logs for RMSE & MAPE metrics.
4. Inspect the plotted graph comparing actual vs. predicted values.

---

📈 Expected Output

- Test RMSE: ~37.97
- Test MAPE: < 10%
- Plot: Forecast vs. Ground truth

---

🔧 Tuning & Extensions

- Adjust `look_back`, LSTM units, batch size, epochs
- Experiment with different architectures (Bidirectional LSTM, Transformers)
- Hyperparameter search (GridSearch, Random Search)
- Incorporate external variables (e.g., holidays, economic indicators)



