from data_loader import load_and_preprocess_data
from model import build_lstm_model
from train_and_predict import train_and_predict
from plotter import plot_predictions
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
X_train, y_train, X_test, y_test, scaler, actual_prices = load_and_preprocess_data()

# Build LSTM model
lstm_model = build_lstm_model(X_train.shape[1:])

# Train and predict
actual_lstm, pred_lstm = train_and_predict(lstm_model, X_train, y_train, X_test, scaler, y_test)

# Evaluation metrics
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n🔎 Evaluation for {name}")
    print(f"📌 MAE  = {mae:.4f}")
    print(f"📌 RMSE = {rmse:.4f}")
    print(f"📌 R²    = {r2:.4f}")

# Print metrics
rmse = np.sqrt(mean_squared_error(actual_lstm, pred_lstm))
print(f"✅ LSTM RMSE: {rmse:.4f}")
evaluate_model("LSTM", actual_lstm, pred_lstm)

# Plot results
plot_predictions(actual_lstm, pred_lstm)
