import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import base64
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback

# Page Config
st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction Using LSTM")
st.markdown("---")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
with col2:
    epochs = st.slider("Training Epochs", 10, 200, 50, 10)

# Fetch Data
raw_data = yf.download(ticker, start="2018-01-01", progress=False)

if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

if raw_data.empty:
    st.error("⚠️ Could not fetch data. Please check ticker.")
    st.stop()

# Latest Raw Data
st.subheader("📋 Latest 3 Days of Raw Stock Data")
st.dataframe(raw_data.tail(3), use_container_width=True)

# Normalize Close Price
st.subheader("🔻 Normalized Closing Price (0–1 Scale)")
df_close = raw_data[['Close']]
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df_close)
norm_df = pd.DataFrame(scaled_close, index=df_close.index, columns=['Normalized Close'])
fig_norm = px.line(norm_df, x=norm_df.index, y='Normalized Close', title="Normalized Close Price", template="plotly_dark")
fig_norm.update_layout(height=400)
st.plotly_chart(fig_norm, use_container_width=True)

# Prepare Sequences
train_len = int(len(scaled_close) * 0.8)
train_data = scaled_close[:train_len]
test_data = scaled_close[train_len - 60:]

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Callback for Training
class StreamlitCallback(Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        progress = int((epoch + 1) / self.epochs * 100)
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} completed.")

# Train Model (Only if not in session state)
if "trained_model" not in st.session_state:
    st.subheader("⚙️ Model Training Progress")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    callback = StreamlitCallback(epochs)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[callback])

    st.session_state.trained_model = model
    st.session_state.history = history
    st.success("✅ Model Trained Successfully!")
else:
    model = st.session_state.trained_model
    history = st.session_state.history
    st.success("✅ Loaded trained model from memory.")

# Make Predictions
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test)

# Evaluation Metrics
st.subheader("📊 Evaluation Metrics")
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

metrics_dict = {
    "Mean Squared Error (MSE)": round(mse, 4),
    "Root Mean Squared Error (RMSE)": round(rmse, 4),
    "Mean Absolute Error (MAE)": round(mae, 4),
    "R² Score": round(r2, 4)
}
st.write(metrics_dict)

# Training Loss Curve
st.subheader("📉 Training Loss Curve")
fig_loss = px.line(y=history.history['loss'], labels={'x': 'Epoch', 'y': 'Loss'}, title="Training Loss per Epoch", template="plotly_dark")
fig_loss.update_layout(height=400)
st.plotly_chart(fig_loss, use_container_width=True)

# Actual vs Predicted (Line Plot)
st.subheader("📊 Actual vs Predicted (Line Plot)")
result_df = pd.DataFrame({'Actual Price': actual_prices.flatten(), 'Predicted Price': predicted_prices.flatten()})
fig_line = px.line(result_df, title="Actual vs Predicted Prices", template="plotly_dark")
st.plotly_chart(fig_line, use_container_width=True)

# Actual vs Predicted (Trendline)
st.subheader("📈 Actual vs Predicted Scatter with Trendline")
fig_ap = px.scatter(result_df, y="Actual Price", x="Predicted Price", trendline="ols", template="plotly_dark")
fig_ap.update_traces(marker=dict(size=4, color='orange'))
fig_ap.update_layout(height=400)
st.plotly_chart(fig_ap, use_container_width=True)

# Residual Plot
st.subheader("📉 Residual Plot")
residuals = actual_prices.flatten() - predicted_prices.flatten()
fig_residual = go.Figure()
fig_residual.add_trace(go.Scatter(x=predicted_prices.flatten(), y=residuals, mode='markers', marker=dict(color='lightgreen')))
fig_residual.update_layout(title="Residuals vs Predicted Price", xaxis_title="Predicted Price", yaxis_title="Residual", template="plotly_dark", height=400)
st.plotly_chart(fig_residual, use_container_width=True)

# Residual Histogram
st.subheader("🔔 Distribution of Residuals")
fig_bell = go.Figure()
fig_bell.add_trace(go.Histogram(x=residuals, nbinsx=50, marker_color="lightblue", opacity=0.75, histnorm="probability density"))
fig_bell.update_layout(title="Bell Curve of Residuals", xaxis_title="Residual", yaxis_title="Density", template="plotly_dark", height=400)
st.plotly_chart(fig_bell, use_container_width=True)

# Candlestick Chart
st.subheader("🕯️ Candlestick Chart – Last 100 Days")
try:
    candle_data = raw_data[['Open', 'High', 'Low', 'Close']].dropna().tail(100)
    candle_data['Date'] = candle_data.index
    fig_candle = go.Figure(data=[go.Candlestick(
        x=candle_data['Date'],
        open=candle_data['Open'],
        high=candle_data['High'],
        low=candle_data['Low'],
        close=candle_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig_candle.update_layout(title=f"{ticker} – Candlestick Chart", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template="plotly_dark", height=400)
    st.plotly_chart(fig_candle, use_container_width=True)
except Exception as e:
    st.error(f"⚠️ Candlestick chart error: {e}")

# Prediction Table
st.subheader("📋 Prediction Results Table")
st.dataframe(result_df.tail(10), use_container_width=True)

# Metrics Table
st.subheader("🧾 Evaluation Metrics Table")
metrics_table = pd.DataFrame({'Metric': list(metrics_dict.keys()), 'Value': list(metrics_dict.values())})
st.table(metrics_table)

# Download CSV
st.subheader("📥 Download Prediction Results")
csv = result_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.markdown(f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📄 Download CSV</a>', unsafe_allow_html=True)

# Save Trained Model
st.subheader("💾 Save Trained Model")
if st.button("📁 Save Model as .h5"):
    if "trained_model" in st.session_state:
        with st.spinner("Saving trained model..."):
            save_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                save_bar.progress(i + 1)
            st.session_state.trained_model.save("trained_lstm_model.h5")
        st.success("✅ Model saved as `trained_lstm_model.h5`")
    else:
        st.error("⚠️ Please train the model first.")
