import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import base64

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction Using LSTM")
st.markdown("---")


# ----------------------------
# User Inputs
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
with col2:
    epochs = st.slider("Training Epochs", 10, 200, 50, 10)

# ----------------------------
# Fetch Data
# ----------------------------
raw_data = yf.download(ticker, start="2018-01-01", progress=False)

if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

if raw_data.empty:
    st.error("⚠️ Could not fetch data. Please check ticker.")
    st.stop()

# ----------------------------
# 📋 Latest 3 Days of Raw Stock Data
# ----------------------------
st.subheader("📋 Latest 3 Days of Raw Stock Data")
st.dataframe(raw_data.tail(3), use_container_width=True)

# ----------------------------
# Normalized Chart
# ----------------------------
st.subheader("🔻 Normalized Closing Price (0–1 Scale)")
df_close = raw_data[['Close']]
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df_close)
norm_df = pd.DataFrame(scaled_close, index=df_close.index, columns=['Normalized Close'])

fig_norm = px.line(norm_df, x=norm_df.index, y='Normalized Close',
                   title="Normalized Close Price", template="plotly_dark")
fig_norm.update_layout(height=400)
st.plotly_chart(fig_norm, use_container_width=True)

# ----------------------------
# Prepare Data
# ----------------------------
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

# ----------------------------
# Build and Train LSTM Model
# ----------------------------


st.subheader("⚙️ Training Progress")
callback = StreamlitCallback(epochs)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[callback])
st.success("✅ Model Trained")

# ----------------------------
# Predictions and Metrics
# ----------------------------
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test)

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

# ----------------------------
# Training Loss Curve
# ----------------------------
st.subheader("📉 Training Loss Curve")
fig_loss = px.line(y=history.history['loss'], labels={'x': 'Epoch', 'y': 'Loss'},
                   title="Training Loss per Epoch", template="plotly_dark")
fig_loss.update_layout(height=400)
st.plotly_chart(fig_loss, use_container_width=True)

# ----------------------------
# Actual vs Predicted Plot (with Trendline)
# ----------------------------
st.subheader("📈 Actual vs Predicted Closing Prices")
result_df = pd.DataFrame({
    'Actual Price': actual_prices.flatten(),
    'Predicted Price': predicted_prices.flatten()
})
fig_ap = px.scatter(result_df, y="Actual Price", x="Predicted Price",
                    trendline="ols", title="Actual vs Predicted with Trendline",
                    template="plotly_dark")
fig_ap.update_traces(marker=dict(size=4, color='orange'))
fig_ap.update_layout(height=400)
st.plotly_chart(fig_ap, use_container_width=True)

# ----------------------------
# Residual Plot
# ----------------------------
st.subheader("📉 Residual Plot")
residuals = actual_prices.flatten() - predicted_prices.flatten()
fig_residual = go.Figure()
fig_residual.add_trace(go.Scatter(x=predicted_prices.flatten(), y=residuals,
                                  mode='markers', marker=dict(color='lightgreen')))
fig_residual.update_layout(title="Residuals vs Predicted Price",
                           xaxis_title="Predicted Price", yaxis_title="Residual",
                           template="plotly_dark", height=400)
st.plotly_chart(fig_residual, use_container_width=True)

# ----------------------------
# Bell Curve / Histogram of Residuals
# ----------------------------
st.subheader("🔔 Distribution of Residuals")
fig_bell = go.Figure()
fig_bell.add_trace(go.Histogram(
    x=residuals, nbinsx=50, name="Residuals",
    marker_color="lightblue", opacity=0.75, histnorm="probability density"
))
fig_bell.update_layout(title="Bell Curve of Residuals",
                       xaxis_title="Residual Value", yaxis_title="Density",
                       template="plotly_dark", height=400)
st.plotly_chart(fig_bell, use_container_width=True)

# ----------------------------
# Candlestick Chart
# ----------------------------
st.subheader("🕯️ Candlestick Chart – Last 100 Days")

try:
    required_cols = ['Open', 'High', 'Low', 'Close']
    candle_data = raw_data.copy()

    if all(col in candle_data.columns for col in required_cols):
        candle_data = candle_data[required_cols].dropna()
        candle_data = candle_data[(candle_data != 0).all(axis=1)]
        candle_data = candle_data.tail(100).copy()
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

        fig_candle.update_layout(
            title=f"{ticker} – Candlestick Chart (Last 100 Days)",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=400,
            template="plotly_dark"
        )

        st.plotly_chart(fig_candle, use_container_width=True)
    else:
        missing = [col for col in required_cols if col not in candle_data.columns]
        st.error(f"❌ Cannot plot candlestick chart. Missing columns: {', '.join(missing)}")

except Exception as e:
    st.error(f"⚠️ Error generating candlestick chart: {e}")

# ----------------------------
# Prediction Result Table
# ----------------------------
st.subheader("📋 Prediction Results Table")
st.dataframe(result_df.tail(10), use_container_width=True)

# ----------------------------
# Metrics Summary Table
# ----------------------------
st.subheader("🧾 Evaluation Metrics Table")
metrics_table = pd.DataFrame({
    'Metric': list(metrics_dict.keys()),
    'Value': list(metrics_dict.values())
})
st.table(metrics_table)

# ----------------------------
# Download CSV
# ----------------------------
st.subheader("📥 Download Prediction Results")
csv = result_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.markdown(f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📄 Download CSV</a>', unsafe_allow_html=True)
