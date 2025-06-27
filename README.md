# Stock Price Prediction using LSTM
------------------------------------

This project presents a time series forecasting application that predicts stock closing prices using an LSTM (Long Short-Term Memory) neural network. It is built with Python and deployed as an interactive web app using Streamlit.

----------------------------------------------------------------------------------------------------------------------


## Key Features
----------------
- Real-time data retrieval using Yahoo Finance (yfinance)
- MinMax scaling for input normalization
- Sequence creation for LSTM training
- LSTM-based deep learning model for price prediction
- Evaluation metrics: MAE, RMSE, R² Score
- Residual distribution and model error analysis
- Candlestick chart visualization for recent trading days
- Exportable prediction results in CSV format
- Interactive visualizations and progress monitoring via Streamlit

----------------------------------------------------------------------------------------------------------------------


## Application Flow
--------------------

1. **Data Loading:** Pulls historical stock data using the user-inputted ticker.
2. **Preprocessing:** Normalizes prices and prepares training/testing sequences.
3. **Model Training:** LSTM model is trained on the past 60-day window of stock prices.
4. **Prediction:** Future stock prices are predicted and compared against actual values.
5. **Visualization:** Includes actual vs. predicted graphs, candlestick charts, residual plots, and tables.
6. **Evaluation:** Quantitative metrics (MAE, RMSE, R²) are calculated and displayed.

