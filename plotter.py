import matplotlib.pyplot as plt

def plot_predictions(actual, lstm_pred):
    actual = actual.reshape(-1)
    lstm_pred = lstm_pred.reshape(-1)

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Price', color='blue')
    plt.plot(lstm_pred, label='LSTM Prediction', color='orange')
    plt.title('Stock Price Prediction using LSTM')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
