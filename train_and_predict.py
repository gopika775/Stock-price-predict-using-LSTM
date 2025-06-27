import numpy as np

def train_and_predict(model, X_train, y_train, X_test, scaler, y_test, epochs=50):
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return actual.flatten(), predicted.flatten()
