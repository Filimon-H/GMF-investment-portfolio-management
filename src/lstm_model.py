import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


from sklearn.preprocessing import MinMaxScaler

def create_lstm_sequences(data, window_size):
    """
    Converts 1D array into LSTM-ready sequences (X, y).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Build a simple LSTM model using Keras Sequential API.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
