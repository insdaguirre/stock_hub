import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class LSTMPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(30, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        
        self.model = model
        return model

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def train(self, historical_data, symbol, epochs=50, validation_split=0.2):
        # Extract prices and scale them
        prices = np.array([float(x['price']) for x in historical_data]).reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)

        # Create sequences
        X, y = self.create_sequences(scaled_prices)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and train model
        if not self.model:
            self.build_model((self.sequence_length, 1))

        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1,
            batch_size=32
        )

        # Save the model and scaler
        self.save_model(symbol)
        
        return history.history

    def predict(self, historical_data, symbol):
        """Predict only the next day's price"""
        # Load the model and scaler if they exist
        self.load_model(symbol)
        
        if not self.model:
            raise ValueError(f"No trained model found for symbol {symbol}")

        # Prepare the data
        prices = np.array([float(x['price']) for x in historical_data]).reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        
        # Get the last sequence
        last_sequence = scaled_prices[-self.sequence_length:]
        current_sequence = last_sequence.reshape(1, self.sequence_length, 1)
        
        # Predict next value
        next_pred = self.model.predict(current_sequence, verbose=0)
        
        # Inverse transform prediction
        prediction = self.scaler.inverse_transform(next_pred.reshape(-1, 1))
        
        return float(prediction[0, 0])

    def calculate_accuracy(self, historical_data):
        """Calculate accuracy using only next-day predictions"""
        prices = np.array([float(x['price']) for x in historical_data]).reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        
        X, y_true = self.create_sequences(scaled_prices)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        y_pred = self.model.predict(X, verbose=0)
        
        # Calculate RMSE for next-day predictions
        mse = np.mean((y_true - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        
        # Convert RMSE to accuracy percentage (inverse relationship)
        accuracy = max(0, 100 - (rmse * 100))
        return float(accuracy)

    def save_model(self, symbol):
        # Create symbol-specific directory
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
            
        # Save the model
        self.model.save(os.path.join(symbol_path, 'lstm_model'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'scaler.save'))

    def load_model(self, symbol):
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'lstm_model')
        scaler_path = os.path.join(symbol_path, 'scaler.save')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False 