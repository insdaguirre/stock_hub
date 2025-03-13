import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GRUPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def create_features(self, data: List[Dict]) -> pd.DataFrame:
        """Create features for GRU model"""
        df = pd.DataFrame([{'date': x['date'], 'price': float(x['price'])} for x in data])
        df['date'] = pd.to_datetime(df['date'])
        
        # Technical indicators
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'std_{window}'] = df['price'].rolling(window=window).std()
            df[f'momentum_{window}'] = df['price'].pct_change(periods=window)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Price differences
        df['price_diff'] = df['price'].diff()
        df['price_diff2'] = df['price_diff'].diff()
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        return df

    def prepare_sequences(self, df: pd.DataFrame) -> tuple:
        """Prepare sequences for GRU model"""
        # Select features for training
        feature_columns = [col for col in df.columns if col not in ['date']]
        features = df[feature_columns].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(scaled_features[i + self.sequence_length, 0])  # Price is first column
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape: tuple):
        """Build GRU model with advanced architecture"""
        model = tf.keras.Sequential([
            # First GRU layer with return sequences
            tf.keras.layers.GRU(128, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Second GRU layer
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Third GRU layer
            tf.keras.layers.GRU(32),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile model with Adam optimizer and learning rate scheduler
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(optimizer=optimizer, loss='mse')
        self.model = model
        return model

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the GRU model"""
        # Prepare data
        df = self.create_features(historical_data)
        X, y = self.prepare_sequences(df)
        
        # Build model if not exists
        if not self.model:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Train model with early stopping and model checkpoint
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Train with validation split
        history = self.model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Save the model and scaler
        self.save_model(symbol)
        
        return {
            'training_loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1])
        }

    def predict(self, historical_data: List[Dict], symbol: str) -> float:
        """Predict the next day's price"""
        # Load the model and scaler if they exist
        self.load_model(symbol)
        
        if not self.model:
            raise ValueError(f"No trained model found for symbol {symbol}")

        # Prepare data
        df = self.create_features(historical_data)
        X, _ = self.prepare_sequences(df)
        
        # Get last sequence
        last_sequence = X[-1:]
        
        # Make prediction
        scaled_prediction = self.model.predict(last_sequence, verbose=0)[0]
        
        # Inverse transform prediction
        dummy = np.zeros((1, df.shape[1]))
        dummy[0, 0] = scaled_prediction
        prediction = self.scaler.inverse_transform(dummy)[0, 0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using MAPE"""
        df = self.create_features(historical_data)
        X, y = self.prepare_sequences(df)
        
        # Make predictions
        scaled_predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform predictions and actual values
        dummy_pred = np.zeros((len(scaled_predictions), df.shape[1]))
        dummy_pred[:, 0] = scaled_predictions.flatten()
        predictions = self.scaler.inverse_transform(dummy_pred)[:, 0]
        
        dummy_actual = np.zeros((len(y), df.shape[1]))
        dummy_actual[:, 0] = y
        actual = self.scaler.inverse_transform(dummy_actual)[:, 0]
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        # Convert MAPE to accuracy
        accuracy = max(0, 100 - mape)
        return float(accuracy)

    def save_model(self, symbol: str):
        """Save the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
        
        # Save the model
        self.model.save(os.path.join(symbol_path, 'gru_model'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'gru_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'gru_model')
        scaler_path = os.path.join(symbol_path, 'gru_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False 