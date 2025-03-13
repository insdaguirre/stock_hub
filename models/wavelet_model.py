import numpy as np
import tensorflow as tf
import pywt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WaveletNNPredictor:
    def __init__(self, sequence_length=20, wavelet='db1', level=3):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        self.wavelet = wavelet
        self.level = level
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def wavelet_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply wavelet transform to decompose the signal"""
        # Apply wavelet decomposition
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # Concatenate coefficients
        features = np.concatenate([coeffs[0]] + [c for c in coeffs[1:]], axis=0)
        return features

    def create_features(self, data: List[Dict]) -> pd.DataFrame:
        """Create features with wavelet transform"""
        # Convert to DataFrame
        df = pd.DataFrame([{'date': x['date'], 'price': float(x['price'])} for x in data])
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate returns and technical indicators
        df['return'] = df['price'].pct_change()
        df['volatility'] = df['return'].rolling(window=20).std()
        
        # Apply wavelet transform to price and returns
        price_wavelet = self.wavelet_transform(df['price'].values)
        returns_wavelet = self.wavelet_transform(df['return'].fillna(0).values)
        
        # Create wavelet feature columns
        for i in range(len(price_wavelet)):
            df[f'price_wavelet_{i}'] = price_wavelet[i]
            df[f'returns_wavelet_{i}'] = returns_wavelet[i]
        
        # Add technical indicators
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'std_{window}'] = df['price'].rolling(window=window).std()
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        return df

    def prepare_sequences(self, df: pd.DataFrame) -> tuple:
        """Prepare sequences for training"""
        feature_columns = [col for col in df.columns if col not in ['date', 'price']]
        
        # Scale features
        features = df[feature_columns].values
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(df['price'].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape: tuple):
        """Build Wavelet Neural Network model"""
        model = tf.keras.Sequential([
            # CNN layers for wavelet feature extraction
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            
            # LSTM layers for temporal dependencies
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers for prediction
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
        """Train the Wavelet Neural Network model"""
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
        prediction = self.model.predict(last_sequence, verbose=0)[0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using MAPE"""
        df = self.create_features(historical_data)
        X, y = self.prepare_sequences(df)
        
        # Make predictions
        y_pred = self.model.predict(X, verbose=0)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y - y_pred.flatten()) / y)) * 100
        
        # Convert MAPE to accuracy
        accuracy = max(0, 100 - mape)
        return float(accuracy)

    def save_model(self, symbol: str):
        """Save the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
        
        # Save the model
        self.model.save(os.path.join(symbol_path, 'wavelet_model'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'wavelet_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'wavelet_model')
        scaler_path = os.path.join(symbol_path, 'wavelet_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False 