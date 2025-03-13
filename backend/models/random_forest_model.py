import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict

class RandomForestPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def create_features(self, data: List[Dict]) -> np.ndarray:
        """Create technical indicators as features"""
        df = pd.DataFrame([{'price': float(x['price'])} for x in data])
        
        # Technical indicators
        df['SMA_5'] = df['price'].rolling(window=5).mean()
        df['SMA_20'] = df['price'].rolling(window=20).mean()
        df['EMA_5'] = df['price'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['price'].ewm(span=20, adjust=False).mean()
        
        # Price momentum
        df['momentum_1'] = df['price'].pct_change(periods=1)
        df['momentum_5'] = df['price'].pct_change(periods=5)
        
        # Volatility
        df['volatility'] = df['price'].rolling(window=20).std()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['price'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['price'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['price'].rolling(window=20).std()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

        # Create sequences
        df = df.fillna(0)  # Fill NaN values with 0
        return df.values

    def prepare_data(self, features: np.ndarray) -> tuple:
        """Prepare sequences for training"""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(features[i + self.sequence_length, 0])  # Target is the price
        return np.array(X), np.array(y)

    def build_model(self):
        """Initialize the Random Forest model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=6,  # Use 6 CPU cores
            random_state=42
        )
        return self.model

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the Random Forest model"""
        # Create features
        features = self.create_features(historical_data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Prepare sequences
        X, y = self.prepare_data(scaled_features)
        
        # Reshape for Random Forest (2D array)
        X = X.reshape(X.shape[0], -1)  # Flatten the sequences
        
        # Build and train model
        if not self.model:
            self.build_model()
        
        # Train the model
        self.model.fit(X, y)
        
        # Save the model and scaler
        self.save_model(symbol)
        
        return {
            'feature_importances': dict(zip(
                [f'feature_{i}' for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        }

    def predict(self, historical_data: List[Dict], symbol: str) -> float:
        """Predict the next day's price"""
        # Load the model and scaler if they exist
        self.load_model(symbol)
        
        if not self.model:
            raise ValueError(f"No trained model found for symbol {symbol}")

        # Create and scale features
        features = self.create_features(historical_data)
        scaled_features = self.scaler.transform(features)
        
        # Get the last sequence
        last_sequence = scaled_features[-self.sequence_length:]
        
        # Reshape for prediction
        X = last_sequence.reshape(1, -1)  # Flatten the sequence
        
        # Make prediction
        scaled_prediction = self.model.predict(X)[0]
        
        # Inverse transform the prediction
        # We need to create a dummy array with the same shape as the original features
        dummy = np.zeros((1, scaled_features.shape[1]))
        dummy[0, 0] = scaled_prediction
        prediction = self.scaler.inverse_transform(dummy)[0, 0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using R² score"""
        features = self.create_features(historical_data)
        scaled_features = self.scaler.transform(features)
        X, y = self.prepare_data(scaled_features)
        X = X.reshape(X.shape[0], -1)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate R² score
        r2_score = self.model.score(X, y)
        
        # Convert R² to accuracy percentage
        accuracy = max(0, 100 * r2_score)
        return float(accuracy)

    def save_model(self, symbol: str):
        """Save the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
            
        # Save the model
        joblib.dump(self.model, os.path.join(symbol_path, 'random_forest_model.joblib'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'random_forest_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'random_forest_model.joblib')
        scaler_path = os.path.join(symbol_path, 'random_forest_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False 