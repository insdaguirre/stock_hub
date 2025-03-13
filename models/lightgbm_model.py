import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LightGBMPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def create_features(self, data: List[Dict]) -> pd.DataFrame:
        """Create advanced technical indicators as features"""
        # Convert to DataFrame
        df = pd.DataFrame([{'date': x['date'], 'price': float(x['price'])} for x in data])
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic price data
        df['return'] = df['price'].pct_change()
        df['log_return'] = np.log1p(df['return'])
        
        # Moving averages and standard deviations
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False).mean()
            df[f'std_{window}'] = df['price'].rolling(window=window).std()
            df[f'roc_{window}'] = df['price'].pct_change(periods=window)
        
        # Price momentum
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['price'].diff(window)
            df[f'acceleration_{window}'] = df[f'momentum_{window}'].diff()
        
        # Volatility
        df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(252)
        
        # Relative strength index (RSI)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in [20]:
            mid = df['price'].rolling(window=window).mean()
            std = df['price'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = mid + 2 * std
            df[f'bb_lower_{window}'] = mid - 2 * std
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / mid
        
        # Mean reversion
        for window in [5, 10, 20]:
            df[f'mean_reversion_{window}'] = (df['price'] - df['price'].rolling(window=window).mean()) / df['price'].rolling(window=window).std()
        
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

    def build_model(self):
        """Initialize the LightGBM model with optimized parameters"""
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 6,  # Use 6 CPU cores
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 20,
            'max_depth': -1  # No limit
        }
        self.model = lgb.LGBMRegressor(**params)
        return self.model

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the LightGBM model"""
        # Create features
        df = self.create_features(historical_data)
        
        # Prepare sequences
        X, y = self.prepare_sequences(df)
        
        # Reshape for LightGBM (2D array)
        X = X.reshape(X.shape[0], -1)
        
        # Build and train model
        if not self.model:
            self.build_model()
        
        # Create validation set
        val_size = int(len(X) * 0.2)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mse',
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Save the model and scaler
        self.save_model(symbol)
        
        # Get feature importance
        feature_importance = dict(zip(
            [f'feature_{i}' for i in range(X.shape[1])],
            self.model.feature_importances_
        ))
        
        return {'feature_importance': feature_importance}

    def predict(self, historical_data: List[Dict], symbol: str) -> float:
        """Predict the next day's price"""
        # Load the model and scaler if they exist
        self.load_model(symbol)
        
        if not self.model:
            raise ValueError(f"No trained model found for symbol {symbol}")

        # Create features
        df = self.create_features(historical_data)
        
        # Get the last sequence
        feature_columns = [col for col in df.columns if col not in ['date', 'price']]
        last_sequence = df[feature_columns].iloc[-self.sequence_length:].values
        
        # Scale the sequence
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # Reshape for prediction
        X = scaled_sequence.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using R² score"""
        df = self.create_features(historical_data)
        X, y = self.prepare_sequences(df)
        X = X.reshape(X.shape[0], -1)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate R² score
        r2_score = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Convert R² to accuracy percentage (clip between 0 and 100)
        accuracy = max(0, min(100, 100 * r2_score))
        return float(accuracy)

    def save_model(self, symbol: str):
        """Save the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
        
        # Save the model
        self.model.booster_.save_model(os.path.join(symbol_path, 'lightgbm_model.txt'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'lightgbm_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'lightgbm_model.txt')
        scaler_path = os.path.join(symbol_path, 'lightgbm_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            if not self.model:
                self.build_model()
            self.model.booster_ = lgb.Booster(model_file=model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False 