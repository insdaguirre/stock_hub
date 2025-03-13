import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')

class XGBoostPredictor:
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
        
        # Convert price to numpy array for talib
        prices = df['price'].values
        
        # Trend Indicators
        df['sma_5'] = talib.SMA(prices, timeperiod=5)
        df['sma_20'] = talib.SMA(prices, timeperiod=20)
        df['ema_5'] = talib.EMA(prices, timeperiod=5)
        df['ema_20'] = talib.EMA(prices, timeperiod=20)
        
        # Momentum Indicators
        df['rsi'] = talib.RSI(prices, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Volatility Indicators
        df['atr'] = talib.ATR(prices, prices, prices, timeperiod=14)
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(
            prices, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Volume-based indicators (simulated since we only have price)
        df['obv'] = talib.OBV(prices, np.ones_like(prices))
        
        # Custom Features
        df['price_position'] = (df['price'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        df['sma_ratio'] = df['sma_5'] / df['sma_20']
        df['ema_ratio'] = df['ema_5'] / df['ema_20']
        
        # Momentum and Mean Reversion
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['price'].pct_change(window)
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
        """Initialize the XGBoost model with optimized parameters"""
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=6  # Use 6 CPU cores
        )
        return self.model

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the XGBoost model"""
        # Create features
        df = self.create_features(historical_data)
        
        # Prepare sequences
        X, y = self.prepare_sequences(df)
        
        # Reshape for XGBoost (2D array)
        X = X.reshape(X.shape[0], -1)
        
        # Build and train model
        if not self.model:
            self.build_model()
        
        # Train the model with early stopping
        eval_set = [(X[-100:], y[-100:])]  # Use last 100 samples as validation
        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Save the model and scaler
        self.save_model(symbol)
        
        # Return feature importance
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
        self.model.save_model(os.path.join(symbol_path, 'xgboost_model.json'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'xgboost_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'xgboost_model.json')
        scaler_path = os.path.join(symbol_path, 'xgboost_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            if not self.model:
                self.build_model()
            self.model.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False 