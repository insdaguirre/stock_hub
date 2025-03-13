import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CatBoostPredictor:
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
        
        # Extract date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
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
        
        # RSI
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
            df[f'bb_position_{window}'] = (df['price'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # Mean reversion
        for window in [5, 10, 20]:
            df[f'mean_reversion_{window}'] = (df['price'] - df['price'].rolling(window=window).mean()) / df['price'].rolling(window=window).std()
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        return df

    def prepare_sequences(self, df: pd.DataFrame) -> tuple:
        """Prepare sequences for training"""
        feature_columns = [col for col in df.columns if col not in ['date', 'price']]
        categorical_features = ['day_of_week', 'month', 'quarter']
        
        # Scale numerical features
        numerical_features = [col for col in feature_columns if col not in categorical_features]
        numerical_data = df[numerical_features].values
        scaled_numerical = self.scaler.fit_transform(numerical_data)
        
        # Combine with categorical features
        categorical_data = df[categorical_features].values
        features = np.hstack([scaled_numerical, categorical_data])
        
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(df['price'].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y), categorical_features

    def build_model(self):
        """Initialize the CatBoost model with optimized parameters"""
        params = {
            'iterations': 1000,
            'learning_rate': 0.01,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'early_stopping_rounds': 50,
            'verbose': False,
            'thread_count': 6  # Use 6 CPU cores
        }
        self.model = CatBoostRegressor(**params)
        return self.model

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the CatBoost model"""
        # Create features
        df = self.create_features(historical_data)
        
        # Prepare sequences
        X, y, categorical_features = self.prepare_sequences(df)
        
        # Reshape for CatBoost (2D array)
        X = X.reshape(X.shape[0], -1)
        
        # Calculate categorical feature indices
        cat_feature_indices = []
        features_per_seq = X.shape[1] // self.sequence_length
        for i, feat in enumerate(categorical_features):
            for j in range(self.sequence_length):
                cat_feature_indices.append(features_per_seq * j + (len(categorical_features) - len(categorical_features) + i))
        
        # Build and train model
        if not self.model:
            self.build_model()
        
        # Create validation set
        val_size = int(len(X) * 0.2)
        train_data = Pool(
            X[:-val_size], 
            y[:-val_size],
            cat_features=cat_feature_indices
        )
        val_data = Pool(
            X[-val_size:], 
            y[-val_size:],
            cat_features=cat_feature_indices
        )
        
        # Train the model
        self.model.fit(
            train_data,
            eval_set=val_data,
            use_best_model=True
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
        
        # Prepare sequences
        X, _, categorical_features = self.prepare_sequences(df)
        
        # Get last sequence
        last_sequence = X[-1:]
        
        # Reshape for prediction
        X = last_sequence.reshape(1, -1)
        
        # Calculate categorical feature indices
        cat_feature_indices = []
        features_per_seq = X.shape[1] // self.sequence_length
        for i, feat in enumerate(categorical_features):
            for j in range(self.sequence_length):
                cat_feature_indices.append(features_per_seq * j + (len(categorical_features) - len(categorical_features) + i))
        
        # Create prediction pool
        pred_pool = Pool(X, cat_features=cat_feature_indices)
        
        # Make prediction
        prediction = self.model.predict(pred_pool)[0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using R² score"""
        df = self.create_features(historical_data)
        X, y, categorical_features = self.prepare_sequences(df)
        X = X.reshape(X.shape[0], -1)
        
        # Calculate categorical feature indices
        cat_feature_indices = []
        features_per_seq = X.shape[1] // self.sequence_length
        for i, feat in enumerate(categorical_features):
            for j in range(self.sequence_length):
                cat_feature_indices.append(features_per_seq * j + (len(categorical_features) - len(categorical_features) + i))
        
        # Create pool for prediction
        pred_pool = Pool(X, cat_features=cat_feature_indices)
        
        # Make predictions
        y_pred = self.model.predict(pred_pool)
        
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
        self.model.save_model(os.path.join(symbol_path, 'catboost_model.cbm'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'catboost_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'catboost_model.cbm')
        scaler_path = os.path.join(symbol_path, 'catboost_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            if not self.model:
                self.build_model()
            self.model.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False 