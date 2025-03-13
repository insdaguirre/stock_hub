import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import warnings
from multiprocessing import Pool
warnings.filterwarnings('ignore')

class VARPredictor:
    def __init__(self, max_lags=20):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        self.max_lags = max_lags
        self.selected_lags = None
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def create_features(self, data: List[Dict]) -> pd.DataFrame:
        """Create multivariate features for VAR model"""
        # Convert to DataFrame
        df = pd.DataFrame([{'date': x['date'], 'price': float(x['price'])} for x in data])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Create features
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'std_{window}'] = df['price'].rolling(window=window).std()
            df[f'momentum_{window}'] = df['price'].pct_change(periods=window)
        
        # Price differences
        df['price_diff'] = df['price'].diff()
        df['price_diff2'] = df['price_diff'].diff()
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        # Scale all features
        scaled_data = self.scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        
        return df_scaled

    def find_best_lag(self, data: pd.DataFrame) -> int:
        """Find the optimal lag order using parallel processing"""
        def calculate_aic(lag):
            try:
                model = VAR(data)
                results = model.fit(lag)
                return lag, results.aic
            except:
                return lag, np.inf

        # Use parallel processing to calculate AIC for different lags
        with Pool(6) as pool:  # Use 6 CPU cores
            results = pool.map(calculate_aic, range(1, self.max_lags + 1))
        
        # Find the lag with minimum AIC
        best_lag = min(results, key=lambda x: x[1])[0]
        return best_lag

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the VAR model"""
        # Prepare multivariate data
        df = self.create_features(historical_data)
        
        # Find optimal lag order
        self.selected_lags = self.find_best_lag(df)
        
        # Train VAR model
        self.model = VAR(df)
        self.model = self.model.fit(self.selected_lags)
        
        # Save the model and parameters
        self.save_model(symbol)
        
        return {
            'selected_lags': self.selected_lags,
            'aic': self.model.aic,
            'bic': self.model.bic
        }

    def predict(self, historical_data: List[Dict], symbol: str) -> float:
        """Predict the next day's price"""
        # Load the model and scaler if they exist
        self.load_model(symbol)
        
        if not self.model:
            raise ValueError(f"No trained model found for symbol {symbol}")

        # Prepare data
        df = self.create_features(historical_data)
        
        # Get last observations for prediction
        last_data = df.iloc[-self.selected_lags:]
        
        # Make prediction
        forecast = self.model.forecast(last_data.values, steps=1)
        
        # Get price prediction (first column)
        scaled_prediction = forecast[0, 0]
        
        # Inverse transform the prediction
        prediction = self.scaler.inverse_transform(
            np.array([[scaled_prediction] + [0] * (df.shape[1] - 1)])
        )[0, 0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using MAPE"""
        df = self.create_features(historical_data)
        
        # Make in-sample predictions
        predictions = self.model.fittedvalues
        
        # Get actual values (excluding the first lag observations)
        actual_scaled = df.iloc[self.selected_lags:]['price'].values
        pred_scaled = predictions[:, 0]  # First column is price
        
        # Inverse transform
        actual_values = self.scaler.inverse_transform(
            np.column_stack([actual_scaled, np.zeros((len(actual_scaled), df.shape[1]-1))])
        )[:, 0]
        predicted_values = self.scaler.inverse_transform(
            np.column_stack([pred_scaled, np.zeros((len(pred_scaled), df.shape[1]-1))])
        )[:, 0]
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        
        # Convert MAPE to accuracy
        accuracy = max(0, 100 - mape)
        return float(accuracy)

    def save_model(self, symbol: str):
        """Save the model, scaler, and parameters"""
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
        
        # Save model parameters
        model_params = {
            'selected_lags': self.selected_lags,
            'params': self.model.params,
            'sigma_u': self.model.sigma_u,
            'names': self.model.names
        }
        
        # Save parameters and scaler
        joblib.dump(model_params, os.path.join(symbol_path, 'var_params.joblib'))
        joblib.dump(self.scaler, os.path.join(symbol_path, 'var_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model, scaler, and parameters"""
        symbol_path = os.path.join(self.model_path, symbol)
        params_path = os.path.join(symbol_path, 'var_params.joblib')
        scaler_path = os.path.join(symbol_path, 'var_scaler.joblib')
        
        if os.path.exists(params_path) and os.path.exists(scaler_path):
            # Load parameters and scaler
            model_params = joblib.load(params_path)
            self.scaler = joblib.load(scaler_path)
            self.selected_lags = model_params['selected_lags']
            
            # Prepare data for model reconstruction
            df = self.create_features(historical_data)
            
            # Rebuild model
            self.model = VAR(df)
            self.model = self.model.fit(self.selected_lags)
            
            return True
        return False 