import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import warnings
import pmdarima as pm
from multiprocessing import Pool
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        self.order = None  # Store the ARIMA order
        self.diff_order = None  # Store the differencing order
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def check_stationarity(self, data: np.ndarray) -> int:
        """
        Check stationarity using Augmented Dickey-Fuller test
        Returns the order of differencing needed
        """
        max_diff = 2  # Maximum differencing order
        for d in range(max_diff + 1):
            if d == 0:
                test_data = data
            else:
                test_data = np.diff(data, n=d)
            
            adf_result = adfuller(test_data, regression='ct')
            if adf_result[1] < 0.05:  # p-value < 0.05 indicates stationarity
                return d
        return max_diff

    def prepare_data(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Prepare data for ARIMA format"""
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['price'] = self.scaler.fit_transform(df[['price']])
        return df

    def find_best_parameters(self, data: np.ndarray) -> tuple:
        """
        Find the best ARIMA parameters using parallel processing
        Returns the optimal (p, d, q) order
        """
        # First, determine the differencing order
        self.diff_order = self.check_stationarity(data)
        
        # Use auto_arima with parallel processing for parameter selection
        model = pm.auto_arima(
            data,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=self.diff_order,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=10,
            n_jobs=6  # Use 6 CPU cores
        )
        
        return model.order

    def build_model(self, data: np.ndarray, order: tuple):
        """Initialize the ARIMA model with optimal parameters"""
        self.model = ARIMA(data, order=order)
        return self.model

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the ARIMA model"""
        # Prepare data
        df = self.prepare_data(historical_data)
        data = df['price'].values
        
        # Find best parameters
        self.order = self.find_best_parameters(data)
        
        # Build and train model
        self.model = self.build_model(data, self.order)
        self.model = self.model.fit()
        
        # Save the model and parameters
        self.save_model(symbol)
        
        return {
            'order': self.order,
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
        df = self.prepare_data(historical_data)
        
        # Make prediction
        forecast = self.model.forecast(steps=1)
        
        # Inverse transform the prediction
        prediction = self.scaler.inverse_transform(forecast.reshape(-1, 1))[0, 0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using MAPE (Mean Absolute Percentage Error)"""
        df = self.prepare_data(historical_data)
        data = df['price'].values
        
        # Make in-sample predictions
        predictions = self.model.predict(start=self.order[0])
        
        # Get actual values (excluding the first p observations where p is the AR order)
        actual = data[self.order[0]:]
        
        # Inverse transform predictions and actual values
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        actual = self.scaler.inverse_transform(actual.reshape(-1, 1))
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        # Convert MAPE to accuracy (100 - MAPE)
        accuracy = max(0, 100 - mape)
        return float(accuracy)

    def save_model(self, symbol: str):
        """Save the model, scaler, and parameters"""
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
        
        # Save model parameters and state
        model_params = {
            'order': self.order,
            'diff_order': self.diff_order,
            'params': self.model.params,
            'sigma2': self.model.sigma2
        }
        
        # Save parameters
        joblib.dump(model_params, os.path.join(symbol_path, 'arima_params.joblib'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'arima_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model, scaler, and parameters"""
        symbol_path = os.path.join(self.model_path, symbol)
        params_path = os.path.join(symbol_path, 'arima_params.joblib')
        scaler_path = os.path.join(symbol_path, 'arima_scaler.joblib')
        
        if os.path.exists(params_path) and os.path.exists(scaler_path):
            # Load parameters
            model_params = joblib.load(params_path)
            self.order = model_params['order']
            self.diff_order = model_params['diff_order']
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Prepare data for model reconstruction
            df = self.prepare_data(historical_data)
            data = df['price'].values
            
            # Rebuild model with loaded parameters
            self.model = ARIMA(data, order=self.order)
            self.model = self.model.fit(start_params=model_params['params'])
            
            return True
        return False 