import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

class ProphetPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'saved_models'
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def prepare_data(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Prepare data for Prophet format (ds and y columns)"""
        df = pd.DataFrame(historical_data)
        df.columns = ['ds', 'y']  # Prophet requires these column names
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = self.scaler.fit_transform(df[['y']])
        return df

    def build_model(self):
        """Initialize the Prophet model with optimized parameters"""
        self.model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10,    # Flexibility of seasonality
            holidays_prior_scale=10,       # Flexibility of holiday effects
            daily_seasonality=False,       # Stock market doesn't have strong daily patterns
            weekly_seasonality=True,       # Weekly patterns are important for stocks
            yearly_seasonality=True,       # Yearly patterns can be significant
            seasonality_mode='multiplicative'  # Better for stock prices
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Add stock market holidays
        self.model.add_country_holidays(country_name='US')
        
        return self.model

    def train(self, historical_data: List[Dict], symbol: str):
        """Train the Prophet model"""
        # Prepare data
        df = self.prepare_data(historical_data)
        
        # Build and train model
        if not self.model:
            self.build_model()
        
        # Train the model
        self.model.fit(df)
        
        # Save the model and scaler
        self.save_model(symbol)
        
        # Return model components for analysis
        return {
            'seasonalities': self.model.seasonalities,
            'changepoints': self.model.changepoints.tolist()
        }

    def predict(self, historical_data: List[Dict], symbol: str) -> float:
        """Predict the next day's price"""
        # Load the model and scaler if they exist
        self.load_model(symbol)
        
        if not self.model:
            raise ValueError(f"No trained model found for symbol {symbol}")

        # Prepare prediction DataFrame
        last_date = pd.to_datetime(historical_data[-1]['date'])
        future_dates = pd.DataFrame({
            'ds': [last_date + timedelta(days=1)]
        })
        
        # Skip weekends
        while future_dates['ds'].dt.dayofweek[0] >= 5:
            future_dates['ds'] = future_dates['ds'] + timedelta(days=1)
        
        # Make prediction
        forecast = self.model.predict(future_dates)
        
        # Inverse transform the prediction
        prediction = self.scaler.inverse_transform(forecast[['yhat']])[0, 0]
        
        return float(prediction)

    def calculate_accuracy(self, historical_data: List[Dict]) -> float:
        """Calculate model accuracy using MAPE (Mean Absolute Percentage Error)"""
        df = self.prepare_data(historical_data)
        
        # Make in-sample predictions
        predictions = self.model.predict(df[['ds']])
        
        # Inverse transform predictions and actual values
        y_pred = self.scaler.inverse_transform(predictions[['yhat']])
        y_true = self.scaler.inverse_transform(df[['y']])
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Convert MAPE to accuracy (100 - MAPE)
        accuracy = max(0, 100 - mape)
        return float(accuracy)

    def save_model(self, symbol: str):
        """Save the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)
            
        # Save the model
        with open(os.path.join(symbol_path, 'prophet_model.json'), 'w') as f:
            self.model.serialize_model(f)
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(symbol_path, 'prophet_scaler.joblib'))

    def load_model(self, symbol: str) -> bool:
        """Load the model and scaler"""
        symbol_path = os.path.join(self.model_path, symbol)
        model_path = os.path.join(symbol_path, 'prophet_model.json')
        scaler_path = os.path.join(symbol_path, 'prophet_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load Prophet model
            with open(model_path, 'r') as f:
                self.model = Prophet.deserialize_model(f.read())
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            return True
        return False 