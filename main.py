from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta
import numpy as np
from models.lstm_model import LSTMPredictor
from models.random_forest_model import RandomForestPredictor
from models.prophet_model import ProphetPredictor
from models.xgboost_model import XGBoostPredictor
from models.arima_model import ARIMAPredictor
from models.var_model import VARPredictor
from models.gru_model import GRUPredictor
from models.lightgbm_model import LightGBMPredictor
from models.catboost_model import CatBoostPredictor
from models.wavelet_model import WaveletNNPredictor
import asyncio
from concurrent.futures import ProcessPoolExecutor

load_dotenv()

app = FastAPI()

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "API is running", "endpoints": ["/api/stock/{symbol}", "/api/predictions/{symbol}"]}

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://insdaguirre.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model cache
model_cache = {}
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

async def fetch_stock_data(symbol: str):
    """Fetch stock data from Alpha Vantage"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise HTTPException(status_code=404, detail=f"Stock symbol {symbol} not found")
        
        time_series = data.get('Time Series (Daily)', {})
        historical_data = [
            {
                'date': date,
                'price': float(values['4. close'])
            }
            for date, values in time_series.items()
        ]
        
        return sorted(historical_data, key=lambda x: x['date'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def train_models_task(historical_data, symbol):
    """Training task to be run in a separate process"""
    # Train LSTM
    lstm_predictor = LSTMPredictor()
    lstm_predictor.train(historical_data, symbol)
    
    # Train Random Forest
    rf_predictor = RandomForestPredictor()
    rf_predictor.train(historical_data, symbol)
    
    # Train Prophet
    prophet_predictor = ProphetPredictor()
    prophet_predictor.train(historical_data, symbol)
    
    # Train XGBoost
    xgb_predictor = XGBoostPredictor()
    xgb_predictor.train(historical_data, symbol)
    
    # Train ARIMA
    arima_predictor = ARIMAPredictor()
    arima_predictor.train(historical_data, symbol)
    
    # Train VAR
    var_predictor = VARPredictor()
    var_predictor.train(historical_data, symbol)
    
    # Train GRU
    gru_predictor = GRUPredictor()
    gru_predictor.train(historical_data, symbol)
    
    # Train LightGBM
    lgb_predictor = LightGBMPredictor()
    lgb_predictor.train(historical_data, symbol)
    
    # Train CatBoost
    cat_predictor = CatBoostPredictor()
    cat_predictor.train(historical_data, symbol)
    
    # Train Wavelet
    wav_predictor = WaveletNNPredictor()
    wav_predictor.train(historical_data, symbol)
    
    return symbol

@app.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get current stock data"""
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise HTTPException(status_code=404, detail=f"Stock symbol {symbol} not found")
        
        quote = data.get('Global Quote', {})
        return {
            'price': float(quote.get('05. price', 0)),
            'previousClose': float(quote.get('08. previous close', 0))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str):
    """Get next-day predictions for a stock symbol"""
    try:
        # Fetch historical data
        historical_data = await fetch_stock_data(symbol)
        
        # Initialize predictors
        lstm_predictor = LSTMPredictor()
        rf_predictor = RandomForestPredictor()
        prophet_predictor = ProphetPredictor()
        xgb_predictor = XGBoostPredictor()
        arima_predictor = ARIMAPredictor()
        var_predictor = VARPredictor()
        gru_predictor = GRUPredictor()
        lgb_predictor = LightGBMPredictor()
        cat_predictor = CatBoostPredictor()
        wav_predictor = WaveletNNPredictor()
        
        # Check if models exist and load them
        lstm_exists = lstm_predictor.load_model(symbol)
        rf_exists = rf_predictor.load_model(symbol)
        prophet_exists = prophet_predictor.load_model(symbol)
        xgb_exists = xgb_predictor.load_model(symbol)
        arima_exists = arima_predictor.load_model(symbol)
        var_exists = var_predictor.load_model(symbol)
        gru_exists = gru_predictor.load_model(symbol)
        lgb_exists = lgb_predictor.load_model(symbol)
        cat_exists = cat_predictor.load_model(symbol)
        wav_exists = wav_predictor.load_model(symbol)
        
        if not all([lstm_exists, rf_exists, prophet_exists, xgb_exists, arima_exists,
                   var_exists, gru_exists, lgb_exists, cat_exists, wav_exists]):
            # Train models if they don't exist
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=6) as executor:
                await loop.run_in_executor(executor, train_models_task, historical_data, symbol)
        
        # Make predictions for next day
        lstm_price = lstm_predictor.predict(historical_data, symbol)
        rf_price = rf_predictor.predict(historical_data, symbol)
        prophet_price = prophet_predictor.predict(historical_data, symbol)
        xgb_price = xgb_predictor.predict(historical_data, symbol)
        arima_price = arima_predictor.predict(historical_data, symbol)
        var_price = var_predictor.predict(historical_data, symbol)
        gru_price = gru_predictor.predict(historical_data, symbol)
        lgb_price = lgb_predictor.predict(historical_data, symbol)
        cat_price = cat_predictor.predict(historical_data, symbol)
        wav_price = wav_predictor.predict(historical_data, symbol)
        
        # Calculate accuracies
        lstm_accuracy = lstm_predictor.calculate_accuracy(historical_data)
        rf_accuracy = rf_predictor.calculate_accuracy(historical_data)
        prophet_accuracy = prophet_predictor.calculate_accuracy(historical_data)
        xgb_accuracy = xgb_predictor.calculate_accuracy(historical_data)
        arima_accuracy = arima_predictor.calculate_accuracy(historical_data)
        var_accuracy = var_predictor.calculate_accuracy(historical_data)
        gru_accuracy = gru_predictor.calculate_accuracy(historical_data)
        lgb_accuracy = lgb_predictor.calculate_accuracy(historical_data)
        cat_accuracy = cat_predictor.calculate_accuracy(historical_data)
        wav_accuracy = wav_predictor.calculate_accuracy(historical_data)
        
        # Generate next business day date
        last_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # Skip weekends
            next_date += timedelta(days=1)

        # Calculate prediction percentage changes
        current_price = historical_data[-1]['price']
        lstm_change = ((lstm_price - current_price) / current_price) * 100
        rf_change = ((rf_price - current_price) / current_price) * 100
        prophet_change = ((prophet_price - current_price) / current_price) * 100
        xgb_change = ((xgb_price - current_price) / current_price) * 100
        arima_change = ((arima_price - current_price) / current_price) * 100
        var_change = ((var_price - current_price) / current_price) * 100
        gru_change = ((gru_price - current_price) / current_price) * 100
        lgb_change = ((lgb_price - current_price) / current_price) * 100
        cat_change = ((cat_price - current_price) / current_price) * 100
        wav_change = ((wav_price - current_price) / current_price) * 100

        return {
            'symbol': symbol,
            'predictions': {
                'lstm': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': lstm_price,
                    'change_percent': lstm_change,
                    'accuracy': lstm_accuracy
                },
                'random_forest': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': rf_price,
                    'change_percent': rf_change,
                    'accuracy': rf_accuracy
                },
                'prophet': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': prophet_price,
                    'change_percent': prophet_change,
                    'accuracy': prophet_accuracy
                },
                'xgboost': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': xgb_price,
                    'change_percent': xgb_change,
                    'accuracy': xgb_accuracy
                },
                'arima': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': arima_price,
                    'change_percent': arima_change,
                    'accuracy': arima_accuracy
                },
                'var': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': var_price,
                    'change_percent': var_change,
                    'accuracy': var_accuracy
                },
                'gru': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': gru_price,
                    'change_percent': gru_change,
                    'accuracy': gru_accuracy
                },
                'lightgbm': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': lgb_price,
                    'change_percent': lgb_change,
                    'accuracy': lgb_accuracy
                },
                'catboost': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': cat_price,
                    'change_percent': cat_change,
                    'accuracy': cat_accuracy
                },
                'wavelet': {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': wav_price,
                    'change_percent': wav_change,
                    'accuracy': wav_accuracy
                }
            },
            'historicalData': historical_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=6) 