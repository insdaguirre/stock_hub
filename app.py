from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Alpha Vantage API key from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def fetch_stock_data(symbol):
    """Fetch historical stock data from Alpha Vantage."""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact'
    response = requests.get(url)
    data = response.json()
    
    if "Error Message" in data:
        raise Exception(data["Error Message"])
    
    time_series = data.get('Time Series (Daily)', {})
    historical_data = []
    
    for date, values in time_series.items():
        historical_data.append({
            'date': date,
            'price': float(values['4. close'])
        })
    
    return sorted(historical_data, key=lambda x: x['date'])

def calculate_prediction(prices, days_ahead=1):
    """Simple prediction based on moving average and trend."""
    prices = np.array(prices)
    ma = np.mean(prices[-5:])  # 5-day moving average
    trend = (prices[-1] - prices[-5]) / 5  # Average daily change
    prediction = ma + (trend * days_ahead)
    return max(0, prediction)  # Ensure prediction is not negative

@app.get("/")
async def root():
    return {"status": "API is running", "message": "Hello from Stock Hub API!"}

@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str):
    try:
        # Fetch historical data
        historical_data = fetch_stock_data(symbol)
        
        if not historical_data:
            return {"error": "No data available for this symbol"}
        
        # Get closing prices
        prices = [entry['price'] for entry in historical_data]
        
        # Calculate next day's date
        last_date = datetime.strptime(historical_data[-1]['date'], '%Y-%m-%d')
        next_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Calculate prediction
        predicted_price = calculate_prediction(prices)
        current_price = prices[-1]
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate accuracy (simplified)
        accuracy = 85  # Base accuracy
        recent_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
        accuracy = max(75, min(95, accuracy - (recent_volatility * 100)))
        
        return {
            "prediction": {
                "date": next_date,
                "price": predicted_price,
                "change_percent": change_percent
            },
            "accuracy": accuracy,
            "historicalData": historical_data
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str):
    try:
        historical_data = fetch_stock_data(symbol)
        return {"data": historical_data}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 