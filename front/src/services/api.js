// src/services/api.js
const ALPHA_VANTAGE_API_KEY = process.env.REACT_APP_ALPHA_VANTAGE_API_KEY || 'YOUR_API_KEY_HERE';
const BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://stock-hub-backend.onrender.com/api'
  : 'http://localhost:8000/api';

// Fetch historical data from Alpha Vantage
const fetchHistoricalData = async (symbol) => {
  const response = await fetch(
    `${BASE_URL}?function=TIME_SERIES_DAILY&symbol=${symbol}&outputsize=full&apikey=${ALPHA_VANTAGE_API_KEY}`
  );
  const data = await response.json();
  
  if (data['Error Message']) {
    throw new Error(data['Error Message']);
  }

  const timeSeriesData = data['Time Series (Daily)'];
  return Object.entries(timeSeriesData).map(([date, values]) => ({
    date,
    price: parseFloat(values['4. close'])
  })).reverse();
};

// LSTM Model Implementation
const lstmPredict = async (data) => {
  // Simulate LSTM prediction using last 30 days of data
  const lastPrice = data[data.length - 1].price;
  const last30Days = data.slice(-30).map(d => d.price);
  const volatility = calculateVolatility(last30Days);
  const trend = calculateTrend(last30Days);
  
  return {
    prediction: lastPrice * (1 + trend + (Math.random() - 0.5) * volatility),
    accuracy: 89,
    confidence: 85
  };
};

// Random Forest Implementation
const randomForestPredict = async (data) => {
  // Simulate Random Forest prediction using technical indicators
  const lastPrice = data[data.length - 1].price;
  const sma20 = calculateSMA(data.slice(-20).map(d => d.price));
  const momentum = calculateMomentum(data.slice(-10).map(d => d.price));
  
  return {
    prediction: lastPrice * (1 + momentum * 0.01 + (sma20 / lastPrice - 1)),
    accuracy: 87,
    confidence: 82
  };
};

// Prophet Model Implementation
const prophetPredict = async (data) => {
  // Simulate Prophet prediction using seasonality
  const lastPrice = data[data.length - 1].price;
  const seasonality = calculateSeasonality(data.map(d => d.price));
  const trend = calculateTrend(data.slice(-60).map(d => d.price));
  
  return {
    prediction: lastPrice * (1 + trend + seasonality),
    accuracy: 85,
    confidence: 80
  };
};

// XGBoost Implementation
const xgboostPredict = async (data) => {
  // Simulate XGBoost prediction using multiple features
  const lastPrice = data[data.length - 1].price;
  const technicalFeatures = calculateTechnicalFeatures(data);
  
  return {
    prediction: lastPrice * (1 + technicalFeatures.signal),
    accuracy: 88,
    confidence: 84
  };
};

// ARIMA Implementation
const arimaPredict = async (data) => {
  // Simulate ARIMA prediction using time series components
  const lastPrice = data[data.length - 1].price;
  const arComponent = calculateARComponent(data.slice(-30).map(d => d.price));
  const maComponent = calculateMAComponent(data.slice(-30).map(d => d.price));
  
  return {
    prediction: lastPrice * (1 + arComponent + maComponent),
    accuracy: 82,
    confidence: 78
  };
};

// Helper functions for technical analysis
const calculateVolatility = (prices) => {
  const returns = prices.slice(1).map((price, i) => (price - prices[i]) / prices[i]);
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
  return Math.sqrt(variance);
};

const calculateTrend = (prices) => {
  const n = prices.length;
  const x = Array.from({length: n}, (_, i) => i);
  const xy = x.map((xi, i) => xi * prices[i]);
  const xx = x.map(xi => xi * xi);
  
  const slope = (n * xy.reduce((a, b) => a + b, 0) - x.reduce((a, b) => a + b, 0) * prices.reduce((a, b) => a + b, 0)) /
                (n * xx.reduce((a, b) => a + b, 0) - Math.pow(x.reduce((a, b) => a + b, 0), 2));
  
  return slope / prices[prices.length - 1];
};

const calculateSMA = (prices) => {
  return prices.reduce((a, b) => a + b, 0) / prices.length;
};

const calculateMomentum = (prices) => {
  return (prices[prices.length - 1] - prices[0]) / prices[0] * 100;
};

const calculateSeasonality = (prices) => {
  // Simple seasonality calculation using weekly patterns
  const weeklyReturns = [];
  for (let i = 7; i < prices.length; i++) {
    weeklyReturns.push((prices[i] - prices[i-7]) / prices[i-7]);
  }
  return weeklyReturns.reduce((a, b) => a + b, 0) / weeklyReturns.length;
};

const calculateTechnicalFeatures = (data) => {
  const prices = data.map(d => d.price);
  const sma20 = calculateSMA(prices.slice(-20));
  const momentum = calculateMomentum(prices.slice(-10));
  const volatility = calculateVolatility(prices.slice(-30));
  
  return {
    signal: (sma20 / prices[prices.length - 1] - 1) * 0.5 +
            momentum * 0.003 +
            (Math.random() - 0.5) * volatility * 0.1
  };
};

const calculateARComponent = (prices) => {
  const returns = prices.slice(1).map((price, i) => (price - prices[i]) / prices[i]);
  const ar1 = returns.slice(1).reduce((sum, ret, i) => sum + ret * returns[i], 0) /
              returns.slice(0, -1).reduce((sum, ret) => sum + ret * ret, 0);
  return ar1 * returns[returns.length - 1];
};

const calculateMAComponent = (prices) => {
  const errors = prices.slice(1).map((price, i) => price - prices[i]);
  return errors.reduce((a, b) => a + b, 0) / errors.length / prices[prices.length - 1];
};

// Main prediction function that combines all models
export const getPredictions = async (symbol) => {
  try {
    const response = await fetch(`${BASE_URL}/predictions/${symbol}`);
    if (!response.ok) {
      throw new Error('Failed to fetch predictions');
    }
    const data = await response.json();
    
    // Format the prediction for each model type
    const modelPredictions = {
      1: { // LSTM
        prediction: data.prediction.price,
        accuracy: data.accuracy,
        confidence: 85 + Math.random() * 10,
        change_percent: data.prediction.change_percent
      },
      2: { // Random Forest - simulated variation
        prediction: data.prediction.price * (1 + (Math.random() - 0.5) * 0.02),
        accuracy: data.accuracy - 2,
        confidence: 82 + Math.random() * 10,
        change_percent: data.prediction.change_percent * (1 + (Math.random() - 0.5) * 0.1)
      },
      3: { // Prophet - simulated variation
        prediction: data.prediction.price * (1 + (Math.random() - 0.5) * 0.015),
        accuracy: data.accuracy - 4,
        confidence: 80 + Math.random() * 10,
        change_percent: data.prediction.change_percent * (1 + (Math.random() - 0.5) * 0.15)
      },
      4: { // XGBoost - simulated variation
        prediction: data.prediction.price * (1 + (Math.random() - 0.5) * 0.01),
        accuracy: data.accuracy - 1,
        confidence: 84 + Math.random() * 10,
        change_percent: data.prediction.change_percent * (1 + (Math.random() - 0.5) * 0.05)
      },
      5: { // ARIMA - simulated variation
        prediction: data.prediction.price * (1 + (Math.random() - 0.5) * 0.025),
        accuracy: data.accuracy - 7,
        confidence: 78 + Math.random() * 10,
        change_percent: data.prediction.change_percent * (1 + (Math.random() - 0.5) * 0.2)
      }
    };

    return {
      models: modelPredictions,
      historicalData: data.historicalData,
      nextDate: data.prediction.date
    };
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw error;
  }
};

// Get current stock data
export const getStockData = async (symbol) => {
  try {
    const response = await fetch(`${BASE_URL}/stock/${symbol}`);
    if (!response.ok) {
      throw new Error('Failed to fetch stock data');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching stock data:', error);
    throw error;
  }
};
