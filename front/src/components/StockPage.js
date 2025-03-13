// src/components/StockPage.js

//imports
import React, { useEffect, useState } from 'react'; //React is the core library for building UI's
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'; //A react hook for performing side effects in functional components
import styled from 'styled-components';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'; //A library for building charts in react
import { getStockData, getPredictions } from '../services/api'; //Functions ipported from an API service module to fetch stock data and predictions

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
`;

const StockInfo = styled.div``;

const StockSymbol = styled.h1`
  font-size: 32px;
  margin: 0;
  color: #1c1c1e;
`;

const StockPrice = styled.div`
  font-size: 24px;
  margin-top: 8px;
  color: ${props => props.isPositive ? '#34C759' : '#FF3B30'};
`;

const BackButton = styled.button`
  padding: 8px 16px;
  border: none;
  border-radius: 8px;
  background-color: #f2f2f7;
  font-size: 16px;
  cursor: pointer;
  
  &:hover {
    background-color: #e5e5ea;
  }
`;

const ChartContainer = styled.div`
  background: white;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  height: 400px;
`;

const ModelInfo = styled.div`
  background: white;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

const ModelTitle = styled.h2`
  margin: 0 0 16px 0;
  color: #1c1c1e;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
`;

const Metric = styled.div`
  padding: 16px;
  background: #f2f2f7;
  border-radius: 8px;
`;

const MetricLabel = styled.div`
  font-size: 14px;
  color: #636366;
  margin-bottom: 4px;
`;

const MetricValue = styled.div`
  font-size: 24px;
  font-weight: 600;
  color: ${props => props.isPositive ? '#34C759' : props.isNegative ? '#FF3B30' : '#1c1c1e'};
`;

const models = {
  1: {
    name: 'LSTM Neural Network',
    description: 'Deep learning model specialized in sequence prediction',
    metrics: ['Accuracy', 'RMSE', 'MAE']
  },
  2: {
    name: 'Random Forest',
    description: 'Ensemble learning method for classification and regression',
    metrics: ['Accuracy', 'R²', 'Feature Importance']
  },
  3: {
    name: 'Prophet Model',
    description: 'Forecasting model that handles seasonality and holidays',
    metrics: ['Accuracy', 'Trend', 'Seasonality']
  },
  4: {
    name: 'XGBoost',
    description: 'Gradient boosting model optimized for speed and performance',
    metrics: ['Accuracy', 'RMSE', 'Feature Impact']
  }
};

const StockPage = () => { //Defines StockPage as a functional react component
  const { symbol } = useParams(); //Defines a variable called symbol that gets the value of the URL parameter
  const [searchParams] = useSearchParams();
  const modelId = searchParams.get('model');
  const navigate = useNavigate();
  
  const [stockData, setStockData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => { //Defines a side effect that fetches stock data and predictions when the component mounts
    const fetchData = async () => { //Defines a function to fetch the data
      try {
        setLoading(true);
        const [stockResponse, predictionResponse] = await Promise.all([
          getStockData(symbol),
          getPredictions(symbol)
        ]);
        
        setStockData(stockResponse);
        setPredictions(predictionResponse);
      } catch (err) {
        setError('Failed to fetch data. Please try again later.');
        console.error('Error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData(); //calls the fetchData function
  }, [symbol]); //Runs the side effect when the component mounts and when the symbol changes

  if (loading) return <Container>Loading...</Container>;
  if (error) return <Container>{error}</Container>;
  if (!stockData || !predictions) return <Container>No data available</Container>;

  const currentModel = models[modelId];
  const priceChange = stockData.price - stockData.previousClose;
  const percentChange = (priceChange / stockData.previousClose) * 100;
  const isPositive = priceChange >= 0;

  return ( //defines what the components UI should look like
    <Container>
      <Header>
        <StockInfo>
          <StockSymbol>{symbol}</StockSymbol>
          <StockPrice isPositive={isPositive}>
            ${stockData.price.toFixed(2)} {' '}
            <span>
              {isPositive ? '+' : ''}{priceChange.toFixed(2)} ({percentChange.toFixed(2)}%)
            </span>
          </StockPrice>
        </StockInfo>
        <BackButton onClick={() => navigate('/')}>Back to Search</BackButton>
      </Header>

      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={[...stockData.historicalData, ...predictions.forecastData]}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#1c1c1e" 
              name="Historical Price"
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="prediction" 
              stroke="#34C759" 
              name="Prediction"
              strokeWidth={2}
              strokeDasharray="5 5"
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>

      <ModelInfo>
        <ModelTitle>{currentModel.name}</ModelTitle>
        <p>{currentModel.description}</p>
        <MetricsGrid>
          <Metric>
            <MetricLabel>Prediction (7d)</MetricLabel>
            <MetricValue isPositive={predictions.sevenDayPrediction > 0} isNegative={predictions.sevenDayPrediction < 0}>
              {predictions.sevenDayPrediction > 0 ? '+' : ''}{predictions.sevenDayPrediction.toFixed(2)}%
            </MetricValue>
          </Metric>
          <Metric>
            <MetricLabel>Model Accuracy</MetricLabel>
            <MetricValue>{predictions.accuracy}%</MetricValue>
          </Metric>
          <Metric>
            <MetricLabel>Confidence Score</MetricLabel>
            <MetricValue>{predictions.confidenceScore}%</MetricValue>
          </Metric>
        </MetricsGrid>
      </ModelInfo>
    </Container>
  );
};

export default StockPage; //Exports the StockPage component for use in other parts of the application
