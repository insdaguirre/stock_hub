// src/components/HomePage.js
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { getPredictions } from '../services/api';

const Container = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  background-color: #000000;
  color: #FFFFFF;
  min-height: 100vh;
`;

const Header = styled.div`
  padding: 10px 0;
`;

const Title = styled.h1`
  font-size: 32px;
  font-weight: 700;
  margin: 0;
  color: #FFFFFF;
`;

const DateText = styled.h2`
  font-size: 20px;
  color: #666;
  margin: 5px 0 15px 0;
  font-weight: normal;
`;

const SearchContainer = styled.div`
  margin: 16px 0;
`;

const SearchInput = styled.input`
  width: 100%;
  padding: 12px 16px;
  border: none;
  border-radius: 10px;
  background-color: #1C1C1E;
  font-size: 16px;
  outline: none;
  color: #FFFFFF;
  margin-bottom: 10px;

  &::placeholder {
    color: #666;
  }
`;

const StockSelector = styled.div`
  background-color: #1C1C1E;
  padding: 15px;
  border-radius: 10px;
  margin-bottom: 20px;
`;

const StockSelectorTitle = styled.div`
  font-size: 16px;
  color: #666;
  margin-bottom: 10px;
`;

const ModelsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1px;
  background-color: #1C1C1E;
  border-radius: 10px;
  overflow: hidden;
`;

const ModelCard = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background-color: #000000;
  cursor: pointer;
  transition: background-color 0.2s ease;

  &:hover {
    background-color: #2C2C2E;
  }
`;

const ModelInfo = styled.div`
  flex: 1;
`;

const ModelName = styled.div`
  font-size: 16px;
  color: #FFFFFF;
  margin-bottom: 4px;
`;

const ModelDescription = styled.div`
  font-size: 14px;
  color: #666;
`;

const ModelMetrics = styled.div`
  text-align: right;
`;

const Prediction = styled.div`
  font-size: 16px;
  font-weight: 600;
  color: ${props => props.value >= 0 ? '#34C759' : '#FF3B30'};
  margin-bottom: 4px;
`;

const Accuracy = styled.div`
  font-size: 14px;
  color: #666;
`;

const MiniChart = styled.div`
  width: 60px;
  height: 30px;
  margin: 0 15px;
  opacity: 0.7;
`;

const LoadingOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
  font-size: 18px;
`;

const ErrorMessage = styled.div`
  color: #FF3B30;
  padding: 16px;
  background-color: #1C1C1E;
  border-radius: 10px;
  margin-bottom: 20px;
`;

// Extended list of 20 models with more variety
const models = [
  {
    id: 1,
    name: 'LSTM Neural Network',
    description: 'Deep Learning',
    accuracy: '89%',
    prediction: '+2.3%'
  },
  {
    id: 2,
    name: 'Random Forest',
    description: 'Ensemble Learning',
    accuracy: '87%',
    prediction: '-1.5%'
  },
  {
    id: 3,
    name: 'Prophet',
    description: 'Time Series',
    accuracy: '85%',
    prediction: '+1.8%'
  },
  {
    id: 4,
    name: 'XGBoost',
    description: 'Gradient Boosting',
    accuracy: '88%',
    prediction: '+0.9%'
  },
  {
    id: 5,
    name: 'ARIMA',
    description: 'Statistical Analysis',
    accuracy: '82%',
    prediction: '-0.7%'
  },
  {
    id: 6,
    name: 'Transformer',
    description: 'Deep Learning',
    accuracy: '90%',
    prediction: '+1.2%'
  },
  {
    id: 7,
    name: 'CNN-LSTM',
    description: 'Hybrid Model',
    accuracy: '86%',
    prediction: '+2.1%'
  },
  {
    id: 8,
    name: 'LightGBM',
    description: 'Gradient Boosting',
    accuracy: '87%',
    prediction: '-0.8%'
  },
  {
    id: 9,
    name: 'VAR',
    description: 'Vector Autoregression',
    accuracy: '81%',
    prediction: '+0.5%'
  },
  {
    id: 10,
    name: 'ESN',
    description: 'Echo State Network',
    accuracy: '84%',
    prediction: '-1.2%'
  },
  {
    id: 11,
    name: 'Wavelet Transform',
    description: 'Signal Processing',
    accuracy: '83%',
    prediction: '+1.6%'
  },
  {
    id: 12,
    name: 'Kalman Filter',
    description: 'State Estimation',
    accuracy: '82%',
    prediction: '+0.7%'
  },
  {
    id: 13,
    name: 'Decision Tree',
    description: 'Tree-based Model',
    accuracy: '80%',
    prediction: '-0.9%'
  },
  {
    id: 14,
    name: 'SVM',
    description: 'Support Vector Machine',
    accuracy: '81%',
    prediction: '+1.1%'
  },
  {
    id: 15,
    name: 'Neural Prophet',
    description: 'Neural Forecasting',
    accuracy: '86%',
    prediction: '+1.9%'
  },
  {
    id: 16,
    name: 'Ensemble Mix',
    description: 'Multi-Model Blend',
    accuracy: '91%',
    prediction: '+1.4%'
  },
  {
    id: 17,
    name: 'GRU',
    description: 'Recurrent Neural Net',
    accuracy: '85%',
    prediction: '-1.1%'
  },
  {
    id: 18,
    name: 'CatBoost',
    description: 'Gradient Boosting',
    accuracy: '88%',
    prediction: '+0.8%'
  },
  {
    id: 19,
    name: 'SARIMA',
    description: 'Seasonal ARIMA',
    accuracy: '83%',
    prediction: '-0.6%'
  },
  {
    id: 20,
    name: 'Temporal Fusion',
    description: 'Transformer-based',
    accuracy: '89%',
    prediction: '+1.7%'
  }
];

const HomePage = () => {
  const [stockSymbol, setStockSymbol] = useState('');
  const [activeStock, setActiveStock] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPredictions = async () => {
      if (!activeStock) return;
      
      setLoading(true);
      setError(null);
      try {
        const data = await getPredictions(activeStock);
        setPredictions(data);
      } catch (err) {
        setError('Failed to fetch predictions. Please try again.');
        console.error('Error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [activeStock]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (stockSymbol) {
      setActiveStock(stockSymbol.toUpperCase());
    }
  };

  const handleModelClick = (modelId) => {
    if (activeStock) {
      navigate(`/stock/${activeStock}?model=${modelId}`);
    }
  };

  const currentDate = new Date().toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric'
  });

  const getModelPrediction = (modelId) => {
    if (!predictions || !predictions.models[modelId]) return null;
    
    const model = predictions.models[modelId];
    const predictionValue = ((model.prediction / predictions.historicalData[predictions.historicalData.length - 1].price) - 1) * 100;
    return {
      prediction: `${predictionValue >= 0 ? '+' : ''}${predictionValue.toFixed(2)}%`,
      accuracy: `${model.accuracy}%`,
      value: predictionValue
    };
  };

  // Only show first 5 models that we've implemented
  const implementedModels = models.slice(0, 5);

  return (
    <Container>
      <Header>
        <Title>Stock Predictions</Title>
        <DateText>{currentDate}</DateText>
      </Header>

      <StockSelector>
        <StockSelectorTitle>Enter Stock Symbol</StockSelectorTitle>
        <form onSubmit={handleSearch}>
          <SearchInput
        type="text"
        value={stockSymbol}
        onChange={(e) => setStockSymbol(e.target.value)}
            placeholder="Enter stock symbol (e.g., AAPL)"
          />
        </form>
      </StockSelector>

      {error && <ErrorMessage>{error}</ErrorMessage>}

      {activeStock && (
        <ModelsList style={{ position: 'relative' }}>
          {loading && <LoadingOverlay>Loading predictions...</LoadingOverlay>}
          {implementedModels.map(model => {
            const prediction = getModelPrediction(model.id);
            return (
              <ModelCard key={model.id} onClick={() => handleModelClick(model.id)}>
                <ModelInfo>
                  <ModelName>{model.name}</ModelName>
                  <ModelDescription>{model.description}</ModelDescription>
                </ModelInfo>
                <MiniChart>
                  {/* Mini chart placeholder */}
                </MiniChart>
                <ModelMetrics>
                  <Prediction value={prediction ? parseFloat(prediction.prediction) : 0}>
                    {prediction ? prediction.prediction : 'Loading...'}
                  </Prediction>
                  <Accuracy>{prediction ? prediction.accuracy : '...'}</Accuracy>
                </ModelMetrics>
              </ModelCard>
            );
          })}
        </ModelsList>
      )}
    </Container>
  );
};

export default HomePage;
