// src/components/HomePage.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { getPredictions } from '../services/api';
import ProgressBar from './ProgressBar';

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

// Add new styled components for the progress section
const LoadingContainer = styled.div`
  background-color: #1C1C1E;
  padding: 20px;
  border-radius: 10px;
  margin-bottom: 20px;
`;

const LoadingTitle = styled.h3`
  font-size: 18px;
  color: #FFFFFF;
  margin: 0 0 15px 0;
`;

const ProgressContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 15px;
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
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL'); // Default to Apple
  const [predictionsData, setPredictionsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // New state for tracking loading progress
  const [loadingProgress, setLoadingProgress] = useState({});
  const [overallProgress, setOverallProgress] = useState(0);

  // Function to simulate loading progress for each model
  const simulateProgress = (modelIds) => {
    // Initialize progress for each model
    const initialProgress = {};
    modelIds.forEach(id => {
      initialProgress[id] = 0;
    });
    setLoadingProgress(initialProgress);
    
    // Simulate different completion times for different models
    const modelTimes = {
      1: 3,  // LSTM takes longest
      2: 2,  // Random Forest
      3: 2.5,// Prophet
      4: 1.8,// XGBoost
      5: 1.5 // ARIMA is fastest
    };
    
    // Update progress every 100ms
    const interval = setInterval(() => {
      setLoadingProgress(prev => {
        const updated = { ...prev };
        let allComplete = true;
        let totalProgress = 0;
        
        modelIds.forEach(id => {
          if (updated[id] < 100) {
            // Increase progress based on model complexity
            const increment = 100 / (modelTimes[id] * 10); // 10 updates per second
            updated[id] = Math.min(updated[id] + increment, 100);
            
            if (updated[id] < 100) {
              allComplete = false;
            }
          }
          totalProgress += updated[id];
        });
        
        // Calculate overall progress
        setOverallProgress(totalProgress / modelIds.length);
        
        // If all models are done, clear the interval
        if (allComplete) {
          clearInterval(interval);
        }
        
        return updated;
      });
    }, 100);
    
    // Store the interval ID to clear it if component unmounts
    return interval;
  };

  const fetchPredictions = async () => {
    if (!selectedSymbol) return;
    
    try {
      setLoading(true);
      setError(null);
      
      // Get the model IDs we need to load (1-5)
      const modelIds = [1, 2, 3, 4, 5];
      
      // Start the progress simulation
      const progressInterval = simulateProgress(modelIds);
      
      // Fetch actual predictions
      const data = await getPredictions(selectedSymbol);
      setPredictionsData(data);
      
      // Ensure we show 100% progress before stopping
      setLoadingProgress(prev => {
        const complete = {};
        modelIds.forEach(id => {
          complete[id] = 100;
        });
        return complete;
      });
      setOverallProgress(100);
      
      // Clear the interval after a short delay to ensure UI shows 100%
      setTimeout(() => {
        clearInterval(progressInterval);
        setLoading(false);
      }, 500);
      
    } catch (err) {
      setError('Failed to fetch predictions. Please try again.');
      console.error('Error:', err);
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
    // If it looks like a valid stock symbol, update selectedSymbol
    if (/^[A-Z]{1,5}$/.test(e.target.value.toUpperCase())) {
      setSelectedSymbol(e.target.value.toUpperCase());
    }
  };

  const handleModelClick = (modelId) => {
    if (selectedSymbol) {
      navigate(`/stock/${selectedSymbol}?model=${modelId}`);
    }
  };

  const getModelPrediction = (modelId) => {
    if (!predictionsData || !predictionsData.models[modelId]) return null;
    
    const model = predictionsData.models[modelId];
    const predictionValue = ((model.prediction / predictionsData.historicalData[predictionsData.historicalData.length - 1].price) - 1) * 100;
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
        <Title>Stock Prediction Hub</Title>
        <DateText>{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</DateText>
      </Header>
      
      <SearchInput
        type="text"
        placeholder="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
        value={searchTerm}
        onChange={handleSearch}
      />
      
      <StockSelector>
        <StockSelectorTitle>Selected Symbol: {selectedSymbol}</StockSelectorTitle>
        <SearchInput
          type="button"
          value="Get Predictions"
          onClick={fetchPredictions}
          style={{ 
            backgroundColor: '#0A84FF',
            color: 'white',
            fontWeight: 'bold',
            cursor: 'pointer'
          }}
        />
      </StockSelector>

      {loading && (
        <LoadingContainer>
          <LoadingTitle>Generating Predictions ({Math.round(overallProgress)}% Complete)</LoadingTitle>
          <ProgressContainer>
            <ProgressBar 
              progress={overallProgress}
              label="Overall Progress"
            />
            <ProgressBar 
              progress={loadingProgress[1] || 0}
              label="LSTM Neural Network"
              timeRemaining={Math.ceil((100 - (loadingProgress[1] || 0)) / 33)}
            />
            <ProgressBar 
              progress={loadingProgress[2] || 0}
              label="Random Forest"
              timeRemaining={Math.ceil((100 - (loadingProgress[2] || 0)) / 50)}
            />
            <ProgressBar 
              progress={loadingProgress[3] || 0}
              label="Prophet Model"
              timeRemaining={Math.ceil((100 - (loadingProgress[3] || 0)) / 40)}
            />
            <ProgressBar 
              progress={loadingProgress[4] || 0}
              label="XGBoost"
              timeRemaining={Math.ceil((100 - (loadingProgress[4] || 0)) / 55)}
            />
            <ProgressBar 
              progress={loadingProgress[5] || 0}
              label="ARIMA"
              timeRemaining={Math.ceil((100 - (loadingProgress[5] || 0)) / 66)}
            />
          </ProgressContainer>
        </LoadingContainer>
      )}

      {error && <ErrorMessage>{error}</ErrorMessage>}

      {selectedSymbol && (
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
