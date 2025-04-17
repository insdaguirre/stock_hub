# Stock Hub Backend

A stock prediction API with multiple prediction models including LSTM, Random Forest, Prophet, and more.

## Features

- Stock price predictions using multiple models
- Historical stock data retrieval
- API endpoints for frontend integration

## Deployment

This application is designed to be deployed on Render.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python server.py
```

## Environment Variables

- `ALPHA_VANTAGE__KEY`: Your Alpha Vantage API key
- `PORT`: The port to run the server on (default: 8000)

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Models Used](#models-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

---

## Overview

Modern quantitative finance often involves testing multiple forecasting methods on historical data to see which performs best. This website streamlines that process by:
- Providing a user-friendly interface
- Displaying predictions from various quant models side-by-side
- Enabling visual and statistical comparisons

By making these comparisons more accessible, it's easier to identify model strengths, weaknesses, and applicability.

---

## Features

- **Model Predictions**: Visualize daily, weekly, or monthly forecasts from each model.
- **Interactive Plots**: Zoom in and out of the data range for better inspection.
- **Statistical Metrics**: Quickly see error metrics (e.g., RMSE, MAE) for each model.
- **Easy Comparison**: Toggle models on or off to simplify your view.

---

## Models Used

Below are the models currently integrated into the site, alongside links to their respective overviews or official documentation:

1. **ARIMA (Autoregressive Integrated Moving Average)**
   - [ARIMA (Wikipedia)](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

2. **CatBoost**
   - [CatBoost Official Site](https://catboost.ai/)

3. **GRU (Gated Recurrent Units)**
   - [GRU (Wikipedia)](https://en.wikipedia.org/wiki/Gated_recurrent_unit)

4. **LightGBM**
   - [LightGBM Documentation](https://lightgbm.readthedocs.io/)

5. **LSTM (Long Short-Term Memory)**
   - [LSTM (Wikipedia)](https://en.wikipedia.org/wiki/Long_short-term_memory)

6. **Prophet**
   - [Prophet Documentation](https://facebook.github.io/prophet/)

7. **Random Forest**
   - [Random Forest (Wikipedia)](https://en.wikipedia.org/wiki/Random_forest)

8. **VAR (Vector Autoregression)**
   - [VAR (Wikipedia)](https://en.wikipedia.org/wiki/Vector_autoregression)

9. **Wavelet-Based Model**
   - [Wavelet Transform (Wikipedia)](https://en.wikipedia.org/wiki/Wavelet_transform)

10. **XGBoost**
    - [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

