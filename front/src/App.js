// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './components/HomePage';
import StockPage from './components/StockPage';
import 'react-toastify/dist/ReactToastify.css';
import { ToastContainer } from 'react-toastify';

const App = () => (
  <Router>
    <div>
      <Routes>
        <Route exact path="/" element={<HomePage />} />
        <Route path="/stock/:symbol" element={<StockPage />} />
      </Routes>
      <ToastContainer />
    </div>
  </Router>
);

export default App;
