// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import StockPage from './components/StockPage';
import './App.css';
import 'react-toastify/dist/ReactToastify.css';
import { ToastContainer } from 'react-toastify';

function App() {
  return (
    <Router basename="/stock_hub">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/stock/:symbol" element={<StockPage />} />
      </Routes>
      <ToastContainer />
    </Router>
  );
}

export default App;
