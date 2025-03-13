const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const mongoose = require('mongoose');
const stockRoutes = require('./routes/stocks');
const predictionRoutes = require('./routes/predictions');

//Initialize express app
const app = express();
app.use()

//Connect app to MongoDB
mongoose.connect('mongodb://localhost/stockdb', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

//Connect routes to app
app.use('/stocks', stockRoutes);
app.use('/predictions', predictionRoutes);

//Set port for app 

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});