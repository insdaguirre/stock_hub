const express = require('express');
const axios = require('axios');

//Create router object to handle middleware
const router = express.Router();


//Define a route handler for GET

router.get('/:symbol', async (req, res) => {  //define asybchronous function to fetch stock data
    try {
        const { symbol } = req.params; //extract the symbol from the request parameters
        const apiKey = 'YOUR_API_KEY' //define API key
        const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${symbol}&apikey=${apiKey}`; //define URL for fetching stock data
        const response = await axios.get(url); //sends a GET request to the URL
        res.json(response.data); //Sends the API request back to the client
    } catch (error) {
        console.error('Error fetching stock data:', error); //throw error
        res.status(500).send('Error fetching stock data');

    }
});

    module.exports = router; //export the router object so that it can be used elsewhere in the application 