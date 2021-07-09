# Backend Code for Forecasting Application

## Description
This application has the backend code for the Angular Application named 
**Web Portal for AI-based Tools for
Fresh Produce Procurement Price Forecasting**. This application has two main functionalities:
- Data Preprocessing
- Forecasting

    **Data Preprocessing** has two sub-modules:
    - *Imputation:* This module helps in imputing the time series with missing values at random points or chunks of missing values.
    - *Similarity Check:* This module helps in finding similarity between two time series with detailed analysis.
    
    **Forecasting** has two sub-modules:
    - *Price Forecasting:* This module forecasts the price values using station-based, satellite-based and combined deep learning models.
    - *Yield Forecasting:* This module forecasts the yield values using station-based, satellite-based and combined deep learning models.

  


## Installation
Install any IDE like PyCharm or Visual Studio Code to run the python fiels for serving the 
flask application backend code.

## Usage
Run the *FlaskApp_Main.py* file in the Application folder to serve the Flask Application.

##### Directories
- Application: This has the main python file *FlaskApp_Main.py* for running the code.
- Data: This folder has the different datasets used for training and similarity check in the backend.
- Imputation: This folder has codes required for Imputation.
- SimilarityCodes: Here the codes for similarity check and analysis is present.
- StationCodes: Codes for Station Based forecasting, web scraping, data preprocessing, transfer learning are present.
- SatelliteCodes: Codes for the Satellite Based forecasting and transfer learning are present.
- StationTraining: This folder has codes for retraining the station based models.
- TrainingWeights, SatelliteModels, pcaWeights, sclrWeights: These folder have weights for the different models used in application.
- *client_secrets.json*, *mycreds.txt*: These files have credentials required for google drive authentication to access the satellite images.



