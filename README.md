# MarketForecast
A WIP attempt at stock market prediction using machine learning
Stock Price Prediction Model
This project is a machine learning-based stock price prediction model that uses historical stock price data for prediction. The model uses various technical indicators and XGBoost, a gradient boosting algorithm, to make predictions. This README provides an overview of the project's structure and functionality.

dashboard: Will eventually contain a web dashboard for viewing graphs.
data: Data directory where historical stock price data is stored.
database: Directory for database-related files such as moving the historical data to a database.
models: Directory to store numerous trained machine learning models mostly all very minor parameter tweaks from optimization.
notebooks: Jupyter notebooks related to the project.
prototype: Main directory containing the project's source code (will be moved to training eventually).
find params.py: Script for finding optimal hyperparameters (this may very well be a terrible way of accomplishing this but it works so far).
training: Directory for training scripts.
util: Utility scripts and functions.
visualizations: Scripts for various visualization techniques.

#### Usage
To run, execute the main.py script located in the prototype directory. The script loads historical stock price data, preprocesses it, trains the XGBoost model, and evaluates its performance.

bash
Copy code
python3.10 prototype/main.py
Ensure that you have the necessary Python libraries and dependencies installed. You can modify the script and parameters to customize the model's behavior and hyperparameters.
bash
Copy code
python3.10 -m pip install -r requirements.txt

#### Data
The project uses historical stock price data, which should be placed in the data directory. The data should include columns like 'ticker,' 'date,' 'open,' 'high,' 'low,' 'close,' and other relevant features. The data is separated into 5,541 individual csv files for each ticker.

#### Results
The model generates predictions for stock prices and evaluates its performance, displaying metrics such as the Root Mean Squared Error (RMSE) and feature importances, it kind of maybe works depending on how far into the future it tries to predict.

#### Visualizations
The project includes visualization scripts for analyzing the model's results, including Bokeh-based plots (visbokeh.py), Matplotlib Finance charts (vismplfinance.py), and Plotly visualizations (visplotly.py). You can explore these visualizations to gain insights into the model's predictions and actual stock prices.

If you happen to know what you are doing unlike me submit a PR :)

#### Known Issues
- will not predict values above a certain number and the predicions will plateau (mainly a problem when predicting >7days ahead)
- sometimes it looks like there appears noise in the y_test data which should not happen idk, this causes alignment issues in the visualizations
- vastly different performance depending on the stock(s) it is trained on or predicting, likely due to a certain stock's innate trends being more in line with the architecture of the algorithm but thats a theory.
- categorical features like the ticker symbol or other things I might add are dubious in XGBoost, they are currently being converted to numbers and represented that way but there is probably a better method
- when training the model on multiple stocks it falls apart rapidly, maybe this could be fixed with more categorical features like industry, country, or type.
