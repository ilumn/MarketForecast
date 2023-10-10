import os
import pandas as pd
import xgboost
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, DMatrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas import factorize
from tqdm import tqdm
from multiprocessing import Pool
import joblib
import visplotly as vis
import vismplfinance as visfin
import visbokeh
#import shap

n = 2000
lr = 0.1
md = 5

DATEOFFSET = 1

plt.style.use('bmh')

# Directory where your CSV files are located
data_directory = 'prototype/data'
stock = 'META'
# Load all historical stock data for all tickers
def load_all_data_parallel(directory):
    files = [f for f in os.listdir(directory) if f.endswith(f'{stock}.csv')]
    
    with Pool() as pool:
        all_data = list(tqdm(pool.imap(load_file, [os.path.join(directory, filename) for filename in files]), total=len(files), desc='Loading Data'))
    
    return pd.concat(all_data, ignore_index=True)

def load_file(file_path):
    df = pd.read_csv(file_path)
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Extract year, month, and day into separate columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = calculate_technical_indicators(df)



    df.dropna(inplace=True)
    df = df[['ticker','date','year','month','day','open', 'high', 'low', 'close','pct1D','pct7D','pct30D','pct90D','pct365D','SMA_50','SMA_200','RSI','EMA_12','EMA_26','MACD','Signal_Line']]  # You can add more features here
    
    # Define the target variable (next day's close price)
    df['target'] = df['close'].shift(-DATEOFFSET)
    
    # Remove the last row to align X and y
    df.dropna(subset=['target'], inplace=True)

    # print(df)
    return df

# Calculate technical indicators
def calculate_technical_indicators(df):
    # Calculate Simple Moving Averages (SMA) for different time periods
    df['pct1D'] = df['close'].pct_change(periods=1)
    df['pct7D'] = df['close'].pct_change(periods=7)
    df['pct30D'] = df['close'].pct_change(periods=30)
    df['pct90D'] = df['close'].pct_change(periods=90)
    df['pct365D'] = df['close'].pct_change(periods=395)
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Prepare data for training
def prepare_data(df):
    # # Drop missing values and irrelevant columns
    # df.dropna(inplace=True)
    # df = df[['open', 'high', 'low', 'close']]  # You can add more features here
    
    # # Define the target variable (next day's close price)
    # df['target'] = df[['open', 'high', 'low', 'close']].shift(-1)

    # # Remove the last row to align X and y
    # df.dropna(subset=['target'], inplace=True)
    
    # Split data into features and labels
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y

# Split data into training and testing sets
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md)  # Adjust hyperparameters
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(y_test)
    print(y_pred)

    # X_test.to_csv('X_test.csv',index=False)
    # y_test.to_csv('y_test.csv',index=False)
    # pd.DataFrame(y_pred).to_csv('y_pred.csv',index=False)


    # y_pred = []
    # mse_values = []

    # y_pred.append(model.predict(X_test.iloc[0]))

    # for i in range(len(X_test)):
    #     # Generate predictions for the current input
    #     y_pred_i = model.predict(y_pred.iloc[:i + 1])
    #     y_pred.append(y_pred_i[-1])  # Append the last prediction
        
    #     # Calculate RMSE for the current prediction
    #     mse_i = mean_squared_error(y_test.iloc[:i + 1], y_pred_i)
    #     mse_values.append(mse_i)

    # X_test['date'] = pd.to_datetime(X_test[['year', 'month', 'day']])
    # # X_test['date_60_days_future'] = X_test['date'] + DateOffset(days=60)

    
    # # Create a figure with subplots
    # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    
    # # Plot 'close' from X_test by date
    # axes[0].plot(X_test['date'], X_test['close'], color='purple', label='close')
    # axes[0].set_xlabel('Date')
    # axes[0].set_ylabel('close')
    # axes[0].legend()
    

    # # Plot y_pred by date
    # axes[1].plot(X_test['date'], y_pred, color='blue', label='y_pred')
    # axes[1].set_xlabel('Date')
    # axes[1].set_ylabel('y_pred')
    # axes[1].legend()
    
    # # Plot y_test by date
    # axes[2].plot(X_test['date'], y_test, color='red', label='y_test')
    # axes[2].set_xlabel('Date')
    # axes[2].set_ylabel('y_test')
    # axes[2].legend()
    
    # # Plot y_pred vs y_test
    # axes[3].scatter(y_pred, y_test, color='green')
    # axes[3].set_xlabel('y_pred')
    # axes[3].set_ylabel('y_test')

    # plt.tight_layout()
    # plt.show()
    vis.plot_graphs(X_test, y_pred, y_test, DATEOFFSET)
    visbokeh.plot_graphs(X_test, y_pred, y_test, DATEOFFSET)
    visfin.plot_graphs(X_test, y_pred, y_test, DATEOFFSET)
    rmse = np.sqrt(mse)
    return rmse


# # Define a function to plot graphs
# def plot_graphs(X_test, y_pred, y_test):
#     # Plot 'close' from X_test by date
#     # y_pred_offset = y_pred[60:]
#     # y_test_offset = y_test[60:]

#     # # # Offset dates by 60 days
#     # X_test_offset = X_test[]
#     # X_test_offset['date'] += pd.DateOffset(days=60)
#     plt.figure(figsize=(12, 6))
#     plt.plot(X_test['date'], X_test['close'], color='purple', label='Close (X_test)')
    
#     # Plot y_pred
#     plt.plot((X_test['date'] + pd.DateOffset(days=DATEOFFSET)), y_pred, color='blue', label='Predictions (y_pred)')
    
#     # Plot y_test
#     plt.plot((X_test['date'] + pd.DateOffset(days=DATEOFFSET)), y_test, color='red', label='Actual (y_test)')
    
#     plt.xlabel('Date')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.title('Comparison of Predictions and Actual Values')
#     plt.grid(True)
#     plt.show()

# Sort the data chronologically
def sort_chronologically(data):
    data.sort_values(by='date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def check_fix_method():
    return "y"
    user_input = input("Do you want to drop rows with NaN values (y)es/(n)o? ").strip().lower()
    if user_input in ["yes", "no", "y", "n"]:
        return user_input
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

def splitdata(X, y):
    # Filter data for training (years before 2023) and testing (years 2023 and 2024)
    X_train = X[X['year'] < 2021]
    X_test = X[(X['year'] == 2023)]
    y_train = y[X['year'] < 2021]
    y_test = y[(X['year'] == 2023)]
    
    return X_train, X_test, y_train, y_test

# Main function
def main():
    print("Loading historical stock data...")
    df = load_all_data_parallel(data_directory)

    drop_na = check_fix_method()
        
    if drop_na == "yes" or "y":
        print("Dropping invalid values")
        df.dropna(inplace=True)
    elif drop_na == "no" or "n":
        print("Filling invalid values with 0. WARNING: this may cause corruption in final model")
        df.fillna(0, inplace=True)
        
    df = sort_chronologically(df)
    
    encoder = LabelEncoder()
    df['ticker_encoded'] = encoder.fit_transform(df['ticker'].astype(str))

    # # dump dataframe to csv file
    # print("dumping csv")
    # df.to_csv('test.csv', index=False)
    
    df = df[['ticker_encoded','year','month','day','open', 'high', 'low', 'close','pct1D','pct7D','pct30D','pct90D','pct365D','SMA_50','SMA_200','RSI','EMA_12','EMA_26','MACD','Signal_Line','target']]  # You can add more features here

    # df['ticker_encoded'] = df['ticker_encoded'].astype('category')
    # df['date'] = df['date'].astype('category')


    print("Preparing data...")
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = splitdata(X, y)

     # Remove rows with NaN values from both X_train and y_train
    X_train = X_train.dropna()
    y_train = y_train.dropna()

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    # Check the dimensions of X_train and y_train
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Check the data types of y_train
    print("y_train data type:", y_train.dtypes)

    # Convert X_train and X_test to DMatrix with enable_categorical=True
    # train_dmatrix = DMatrix(X_train, label=y_train, enable_categorical=True)
    # test_dmatrix = DMatrix(X_test, label=y_test, enable_categorical=True)

    model_filename = f'models/MMSF_{stock}_n{n}_lr{lr}_md{md}.pkl'
    if os.path.exists(model_filename):
        # Load the trained model from the file
        print(f"Loading pre-trained model from {model_filename}")
        model = joblib.load(model_filename)
    else:
        # Train the model
        print("Training the model...")
        model = train_model(X_train, y_train)

        # Save the trained model to a file
        joblib.dump(model, model_filename)
        print(f"Trained model saved to {model_filename}")


    print("Evaluating the model...")
    rmse = evaluate_model(model, X_test, y_test)
    print(f"Root Mean Squared Error: {rmse}")

    feature_importances = model.feature_importances_
    print("Feature Importances:")
    for feature, importance in zip(X.columns, feature_importances):
        print(f"{feature}: {importance}")

    # explainer = shap.Explainer(model)
    # shap_values = explainer.shap_values(X_test.iloc[0])
    # shap.summary_plot(shap_values, X_test.iloc[0])    

    # tree_index = 0
    # fig, ax = plt.subplots(figsize=(20, 10))
    # xgboost.plot_tree(model, num_trees=tree_index, ax=ax)
    # plt.show()

if __name__ == "__main__":
    main()
