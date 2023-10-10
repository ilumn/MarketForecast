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
import matplotlib.pyplot as plt
import joblib

# Directory where your CSV files are located
data_directory = 'proof of concept/data'

# Define lists of parameter values to iterate over
param_values = {
    'n_estimators': [10, 50, 100, 200,500,1000,2000],
    'learning_rate': [2.0,1,0.5,0.2,0.1,0.05,0.01],
    'max_depth': [5,10,20,40,100,175,250,400,1000,2000]
}

# Create empty lists to store RMSE values and parameter values
rmse_values = []
param_names = []

# Load all historical stock data for all tickers
def load_all_data_parallel(directory):
    files = [f for f in os.listdir(directory) if f.endswith('META.csv')]
    
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
    df = df[['ticker','date','year','month','day','open', 'high', 'low', 'close','SMA_50','SMA_200','RSI','EMA_12','EMA_26','MACD','Signal_Line']]  # You can add more features here
    
    # Define the target variable (next day's close price)
    df['target'] = df['close'].shift(-100)
    
    # Remove the last row to align X and y
    df.dropna(subset=['target'], inplace=True)

    # print(df)
    return df

# Calculate technical indicators
def calculate_technical_indicators(df):
    # Calculate Simple Moving Averages (SMA) for different time periods
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

# Train the model function
def train_model(X_train, y_train, n_estimators, learning_rate, max_depth):
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    model.fit(X_train, y_train)
    return model

# Evaluate the model function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# Sort the data chronologically
def sort_chronologically(data):
    data.sort_values(by='date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def check_fix_method():
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


# Define a function to update the performance chart during each iteration
def update_chart(param_name, rmse):
    plt.barh(param_name, rmse)
    plt.draw()
    plt.pause(0.01)

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

    # dump dataframe to csv file
    print("dumping csv")
    df.to_csv('test.csv', index=False)
    
    df = df[['ticker_encoded','year','month','day','open', 'high', 'low', 'close','SMA_50','SMA_200','RSI','EMA_12','EMA_26','MACD','Signal_Line','target']]  # You can add more features here

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
    print("y_train data type:", y_train.dtype)

    # Convert X_train and X_test to DMatrix with enable_categorical=True
    # train_dmatrix = DMatrix(X_train, label=y_train, enable_categorical=True)
    # test_dmatrix = DMatrix(X_test, label=y_test, enable_categorical=True)
 # Initialize the performance chart
    plt.figure(figsize=(12, 8))
    plt.xlabel("RMSE")
    plt.title("RMSE for Different Parameter Combinations")

    for n_estimators in param_values['n_estimators']:
        for learning_rate in param_values['learning_rate']:
            for max_depth in param_values['max_depth']:
                print(f"Training the model with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}...")

                model_filename = f'xgboost_model_n{n_estimators}_lr{learning_rate}_md{max_depth}.pkl'

                if os.path.exists(model_filename):
                    # Load the trained model from the file
                    print(f"Loading pre-trained model from {model_filename}")
                    model = joblib.load(model_filename)
                else:
                    # Train the model
                    model = train_model(X_train, y_train, n_estimators, learning_rate, max_depth)
                    # Save the trained model to a file
                    joblib.dump(model, model_filename)
                    print(f"Trained model saved to {model_filename}")

                print(f"Evaluating the model with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}...")
                rmse = evaluate_model(model, X_test, y_test)

                # Store RMSE and parameter values
                rmse_values.append(rmse)
                param_name = f"n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}"

                # Update the performance chart
                update_chart(param_name, rmse)

    # Keep the performance chart open until closed by the user
    plt.show()

if __name__ == "__main__":
    main()