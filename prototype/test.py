import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor
import joblib

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print("Error loading the model:", str(e))
        return None

def load_testing_data(test_data_path):
    try:
        df = pd.read_csv(test_data_path)
        df = df[['ticker_encoded','year','month','day','open', 'high', 'low', 'close','SMA_50','SMA_200','RSI','EMA_12','EMA_26','MACD','Signal_Line','target']]
        return df
    except Exception as e:
        print("Error loading the testing data:", str(e))
        return None

def main():
    # Load the saved XGBoost model
    model_path = 'xgboost_model.pkl'
    model = load_model(model_path)

    if model is None:
        print("Exiting.")
        return

    print("Model loaded successfully.")

    # Load custom testing data from a CSV file
    test_data_path = "proof of concept/test.csv"

    testing_data = load_testing_data(test_data_path)

    if testing_data is None:
        print("Exiting.")
        return

    # Preprocess the testing data (make sure it has the same features as the training data)
    # For example, you might need to select the same columns and apply the same transformations
    
    # Assuming you have a function 'prepare_testing_data' for preprocessing
    # testing_data = prepare_testing_data(testing_data)

    # Predict using the loaded model
    X_test = testing_data.drop('target', axis=1)
    y_test = testing_data['target']

    y_pred = model.predict(X_test)

    # Evaluate the predictions (you can use the same evaluation function as in the first program)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"xtest:{X_test}\ny_test:{y_test}\ny_pred:{y_pred}")

    print(f"Root Mean Squared Error on custom testing data: {rmse}")

if __name__ == "__main__":
    main()
