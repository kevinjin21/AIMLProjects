# import libraries
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

#from json import loads, dumps

# set global seed value
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
FILENAME = '../Preprocessed_Data.xlsx'

# Using our normalizing function from before:


def normalize(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / \
            (df[column].max() - df[column].min())
    return df


def load_normalized_data(address):
    data = pd.read_excel(address)
    return normalize(data)


def feature_target_split(df):
    target = df['Total Sales']
    features = df.drop(columns=['Year', 'Month', 'Total Sales'])
    return features, target


def train_model(X_train, y_train):
    # initialize regressor
    forest = RandomForestRegressor(random_state=SEED_VALUE)
    forest = forest.fit(X_train, y_train)
    return forest


def get_model_metrics(model, X_test, y_test):
    # fit model and predict
    y_pred = model.predict(X_test)

    # mse and rmse as model metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**.5
    metrics = {'mse': mse,
               'rmse': rmse}
    return metrics


if __name__ == '__main__':
    # Load Data
    data = load_normalized_data(FILENAME)

    # Split Data
    features, target = feature_target_split(data)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=SEED_VALUE)

    # Train Model
    model = train_model(X_train, y_train)

    # Validate model with test data
    metrics = get_model_metrics(model, X_test, y_test)

    #result = X_test.to_json()
   # parsed = loads(result)
    #print(dumps(parsed, indent=4))


    # Save model
    pickle.dump(model, open('pharm_model.pkl', 'wb'))
