import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
import os
import csv


path = "Data/data_cleaned.xlsx"
df = pd.read_excel(path)


"""
Random Forest
"""

# Grid Search Cross-Validation
X = df.drop('strength', axis=1)
y = df['strength']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=15)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=25, oob_score=True)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation MSE:", best_score)

# -----------------------------------------------------------------------------------------------------------------------
# Using Randomized Search
# Assuming df is your DataFrame and 'strength' is your target variable
X = df.drop('strength', axis=1)
y = df['strength']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=15)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=25, oob_score=True)

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 8),
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=25)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = random_search.best_params_
best_score = -random_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation MSE:", best_score)

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------










"""
Neural Networks
"""
# Input
input = df.drop('strength' , axis=1)
test_size = 0.2

# Target
target = df['strength']


# Normalization in MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
input_normalized = scaler.fit_transform(input)
#print(input)

scaler = MinMaxScaler()
target_normalized = scaler.fit_transform(target.array.reshape(-1,1))
#print(target)

# Splitting Data
X_train,X_test,Y_train,Y_test = train_test_split(input_normalized,target_normalized,test_size=0.2, random_state=15)


def create_model():

    # Set the random seed for reproducibility
    seed = 7
    
    # For Python random module
    random.seed(seed)
    
    # For NumPy
    np.random.seed(seed)
    
    # For TensorFlow
    tf.random.set_seed(seed)
    
    # Ensure reproducibility by limiting threads
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Define the neural network model
    activation_func = 'sigmoid'
    hl_1 = 8
    hl_2 = 4
    hl_3 = 2
    dropout = 0.05
    model = Sequential([
        # Input layer with the number of features as input_dim and first hidden layer as well
        Dense(hl_1, input_dim=8, activation= activation_func),  
        Dropout(dropout),
        # Second hidden layer with 15 neurons
        Dense(hl_2, activation= activation_func),
        Dropout(dropout),
        # Third hidden layer with 10 neurons
        Dense(hl_3, activation= activation_func),
        Dropout(dropout),
        # Output layer with 1 neuron for price prediction
        Dense(1, activation='linear')  # Assuming price prediction is a regression task
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model, hl_1, hl_2, hl_3, activation_func, dropout



for epochs in range(1000,3001,500):
    print(epochs)

    # Create the model using the function
    model, hl_1, hl_2, hl_3, activation_func, dropout = create_model()

    # Print the returned parameters for reference
    #print(f"Model Parameters: hl_1={hl_1}, hl_2={hl_2}, hl_3={hl_3}, activation={activation_func}, dropout={dropout}")

    # Train the model
    batch_size = 50

    # Suppress verbose output if not needed (set verbose=0 to hide epoch-by-epoch logs)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Calculating Training error
    Y_train_pred = model.predict(X_train)
    # Convert normalized data back to original scale
    original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
    original_Y_train_pred = scaler.inverse_transform(Y_train_pred.reshape(-1,1))
    train_error = mean_squared_error(original_Y_train, original_Y_train_pred)
    print("Mean squared error: %.3f" % train_error)
    
    # Calculating Test error
    Y_test_pred = model.predict(X_test)
    # Convert normalized data back to original scale
    original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
    original_Y_test_pred = scaler.inverse_transform(Y_test_pred.reshape(-1,1))
    test_error = mean_squared_error(original_Y_test, original_Y_test_pred)
    print("Mean squared error: %.3f" % test_error)

    # Add to csv file
    with open('NN_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write data
        # ['Model', 'HL_1', 'HL_2', 'HL_3', 'Act_F', 'epoc', 'Train Error', 'Test Error', 'Optimizer', 'Dropout']
        writer.writerow(['Neural Networks', hl_1, hl_2, hl_3, activation_func, epochs, train_error, test_error, 'adam', dropout])


