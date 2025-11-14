import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from scikeras.wrappers import KerasRegressor
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



path = "Data/data_cleaned.xlsx"
df = pd.read_excel(path)

"""
Linear Regression
"""
print("Linear Regression")

df_28 = df[df["age"] == 28]

# Model 1: "Features: cement, water"
print("Model 1: Features: cement, water")

# Input
input = df_28[['cement' , 'water']]
test_size = 0.2

# Target
target = df_28['strength']


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

# Training Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Calculating Training error
# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred = scaler.inverse_transform(Y_train_pred.reshape(-1,1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

#Train_error_list.append([2, train_error])

# Calculating Test error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred = scaler.inverse_transform(Y_test_pred.reshape(-1,1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

# Check if the file exists and delete it if it does
if os.path.exists('featue selection results.csv'):
    os.remove('featue selection results.csv')


# Create a CSV file for Train Results
with open('featue selection results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers
    writer.writerow(['Model', 'Parameters', 'Train Error', 'Test Error'])
    
    # Write data
    writer.writerow(['Linear Regression', 2, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------
# Model 2: "Features: cement, water, coarse aggregate, fine aggregate"
print("Model 2: Features: cement, water, coarse aggregate, fine aggregate")
# Input
input = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr']]
test_size = 0.2

# Target
target = df_28['strength']


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

# Training Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Calculating Training error
# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred = scaler.inverse_transform(Y_train_pred.reshape(-1,1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

#Train_error_list.append([2, train_error])

# Calculating Test error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred = scaler.inverse_transform(Y_test_pred.reshape(-1,1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Linear Regression', 4, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------
# Model 3: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier"
print("Model 3: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier")

# Input
input = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr' , 'slag' , 'fly_ash' , 'plasticizer']]
test_size = 0.2

# Target
target = df_28['strength']


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

# Training Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Calculating Training error
# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred = scaler.inverse_transform(Y_train_pred.reshape(-1,1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

#Train_error_list.append([2, train_error])

# Calculating Test error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred = scaler.inverse_transform(Y_test_pred.reshape(-1,1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Linear Regression', 7, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------
# Model 4: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age"
print("Model 4: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age")

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

# Training Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Calculating Training error
# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred = scaler.inverse_transform(Y_train_pred.reshape(-1,1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

#Train_error_list.append([2, train_error])

# Calculating Test error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred = scaler.inverse_transform(Y_test_pred.reshape(-1,1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Linear Regression', 8, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------    





# -----------------------------------------------------------------------------------------  
# -----------------------------------------------------------------------------------------  

"""
Polynomial Regression
"""

print("-----------------------------------------------------------------------------------------")
print("Polynomial Regression")

# Model 1: "Features: cement, water"
print("Model 1: Features: cement, water")

# Input 
input = df_28[['cement', 'water']]
test_size = 0.2

# Target (strength)
target = df_28['strength']

# Normalization (MinMax Scaling)
scaler_input = MinMaxScaler()
input_normalized = scaler_input.fit_transform(input)

scaler_target = MinMaxScaler()
target_normalized = scaler_target.fit_transform(target.array.reshape(-1,1))

# Polynomial Feature Transformation
degree = 2  
poly = PolynomialFeatures(degree=degree)
input_poly = poly.fit_transform(input_normalized)

# Splitting Data (train and test)
X_train, X_test, Y_train, Y_test = train_test_split(input_poly, target_normalized, test_size=0.2, random_state=15)

# Training Polynomial Regression Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

# Training Error
print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Convert normalized data back to original scale for training data
original_Y_train = scaler_target.inverse_transform(Y_train.reshape(-1, 1))
original_Y_train_pred = scaler_target.inverse_transform(Y_train_pred.reshape(-1, 1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

# Test Error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)

# Convert normalized data back to original scale for test data
original_Y_test = scaler_target.inverse_transform(Y_test.reshape(-1, 1))
original_Y_test_pred = scaler_target.inverse_transform(Y_test_pred.reshape(-1, 1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Ploynomial Degree 2', 2, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------  


# Model 2: "Features: cement, water, coarse aggregate, fine aggregate"
print("Model 2: Features: cement, water, coarse aggregate, fine aggregate")

# Input
input = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr']]
test_size = 0.2

# Target (strength)
target = df_28['strength']

# Normalization (MinMax Scaling)
scaler_input = MinMaxScaler()
input_normalized = scaler_input.fit_transform(input)

scaler_target = MinMaxScaler()
target_normalized = scaler_target.fit_transform(target.array.reshape(-1,1))

# Polynomial Feature Transformation
degree = 2  
poly = PolynomialFeatures(degree=degree)
input_poly = poly.fit_transform(input_normalized)

# Splitting Data (train and test)
X_train, X_test, Y_train, Y_test = train_test_split(input_poly, target_normalized, test_size=0.2, random_state=15)

# Training Polynomial Regression Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

# Training Error
print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Convert normalized data back to original scale for training data
original_Y_train = scaler_target.inverse_transform(Y_train.reshape(-1, 1))
original_Y_train_pred = scaler_target.inverse_transform(Y_train_pred.reshape(-1, 1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

# Test Error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)

# Convert normalized data back to original scale for test data
original_Y_test = scaler_target.inverse_transform(Y_test.reshape(-1, 1))
original_Y_test_pred = scaler_target.inverse_transform(Y_test_pred.reshape(-1, 1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Ploynomial Degree 2', 4, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------  
# Model 3: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier"
print("Model 3: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier")

# Input
input = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr' , 'slag' , 'fly_ash' , 'plasticizer']]
test_size = 0.2

# Target (strength)
target = df_28['strength']

# Normalization (MinMax Scaling)
scaler_input = MinMaxScaler()
input_normalized = scaler_input.fit_transform(input)

scaler_target = MinMaxScaler()
target_normalized = scaler_target.fit_transform(target.array.reshape(-1,1))

# Polynomial Feature Transformation
degree = 2  
poly = PolynomialFeatures(degree=degree)
input_poly = poly.fit_transform(input_normalized)

# Splitting Data (train and test)
X_train, X_test, Y_train, Y_test = train_test_split(input_poly, target_normalized, test_size=0.2, random_state=15)

# Training Polynomial Regression Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

# Training Error
print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Convert normalized data back to original scale for training data
original_Y_train = scaler_target.inverse_transform(Y_train.reshape(-1, 1))
original_Y_train_pred = scaler_target.inverse_transform(Y_train_pred.reshape(-1, 1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

# Test Error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)

# Convert normalized data back to original scale for test data
original_Y_test = scaler_target.inverse_transform(Y_test.reshape(-1, 1))
original_Y_test_pred = scaler_target.inverse_transform(Y_test_pred.reshape(-1, 1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Ploynomial Degree 2', 7, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------  

# Model 4: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age"
print("Model 4: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age")

# Input
input = df.drop('strength' , axis=1)
test_size = 0.2

# Target (strength)
target = df['strength']

# Normalization (MinMax Scaling)
scaler_input = MinMaxScaler()
input_normalized = scaler_input.fit_transform(input)

scaler_target = MinMaxScaler()
target_normalized = scaler_target.fit_transform(target.array.reshape(-1,1))

# Polynomial Feature Transformation
degree = 2  
poly = PolynomialFeatures(degree=degree)
input_poly = poly.fit_transform(input_normalized)

# Splitting Data (train and test)
X_train, X_test, Y_train, Y_test = train_test_split(input_poly, target_normalized, test_size=0.2, random_state=15)

# Training Polynomial Regression Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

# Training Error
print("Train Error")
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Normalized Mean Squared Error: %.3f" % train_error)

# Convert normalized data back to original scale for training data
original_Y_train = scaler_target.inverse_transform(Y_train.reshape(-1, 1))
original_Y_train_pred = scaler_target.inverse_transform(Y_train_pred.reshape(-1, 1))
train_error_original = mean_squared_error(original_Y_train, original_Y_train_pred)
print("Original Mean Squared Error: %.3f" % train_error_original)

# Test Error
print("Test Error")
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Normalized Mean Squared Error: %.3f" % test_error)

# Convert normalized data back to original scale for test data
original_Y_test = scaler_target.inverse_transform(Y_test.reshape(-1, 1))
original_Y_test_pred = scaler_target.inverse_transform(Y_test_pred.reshape(-1, 1))
test_error_original = mean_squared_error(original_Y_test, original_Y_test_pred)
print("Original Mean Squared Error: %.3f" % test_error_original)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Ploynomial Degree 2', 8, train_error_original, test_error_original])

# -----------------------------------------------------------------------------------------  
# -----------------------------------------------------------------------------------------  

"""
Random Forest
"""

print("-----------------------------------------------------------------------------------------")
print("Random Forest")

# Model 1: "Features: cement, water"
print("Model 1: Features: cement, water")

# Separate features (X) and target variable (y)
X = df_28[['cement' , 'water']]  
y = df_28['strength']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=500, random_state=25, oob_score= True)  # n_estimators as per your choice

# Train the model
rf_regressor.fit(X_train, y_train)

#Evaluate the train set
y_train_pred = rf_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error for Train:", mse_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the test set
mse_test = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Test:", mse_test)

# Add to results
with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Random Forest', 2, mse_train, mse_test])


# -----------------------------------------------------------------------------------------  
# Model 2: "Features: cement, water, coarse aggregate, fine aggregate"
print("Model 2: Features: cement, water, coarse aggregate, fine aggregate")

# Separate features (X) and target variable (y)
X = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr']]  
y = df_28['strength']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=500, random_state=25, oob_score= True)  # n_estimators as per your choice

# Train the model
rf_regressor.fit(X_train, y_train)

#Evaluate the train set
y_train_pred = rf_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error for Train:", mse_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the test set
mse_test = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Test:", mse_test)

# Add to results
with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Random Forest', 4, mse_train, mse_test])


# -----------------------------------------------------------------------------------------  
# Model 3: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier"
print("Model 3: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier")


# Separate features (X) and target variable (y)
X = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr' , 'slag' , 'fly_ash' , 'plasticizer']]
y = df_28['strength']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=500, random_state=25, oob_score= True)  # n_estimators as per your choice

# Train the model
rf_regressor.fit(X_train, y_train)

#Evaluate the train set
y_train_pred = rf_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error for Train:", mse_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the test set
mse_test = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Test:", mse_test)

# Add to results
with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Random Forest', 7, mse_train, mse_test])



# -----------------------------------------------------------------------------------------  
# Model 4: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age"
print("Model 4: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age")


# Separate features (X) and target variable (y)
X = df.drop('strength' , axis=1)
y = df['strength']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=500, random_state=25, oob_score= True)  # n_estimators as per your choice

# Train the model
rf_regressor.fit(X_train, y_train)

#Evaluate the train set
y_train_pred = rf_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error for Train:", mse_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the test set
mse_test = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Test:", mse_test)

# Add to results
with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Random Forest', 8, mse_train, mse_test])

# -----------------------------------------------------------------------------------------  
# -----------------------------------------------------------------------------------------  

"""
Neural Networks
"""

print("-----------------------------------------------------------------------------------------")
print("Neural Networks")


# -----------------------------------------------------------------------------------------  
# Model 1: "Features: cement, water"
print("Model 1: Features: cement, water")

# Input
input = df_28[['cement' , 'water']]
test_size = 0.2

# Target
target = df_28['strength']


# Normalization in MinMax
scaler = MinMaxScaler()
input_normalized = scaler.fit_transform(input)
#print(input)

scaler = MinMaxScaler()
target_normalized = scaler.fit_transform(target.array.reshape(-1,1))
#print(target)

# Splitting Data
X_train,X_test,Y_train,Y_test = train_test_split(input_normalized,target_normalized,test_size=0.2, random_state=15)

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
activation_func = 'relu'
model = Sequential([
    Dense(4, input_dim=2, activation= activation_func),  
    Dense(3, activation= activation_func),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the summary of the model
model.summary()

epoc = 600

# Create a KerasRegressor based on the model
estimator = KerasRegressor(model=model, epochs=epoc, batch_size=50, verbose=0)

# Define the k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform k-fold cross-validation
mse_scores = cross_val_score(estimator, X_train, Y_train, cv=kfold, scoring="neg_mean_squared_error")

mse_scores = -mse_scores

history = estimator.fit(X_train, Y_train)

# Calculating Training error
Y_train_pred_nn = estimator.predict(X_train)

# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred_nn = scaler.inverse_transform(Y_train_pred_nn.reshape(-1,1))

# Mean Squared Error
train_mse_nn = round(mean_squared_error(original_Y_train, original_Y_train_pred_nn), 3)
print("Neural Network Mean Squared Error (Train): %.3f" % train_mse_nn)

# Predicting on test data
Y_test_pred_nn = estimator.predict(X_test)

# Convert normalized data back to original scale
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred_nn = scaler.inverse_transform(Y_test_pred_nn.reshape(-1,1))

# Test Mean Squared Error
test_mse_nn = round(mean_squared_error(original_Y_test, original_Y_test_pred_nn), 3)
print("Neural Network Mean Squared Error (Test): %.3f" % test_mse_nn)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Neural Networks', 2, train_mse_nn, test_mse_nn])

# -----------------------------------------------------------------------------------------  
# Model 2: "Features: cement, water, coarse aggregate, fine aggregate"
print("Model 2: Features: cement, water, coarse aggregate, fine aggregate")

# Input
input = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr']]
test_size = 0.2

# Target
target = df_28['strength']


# Normalization in MinMax
scaler = MinMaxScaler()
input_normalized = scaler.fit_transform(input)
#print(input)

scaler = MinMaxScaler()
target_normalized = scaler.fit_transform(target.array.reshape(-1,1))
#print(target)

# Splitting Data
X_train,X_test,Y_train,Y_test = train_test_split(input_normalized,target_normalized,test_size=0.2, random_state=15)

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
activation_func = 'relu'
model = Sequential([
    Dense(4, input_dim=4, activation= activation_func),  
    Dense(3, activation= activation_func),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the summary of the model
model.summary()

epoc = 600

# Create a KerasRegressor based on the model
estimator = KerasRegressor(model=model, epochs=epoc, batch_size=50, verbose=0)

# Define the k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform k-fold cross-validation
mse_scores = cross_val_score(estimator, X_train, Y_train, cv=kfold, scoring="neg_mean_squared_error")

mse_scores = -mse_scores

history = estimator.fit(X_train, Y_train)

# Calculating Training error
Y_train_pred_nn = estimator.predict(X_train)

# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred_nn = scaler.inverse_transform(Y_train_pred_nn.reshape(-1,1))

# Mean Squared Error
train_mse_nn = round(mean_squared_error(original_Y_train, original_Y_train_pred_nn), 3)
print("Neural Network Mean Squared Error (Train): %.3f" % train_mse_nn)

# Predicting on test data
Y_test_pred_nn = estimator.predict(X_test)

# Convert normalized data back to original scale
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred_nn = scaler.inverse_transform(Y_test_pred_nn.reshape(-1,1))

# Test Mean Squared Error
test_mse_nn = round(mean_squared_error(original_Y_test, original_Y_test_pred_nn), 3)
print("Neural Network Mean Squared Error (Test): %.3f" % test_mse_nn)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Neural Networks', 4, train_mse_nn, test_mse_nn])
# -----------------------------------------------------------------------------------------  

# Model 3: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier"
print("Model 3: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier")

# Input
input = df_28[['cement' , 'water' , 'c_aggr' , 'fi_aggr' , 'slag' , 'fly_ash' , 'plasticizer']]
test_size = 0.2

# Target
target = df_28['strength']


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
activation_func = 'relu'
model = Sequential([
    Dense(4, input_dim=7, activation= activation_func),  
    Dense(3, activation= activation_func),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the summary of the model
model.summary()

epoc = 600

# Create a KerasRegressor based on the model
estimator = KerasRegressor(model=model, epochs=epoc, batch_size=50, verbose=0)

# Define the k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform k-fold cross-validation
mse_scores = cross_val_score(estimator, X_train, Y_train, cv=kfold, scoring="neg_mean_squared_error")

mse_scores = -mse_scores

history = estimator.fit(X_train, Y_train)

# Calculating Training error
Y_train_pred_nn = estimator.predict(X_train)

# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred_nn = scaler.inverse_transform(Y_train_pred_nn.reshape(-1,1))

# Mean Squared Error
train_mse_nn = round(mean_squared_error(original_Y_train, original_Y_train_pred_nn), 3)
print("Neural Network Mean Squared Error (Train): %.3f" % train_mse_nn)

# Predicting on test data
Y_test_pred_nn = estimator.predict(X_test)

# Convert normalized data back to original scale
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred_nn = scaler.inverse_transform(Y_test_pred_nn.reshape(-1,1))

# Test Mean Squared Error
test_mse_nn = round(mean_squared_error(original_Y_test, original_Y_test_pred_nn), 3)
print("Neural Network Mean Squared Error (Test): %.3f" % test_mse_nn)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Neural Networks', 7, train_mse_nn, test_mse_nn])

    
# -----------------------------------------------------------------------------------------  

# Model 4: "Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age"
print("Model 4: Features: cement, water, coarse aggregate, fine aggregate, slag, fly_ash, plastizier, age")

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
activation_func = 'relu'
model = Sequential([
    Dense(4, input_dim=8, activation= activation_func),  
    Dense(3, activation= activation_func),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the summary of the model
model.summary()

epoc = 600

# Create a KerasRegressor based on the model
estimator = KerasRegressor(model=model, epochs=epoc, batch_size=50, verbose=0)

# Define the k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform k-fold cross-validation
mse_scores = cross_val_score(estimator, X_train, Y_train, cv=kfold, scoring="neg_mean_squared_error")

mse_scores = -mse_scores

history = estimator.fit(X_train, Y_train)

# Calculating Training error
Y_train_pred_nn = estimator.predict(X_train)

# Convert normalized data back to original scale
original_Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
original_Y_train_pred_nn = scaler.inverse_transform(Y_train_pred_nn.reshape(-1,1))

# Mean Squared Error
train_mse_nn = round(mean_squared_error(original_Y_train, original_Y_train_pred_nn), 3)
print("Neural Network Mean Squared Error (Train): %.3f" % train_mse_nn)

# Predicting on test data
Y_test_pred_nn = estimator.predict(X_test)

# Convert normalized data back to original scale
original_Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))
original_Y_test_pred_nn = scaler.inverse_transform(Y_test_pred_nn.reshape(-1,1))

# Test Mean Squared Error
test_mse_nn = round(mean_squared_error(original_Y_test, original_Y_test_pred_nn), 3)
print("Neural Network Mean Squared Error (Test): %.3f" % test_mse_nn)

with open('featue selection results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write data
    writer.writerow(['Neural Networks', 8, train_mse_nn, test_mse_nn])



# -----------------------------------------------------------------------------------------  
# -----------------------------------------------------------------------------------------  


"""
Plot Feature Selection Error Results"
"""

path = "featue selection results.csv"
data = pd.read_csv(path)
print(data)


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot training error vs parameters
sns.lineplot(data=data, x='Parameters', y='Train Error', hue='Model', marker='o', palette='Set1', style='Model', markersize=10, ax=axes[0])
axes[0].set_xlabel('Parameters')
axes[0].set_ylabel('Train Error')
axes[0].set_title('Train Error vs Parameters')
axes[0].legend(title='Model', bbox_to_anchor=(1, 1), loc='upper left')
#axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjusted legend position
axes[0].grid(True)
axes[0].set_ylim(0, 120)

# Plot test error vs parameters
sns.lineplot(data=data, x='Parameters', y='Test Error', hue='Model', marker='o', palette='Set1', style='Model', markersize=10, ax=axes[1])
axes[1].set_xlabel('Parameters')
axes[1].set_ylabel('Test Error')
axes[1].set_title('Test Error vs Parameters')
axes[1].legend(title='Model', bbox_to_anchor=(1, 1), loc='upper left')
#axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjusted legend position
axes[1].grid(True)
axes[1].set_ylim(0, 120)

output_path = Path('Results')

# Adjust layout
plt.tight_layout()
plt.savefig(output_path / 'Error_parameters.png')

# ----------------------------------------------------------------------------------------------
# Set up subplots
fig, axes = plt.subplots(1, 4, figsize=(11, 3))

# Filter and plot Linear Regression
data_f = data[data['Model'] == 'Linear Regression']
axes[0].plot(data_f['Parameters'], data_f['Train Error'], marker='o', label='Train Error')
axes[0].plot(data_f['Parameters'], data_f['Test Error'], marker='x', label='Test Error')
axes[0].set_xlabel('Parameters')
axes[0].set_ylabel('Error')
axes[0].set_title('Linear Regression')
axes[0].legend()
axes[0].grid(True)
axes[0].set_ylim(0, 120)

# Filter and plot Polynomial Regression
data_f = data[data['Model'] == 'Ploynomial Degree 2']
axes[1].plot(data_f['Parameters'], data_f['Train Error'], marker='o', label='Train Error')
axes[1].plot(data_f['Parameters'], data_f['Test Error'], marker='x', label='Test Error')
axes[1].set_xlabel('Parameters')
axes[1].set_ylabel('Error')
axes[1].set_title('Ploynomial Degree 2')
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(0, 120)

# Filter and plot Random Forest
data_f = data[data['Model'] == 'Random Forest']
axes[2].plot(data_f['Parameters'], data_f['Train Error'], marker='o', label='Train Error')
axes[2].plot(data_f['Parameters'], data_f['Test Error'], marker='x', label='Test Error')
axes[2].set_xlabel('Parameters')
axes[2].set_ylabel('Error')
axes[2].set_title('Random Forest')
axes[2].legend()
axes[2].grid(True)
axes[2].set_ylim(0, 120)

# Filter and plot Linear Regression
data_f = data[data['Model'] == 'Neural Networks']
axes[3].plot(data_f['Parameters'], data_f['Train Error'], marker='o', label='Train Error')
axes[3].plot(data_f['Parameters'], data_f['Test Error'], marker='x', label='Test Error')
axes[3].set_xlabel('Parameters')
axes[3].set_ylabel('Error')
axes[3].set_title('Neural Networks')
axes[3].legend()
axes[3].grid(True)
axes[3].set_ylim(0, 120)

# Adjust layout
plt.tight_layout()
plt.savefig(output_path / 'Error_parameters_sep.png')

