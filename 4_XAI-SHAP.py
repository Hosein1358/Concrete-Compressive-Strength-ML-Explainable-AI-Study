import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from pathlib import Path
import matplotlib.pyplot as plt
import shap


path = "Data/data_cleaned.xlsx"
df = pd.read_excel(path)



"""
Random Forest Model
"""

# Input
# Model all Features
input = df.drop('strength' , axis=1)

# Target
target = df['strength']

# Normalization in MinMax
scaler = MinMaxScaler()
input_normalized = scaler.fit_transform(input)
#print(input)

# Splitting Data
X_train, X_test, Y_train, Y_test = train_test_split(input_normalized, target, test_size=0.2, random_state=15)

# Initialize the Random Forest Regressor with additional parameters
rf_regressor = RandomForestRegressor(
    n_estimators= 300,  # Reduce number of trees
    max_depth=  None,  # Limit the depth of the trees
    min_samples_split= 2,  # Increase minimum samples required to split
    min_samples_leaf= 1,  # Increase minimum samples required at a leaf node
    max_features= 'log2',  # Limit the number of features considered for splitting
    random_state=25,
    oob_score=True
)

# Train the model
rf_regressor.fit(X_train, Y_train)

#Evaluate the train set
Y_train_pred_rf = rf_regressor.predict(X_train)

# Mean Squared Error
train_mse_rf = round(mean_squared_error(Y_train, Y_train_pred_rf), 3)
print("Random Forest Mean Squared Error (Train): %.3f" % train_mse_rf)

# Mean Absolute Error
train_mae_rf = round(mean_absolute_error(Y_train, Y_train_pred_rf), 3)
print("Random Forest Mean Absolute Error (Train): %.3f" % train_mae_rf)

# R-squared Error
train_r2_rf = round(r2_score(Y_train, Y_train_pred_rf), 3)
print("Random Forest R-squared Error (Train): %.3f" % train_r2_rf)

# Predicting on test data
Y_test_pred_rf = rf_regressor.predict(X_test)

# Test Mean Squared Error
test_mse_rf = round(mean_squared_error(Y_test, Y_test_pred_rf), 3)
print("Polynomial Regression Mean Squared Error (Test): %.3f" % test_mse_rf)

# Test Mean Absolute Error
test_mae_rf = round(mean_absolute_error(Y_test, Y_test_pred_rf), 3)
print("Polynomial Regression Mean Absolute Error (Test): %.3f" % test_mae_rf)

# Test R-squared Error
test_r2_rf = round(r2_score(Y_test, Y_test_pred_rf), 3)
print("Polynomial Regression R-squared Error (Test): %.3f" % test_r2_rf)







"""
Feature Importance
"""
# Get feature importances
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances', fontsize = 16)
plt.bar(range(X_test.shape[1]), importances[indices], align='center')
plt.xticks(range(X_test.shape[1]), input.columns[indices], rotation=45, fontsize = 10)
plt.xlim([-1, X_test.shape[1]])
output_path = Path('Results')
plt.savefig(output_path / 'RF_feature_importance.png')






"""
SHAP
"""

# Ensure X_test is a DataFrame
X_test = pd.DataFrame(X_test, columns=input.columns)

# Explain the model predictions using SHAP
explainer = shap.TreeExplainer(rf_regressor)
shap_values = explainer.shap_values(X_test)

# Ensure SHAP values and input data have the correct shape
if len(shap_values) == 1:  # for models that output a single value, like regression
    shap_values = shap_values[0]

# Verify alignment of SHAP values and feature names
assert shap_values.shape[1] == len(X_test.columns), "Mismatch in SHAP values and feature names"

# Plot the SHAP summary plot with enhanced configuration
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", color_bar=True)
plt.savefig(output_path / 'shap_summary_plot.png', bbox_inches='tight')



"""
SHAP Waterfall for two Instances
"""

# Ensure X_test is a DataFrame
X_test = pd.DataFrame(X_test, columns=input.columns)

# Explain the model predictions using SHAP
explainer = shap.TreeExplainer(rf_regressor)
shap_values = explainer.shap_values(X_test)

# Ensure SHAP values and input data have the correct shape
if len(shap_values) == 1:  # for models that output a single value, like regression
    shap_values = shap_values[0]

# Plot the SHAP waterfall plot for multiple instances
shap.initjs()
fig, axes = plt.subplots(1, 2, figsize=(20, 5))  # Create a figure with 2 subplots

instance_indices = [0, 1]  # Indices of the instances to explain

for i, ax in enumerate(axes):
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[instance_indices[i]], 
            base_values=explainer.expected_value, 
            data=X_test.iloc[instance_indices[i]]
        ), 
        max_display=10, 
        show=False
    )
    plt.sca(ax)
    ax.set_title(f'Instance {instance_indices[i]}')  # Set the title for each subplot

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig(output_path / 'shap_waterfall_plots.png')


