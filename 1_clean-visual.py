import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from ucimlrepo import fetch_ucirepo






"""
Read Dataset
"""

# Fetch dataset
concrete_compressive_strength = fetch_ucirepo(id=165)

# Get features and targets
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# Combine them into one DataFrame
import pandas as pd
df = pd.concat([X, y], axis=1)


"""
Change Column Name
"""

# new column name definition
new_column_names = {
    'Cement': 'cement',
    'Blast Furnace Slag': 'slag',
    'Fly Ash': 'fly_ash',
    'Water': 'water',
    'Superplasticizer': 'plasticizer',
    'Coarse Aggregate': 'c_aggr',
    'Fine Aggregate': 'fi_aggr',
    'Age': 'age',
    'Concrete compressive strength': 'strength'
}

df.rename(columns=new_column_names, inplace=True)

df.describe().round(1)





"""
Claning Dataset
"""

# Check for missing values
print("\nMissing values in the dataset:")

# Handling Duplicates
# Drop duplicate rows
df = df.drop_duplicates()

# Save the cleaned dataset
cleaned_data_path = "Data/data_cleaned.xlsx"
df.to_excel(cleaned_data_path, index=False)

print("\nCleaned dataset:")
print(df.head())
print("\nSummary of cleaned dataset:")
print(df.info())






"""
Visualiztion
"""
# Features Distribution
num_rows = 3
num_cols = 3

# Get the number of features to plot
num_features = len(df.columns)

# Create a figure and axes for the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(11, 11))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

# Plot each feature in its respective subplot
for i, column in enumerate(df.columns):
    sns.histplot(df[column], kde=True, ax=axes[i])
    axes[i].set_title(f'{column} Distribution')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    
    # Save each plot as an individual image
    fig_single, ax_single = plt.subplots(figsize=(5, 5))
    sns.histplot(df[column], kde=True, ax=ax_single)
    ax_single.set_title(f'{column} Distribution')
    ax_single.set_xlabel(column)
    ax_single.set_ylabel('Frequency')
    plt.tight_layout()
    
    plt.close(fig_single)  # Close the single plot figure to free up memory

# Hide any unused subplots if the number of features is less than num_rows*num_cols
for j in range(num_features, num_rows*num_cols):
    fig.delaxes(axes[j])

# Define output folder and file path
output_path = Path('Results')
output_path.mkdir(parents=True, exist_ok=True)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig(output_path / 'all_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# --------------------------------------------------------------------------------------------
# Features Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df)
plt.title('Box Plot of Features')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.savefig(output_path / 'box_plot.png', dpi=300, bbox_inches='tight')


# --------------------------------------------------------------------------------------------
# Water/Cement Ratio vs Strength
df_28 = df[df['age'] == 28]

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

# Plot for Cement vs Strength
sns.regplot(x='cement', y='strength', data=df_28, scatter_kws={'s': 10}, ax=axes[0])
axes[0].set_title('Cement vs Strength (Age = 28d)')
axes[0].set_xlabel('a) Cement')
axes[0].set_ylabel('Strength')

# Plot for Water vs Strength
sns.regplot(x='water', y='strength', data=df_28, scatter_kws={'s': 10}, ax=axes[1])
axes[1].set_title('Water vs Strength (Age = 28d)')
axes[1].set_xlabel('b) Water')

# Plot for Plasticizer vs Strength
df_28["water_cement_ratio"] = df_28["water"] / df_28["cement"]
sns.regplot(x="water_cement_ratio", y="strength", data=df_28, scatter_kws={'s': 10}, ax=axes[2])
#sns.regplot(x='plasticizer', y='strength', data=df_28, scatter_kws={'s': 10}, ax=axes[2])
axes[2].set_title('Water/Cement Ratio vs Strength')
axes[2].set_xlabel('c) Water/Cement')

plt.savefig(output_path / 'strength_vs_c&w.png', dpi=300, bbox_inches='tight')
# --------------------------------------------------------------------------------------------

