import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# This code fetches a dataset containing information about various car models from the UC Irvine Machine Learning Repository. After converting the data into a pandas DataFrame, it filters out rows with missing values in the 'horsepower' and 'acceleration' columns. Then, it groups the data by 'car_name' and calculates the mean horsepower and acceleration for each car manufacturer. Next, it creates a scatter plot showing the relationship between the mean horsepower and acceleration across different car manufacturers. This visualization helps to understand how the average horsepower and acceleration vary across different car brands, providing insights into the performance characteristics of each manufacturer's vehicles.
# Fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# Convert data to DataFrame
df = pd.DataFrame(auto_mpg.data.features, columns=auto_mpg.variables['name'])

# Filter out rows with missing values in 'horsepower' and 'acceleration' columns
df = df.dropna(subset=['horsepower', 'acceleration'])

# Group data by 'car_name' and calculate mean horsepower and acceleration
manufacturer_stats = df.groupby('car_name').agg({'horsepower': 'mean', 'acceleration': 'mean'}).reset_index()
# Display the first few rows of the DataFrame
print(df.head())
# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(manufacturer_stats['horsepower'], manufacturer_stats['acceleration'], alpha=0.7)
plt.xlabel('Mean Horsepower')
plt.ylabel('Mean Acceleration')
plt.title('Relationship between Horsepower and Acceleration across Car Manufacturers')
plt.grid(True)
plt.show()
