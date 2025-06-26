import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Step 1: Load the CSV
df = pd.read_csv('dummy.csv')

# Step 2: Convert the 'Age' column to a clean NumPy array
ages = pd.to_numeric(df['Age'], errors='coerce').dropna().to_numpy()

# Step 3: Compute mean, variance, standard deviation
mean_age = np.mean(ages)
variance_age = np.var(ages, ddof=1)  # sample variance
std_dev_age = np.std(ages, ddof=1)   # sample standard deviation

# Step 4: Filter values
above_mean = ages[ages > mean_age]
below_mean = ages[ages < mean_age]

# Step 5: Print results
print("Mean Age:", mean_age)
print("Variance:", variance_age)
print("Standard Deviation:", std_dev_age)
print("Values Above Mean:", above_mean)
print("Values Below Mean:", below_mean)



import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("dummy.csv")

# Step 2: Display basic info
print("▶ HEAD:")
print(df.head())               # First 5 rows

print("\n▶ DESCRIBE:")
print(df.describe())           # Summary statistics for numerical columns

print("\n▶ INFO:")
df.info()                      # Column types and non-null counts
 
df_cleaned = df.dropna()

df['Age'].fillna(df['Age'].mean(), inplace=True)

 
def categorize_age(age):
    if age < 30:
        return "Low"
    elif 30 <= age <= 35:
        return "Medium"
    else:
        return "High"

df['AgeCategory'] = df['Age'].apply(categorize_age)

# Step 5: Display result
print("\n▶ Modified DataFrame with AgeCategory:")
print(df[['Name', 'Age', 'AgeCategory']].head(10))





# Step 1: Generate dummy data
np.random.seed(42)
height = np.random.normal(loc=165, scale=10, size=50)      # Heights in cm
latitude = np.random.uniform(low=10, high=50, size=50)     # Latitudes

# Step 2: Create DataFrame
df = pd.DataFrame({
    "Height": height,
    "Latitude": latitude
})

# Step 3: Prepare data for linear regression
X = np.array(df["Height"]).reshape(-1, 1)
y = df["Latitude"].values

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Step 5: Plot the data
plt.figure(figsize=(8, 5))
plt.scatter(df["Height"], df["Latitude"], color='blue', label='Data Points')
plt.plot(df["Height"], y_pred, color='red', label='Regression Line')
plt.xlabel("Height (cm)")
plt.ylabel("Latitude")
plt.title("Height vs. Latitude with Linear Regression")
plt.legend()
plt.grid(True)

# Step 6: Save the plot as PNG
plt.savefig("height_vs_latitude.png")
plt.show()
