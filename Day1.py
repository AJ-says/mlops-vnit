# Logistics Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

# Create the dataset
df = pd.DataFrame({
    "height": [6.1, 5.2, 5.7, 5.9, 6.2, 5.3, 6, 5.2],
    "shoe_size": [9, 6, 8, 7, 8, 7, 9, 4],
    "gender": [0, 1, 1, 0, 0, 1, 1, 0]  # 1 = Female, 0 = Male
})
print(f"\n\n{df}")

# Plotting
fig, ax = plt.subplots()
ax.scatter(df[df["gender"] == 1]["height"], df[df["gender"] == 1]["shoe_size"], label="Female")
ax.scatter(df[df["gender"] == 0]["height"], df[df["gender"] == 0]["shoe_size"], label="Male")
ax.legend()
ax.set_xlabel("Height")
ax.set_ylabel("Shoe Size")
plt.show()

# Features and target variable
X = df[["height", "shoe_size"]]  # Input features
Y = df["gender"]                # Target variable

# Step 1: Initialize the Logistic Regression model
model = LogisticRegression()

# Step 2: Train the model
model.fit(X, Y)

# Step 3: Predicting an unknown value
X_unknown = pd.DataFrame([[5.5, 9]], columns=["height", "shoe_size"])
prediction = model.predict(X_unknown)

# Display results
print(f"\nPrediction for X_unknown={X_unknown.values}: {prediction[0]} (0 = Male, 1 = Female)")
if prediction[0]==0: print("\tMale")
else: print("\tFemale")

# Logistic Regression coefficients
coefficients = model.coef_
intercept = model.intercept_

# Extract the coefficients as scalars
coef_height = coefficients[0][0]
coef_shoe_size = coefficients[0][1]
intercept = intercept[0]

# Predicting probability manually
X_unknown = pd.DataFrame([[5.5, 9]], columns=["height", "shoe_size"])

# Calculate the denominator
denominator = 1 + np.exp(
    -(coef_height * X_unknown["height"] + coef_shoe_size * X_unknown["shoe_size"] + intercept)
)

# Sigmoid output
probability = 1 / denominator
print(f"Probability of being female: {probability.iloc[0]:.4f}\n\n")

# Save the model
with open("model.pkl","wb") as fileobject:
    pickle.dump(model,fileobject)


# Load the model
with open("model.pkl", "rb") as fileobj:
    loaded_model = pickle.load(fileobj)

# Use the loaded model for prediction
X_test = pd.DataFrame([[5.2, 5]], columns=["height", "shoe_size"])
prediction = loaded_model.predict(X_test)
print(f"Predicted gender: {prediction[0]} (0=Male, 1=Female)")
if prediction[0]==0: print("\tMale")
else: print("\tFemale")
