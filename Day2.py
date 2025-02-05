from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = pd.read_csv("datasets/Iris.csv")

# Display the first few rows and a random sample
print("\n\nHead of the Dataset\n", data.head())
print("\n\nRandom sample from the Dataset:\n", data.sample(4))
print("\nTypes of Species in the Dataset:\t", data['Species'].unique())

# Define features and target
features = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
target = data['Species']

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='gini')
model.fit(features, target)

# Define unknown data
unknown_data = [[5.5, 4.6, 1, 2.4]]

# Convert unknown data to a DataFrame with matching feature names
unknown_data_df = pd.DataFrame(unknown_data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

# Make predictions
prediction = model.predict(unknown_data_df)
print("Unknown Data :", unknown_data,"\t Predicted Species :", prediction)

# Print details of the decision tree
print("\nDecision Tree Details:")

# Features used for splits
print("\nFeatures used at each node:")
for i, feature_index in enumerate(model.tree_.feature):
    if feature_index != -2:  # Non-leaf nodes
        print(f"\tNode {i}: Splits on feature {features.columns[feature_index]}")

# Thresholds for splits
print("\nThresholds at each node:")
for i, threshold in enumerate(model.tree_.threshold):
    if threshold != -2:  # Non-leaf nodes
        print(f"\tNode {i}: Threshold = {threshold}")

# Class distribution at each node
print("\nClass distributions at each node:")
for i, value in enumerate(model.tree_.value):
    print(f"\tNode {i}: Class counts = {value}")

# Entropy at each node
print("\nEntropy at each node:")
for i, entropy in enumerate(model.tree_.impurity):
    print(f"\tNode {i}: Entropy = {entropy}")

plt.figure(figsize=(12,12))
tree.plot_tree(model, fontsize=11, feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
plt.show()


X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.2, shuffle=True)

y_pred = model.predict(X_test)
df_check= pd.DataFrame({"Actual Species":y_test, "Predicted Species":y_pred})
print(df_check)
print(accuracy_score(y_test,y_pred))