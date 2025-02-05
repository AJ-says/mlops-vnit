# Import necessary libraries
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0 = setosa, 1 = versicolor, 2 = virginica

mlflow.set_tracking_uri(uri="mlflow_tracking")

# Set the experiment name
mlflow.set_experiment("AzH - MLops")

## enable autologging
mlflow.sklearn.autolog()

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

with mlflow.start_run():
    # Step 3: Create a logistic regression model and fit it to the training data
    model = LogisticRegression(max_iter=200)  # max_iter set higher to ensure convergence
    model.fit(X_train, y_train)
    
    # Step 4: Predict the labels on the test set
    y_pred = model.predict(X_test)
    
    # Step 5: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    
    print(f"Accuracy: {accuracy}")
    mlflow.log_metric("accuracy",accuracy)
    print("Classification Report:")
    print(report)

mlflow.end_run()
