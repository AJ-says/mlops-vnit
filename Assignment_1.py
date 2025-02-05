import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore")

# Set up the experiment
mlflow.set_experiment("RandomForest_Hyperparameter_Tuning")

# Load the dataset from UCI Repository
url = "datasets/adult_income_dataset.csv"
columns = [
    "age", "workclass", "final_weight", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
data = pd.read_csv(url, header=None, names=columns, na_values=' ?')

# Display dataset info
# print(data.sample(10))
print("\nInfo on the dataset:\n")
print(data.info())
print(f"\n\nHead rows of the dataset:\n{data.head()}")

# Data preprocessing
print(f"\n\nInitial Data Shape: {data.shape}")
data.dropna(inplace=True)
print(f"\nData Shape after Dropping Missing Values: {data.shape}")
print(f"\n\nColumn names:\t{list(data.columns)}")

# Explore categorical features
print("\n\nFeatures with unique values (categorical only):\n")
for column in data.select_dtypes(include=['object']).columns:
    print(f"\t{column}:\t{data[column].unique()}\n")

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

print(data.info())

print("\n\nMapping of Categories to Encoded values:\n")
for column, encoder in label_encoders.items():
    print(f"\tColumn: {column}")
    # Display the mapping of categories to encoded values
    mapping = {class_label: encoded_label for encoded_label, class_label in enumerate(encoder.classes_)}
    print(f"\tMapping: {mapping}")
    print()

# Split features and target
X = data.drop('income', axis=1)
y = data['income']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

print("\nParameter Grid:")
for key, values in param_grid.items():
    print(f"\t{key}: {values}")

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Start an MLflow run
with mlflow.start_run():

    # Log all parameters being tested
    mlflow.log_param("param_grid", param_grid)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)  # Log best hyperparameters
    mlflow.log_metric("cross_validated_accuracy", best_score)  # Log cross-validation accuracy
    
    # Test accuracy on the testing set
    best_model = grid_search.best_estimator_
    mlflow.sklearn.log_model(best_model, "best_random_forest_model") # Log the best model
    
    # Test accuracy on the testing set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_accuracy) # Log test accuracy
    
    # Print results
    print("\nBest Model:", best_model)
    print("\nBest Hyperparameters:", best_params)
    print(f"\nCross-Validated Accuracy: {best_score:.3f}")
    print(f"\nTest Accuracy: {test_accuracy:.3f}\n")

