from flask import Flask, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

app = Flask(__name__)  # Initialize Flask app

# Load dataset and split into train & test sets
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

@app.route("/get_status", methods=["GET"])
def get_status():
    return jsonify({
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    })

if __name__ == "__main__":
    app.run(port=5000)
