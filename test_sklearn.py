from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Load MNIST test data
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Load the trained scikit-learn model (for example, logistic regression)
model = joblib.load('mnist_logistic_regression_model.pkl')

# Predict on the test data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('\nTest accuracy (scikit-learn model):', accuracy)
