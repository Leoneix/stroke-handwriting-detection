import sys
import os
import joblib

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_loader import StrokeDatasetLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


loader = StrokeDatasetLoader("data/raw")

X, y = loader.load_dataset()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])


pipeline.fit(X_train, y_train)


pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))


joblib.dump(pipeline, "saved_models/model.pkl")

print("Model saved")