from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from data_preprocessing import load_data, preprocess

# Load
df = load_data("../data/raw/data.csv")
X, y = preprocess(df)

model = joblib.load("../models/model.pkl")

y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nReport:\n", classification_report(y, y_pred))
