from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from data_preprocessing import load_data, preprocess

# Load
df = load_data("../data/raw/data.csv")
X, y = preprocess(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "../models/model.pkl")

print("Model trained successfully!")
