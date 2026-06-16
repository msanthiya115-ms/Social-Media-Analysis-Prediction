import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Drop Id column
df = df.drop("Id", axis=1)

# Encode categorical columns
encoders = {}

for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Target Variable
target = "How addicted do you feel to social media?"

X = df.drop(target, axis=1)
y = df[target]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoder.pkl", "wb"))

print("Model Saved Successfully")