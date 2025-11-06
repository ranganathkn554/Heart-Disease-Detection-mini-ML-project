# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import joblib

# Load dataset
df = pd.read_csv("heart.csv")

# One-hot encode 'cp'
df = pd.get_dummies(df, columns=['cp'], drop_first=True)

# Balance dataset
df_majority = df[df.target == 0]
df_minority = df[df.target == 1]
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split features and target
X = df_balanced.drop('target', axis=1)
y = df_balanced['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(f"✅ Model trained with accuracy: {model.score(X_test, y_test)*100:.2f}%")

# Save model, scaler, and feature columns
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("✅ Model, scaler, and feature columns saved successfully!")
