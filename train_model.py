import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("career_data.csv")

# Feature and target
X = df[['Math', 'Biology', 'English', 'Likes', 'Hobby']]
y = df['Career']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Math', 'Biology', 'English']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Likes', 'Hobby'])
    ])

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")