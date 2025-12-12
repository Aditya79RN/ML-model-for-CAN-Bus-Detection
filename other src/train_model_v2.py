# Runs for days 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer  # For handling missing values
import joblib  # To save the model

# Load the dataset (Updated path)
print("🔹 Loading ML-ready v2 dataset...")
df = pd.read_csv("D:/CAN_ML_Project/processed/ml_ready_v2.csv")

# Check initial dataset shape and distribution of labels
print(f"📊 Dataset shape: {df.shape}, Labels: {df['Label'].value_counts().to_dict()}")

# Optional: Reduce memory usage by downcasting integers and floats
def reduce_memory_usage(df):
    """Reduce memory usage by downcasting numerical columns."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

df = reduce_memory_usage(df)

# Optional: Sample a subset of the dataset to reduce memory usage
# Uncomment this line to sample 10% of the dataset for training
# df = df.sample(frac=0.1, random_state=42)

# Split the dataset into features (X) and labels (y)
X = df.drop("Label", axis=1)
y = df["Label"]

# Handle missing values by filling them with the median value of each column
print("🔹 Handling missing values...")
imputer = SimpleImputer(strategy='median')  # You can also use 'mean', 'most_frequent', or 'constant'
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
print("🔹 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# Balance the dataset using SMOTE (to handle class imbalance)
print("🔹 Balancing dataset with SMOTE (if needed)...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"✅ Resampled dataset size: {X_train_resampled.shape}")

# Train the Random Forest model
print("🚀 Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)  # Limiting n_jobs to 1 to reduce memory usage
rf_clf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
print("📊 Model Performance:")
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
print("💾 Saving model as 'can_ml_model.pkl'...")
joblib.dump(rf_clf, 'can_ml_model.pkl')

# Save the resampled dataset for future use if needed
df_resampled = pd.DataFrame(X_train_resampled, columns=df.columns[:-1])
df_resampled['Label'] = y_train_resampled
df_resampled.to_csv('resampled_CAN_dataset.csv', index=False)

print("✅ Model training complete and saved!")
