import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier # Import KNN Model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ----------------------------------------------------
# 1. Data Loading and Separation
# ----------------------------------------------------

FILE_PATH = 'migraine_data.csv'
TARGET_COLUMN = 'Type'

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: File {FILE_PATH} not found. Please check the file path.")
    exit()

y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])

#print("--- Dataset loaded successfully ---")
print("-" * 30)

# ----------------------------------------------------
# 2. Data Preprocessing - Including Feature Scaling
# ----------------------------------------------------

# 2.1 Label Encoding the Target Variable (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"Encoded classification labels (Total {len(class_names)} classes): {class_names}")

# 2.2 Feature Variables (X) One-Hot Encoding
X_processed = pd.get_dummies(X)

# 2.3 Handle missing values
X_processed = X_processed.fillna(0)

# 2.4 **Feature Standardization/Scaling (Standard Scaling)**
# Standardization is necessary because KNN relies on distance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

print(f"Number of features after preprocessing: {X_scaled.shape[1]} columns")
print("-" * 30)

# ----------------------------------------------------
# 3. Splitting and Training
# ----------------------------------------------------

# 3.1 Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, # Use scaled data
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)


#print(f"Number of training samples: {X_train.shape[0]}, Number of test samples: {X_test.shape[0]}")

# 3.2 Initialize and Train K-Nearest Neighbors (KNN) Model
# n_neighbors=5: This is the value of K, meaning the model considers the 5 nearest neighbors for prediction.
# The choice of K is a critical hyperparameter and usually requires optimization through cross-validation.
knn_model = KNeighborsClassifier(n_neighbors=5)

# KNN essentially just stores the data during the fit phase; the actual computation happens during prediction.
knn_model.fit(X_train, y_train)
print("Training finish")
print("-" * 30)

# ----------------------------------------------------
# 4. Model Prediction and Evaluation
# ----------------------------------------------------

# 4.1 Make predictions using the test set
y_pred = knn_model.predict(X_test)

# 4.2 Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
)


print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix")
print(conf_matrix)
print("\nKNN Prediction Report")
print(report)
print("-" * 30)