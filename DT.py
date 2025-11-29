import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# 1. Data Loading and Separation


FILE_PATH = 'migraine_data.csv'
TARGET_COLUMN = 'Type'

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: {FILE_PATH} can't be found. Please check the file path.")
    exit()

y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])


print("-" * 30)


# 2. Data Preprocessing


# 2.1 Label Encoding the Target Variable (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
#print(f"Encoded classification labels (Total {len(class_names)}): {class_names}")

# 2.2 One-Hot Encoding X
X_processed = pd.get_dummies(X)

# 2.3 Handle missing values (e.g., fill with 0)
X_processed = X_processed.fillna(0)

# Note: Decision Trees are not sensitive to distance, so feature scaling (StandardScaler) is not used here.


#print(f"Number of features after preprocessing: {X_processed.shape[1]}")
print("-" * 30)


# 3. Splitting and Training


X_train, X_test, y_train, y_test = train_test_split(
    X_processed, # Use preprocessed data
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print(f"Number of training samples: {X_train.shape[0]}, Number of test samples: {X_test.shape[0]}")

# 3.2 Initialize and Train Decision Tree Model

dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)


dt_model.fit(X_train, y_train)
print("-" * 30)


# 4. Prediction and Evaluation


y_pred = dt_model.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nDT Prediction Report:")
print(report)
print("-" * 30)