import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # 결정 트리 모델 임포트
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# 1. Data Loading and Separation


FILE_PATH = 'migraine_data.csv'
TARGET_COLUMN = 'Type'

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"False: {FILE_PATH} can't be found. Please check the file path.")
    exit()

y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])


print("-" * 30)


# 2. Data Preprocessing


# 2.1 목표 변수 (y) 标签 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
#print(f"인코딩된 분류 레이블 (총 {len(class_names)} 개): {class_names}")

# 2.2 One-Hot Encoding X
X_processed = pd.get_dummies(X)

# 2.3
X_processed = X_processed.fillna(0)

# 참고: 결정 트리는 거리에 민감하지 않으므로, 이 단계에서는 특성 축척 (StandardScaler)을 사용하지 않습니다.


#print(f"전처리 후 특성 개수: {X_processed.shape[1]} 개")
print("-" * 30)


# 3. Splitting and Training



X_train, X_test, y_train, y_test = train_test_split(
    X_processed, # 전처리된 데이터 사용
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print(f"훈련 데이터 샘플 수: {X_train.shape[0]}, 테스트 데이터 샘플 수: {X_test.shape[0]}")

# 3.2 결정 트리 모델 초기화 및 훈련

dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)


dt_model.fit(X_train, y_train)
print("-" * 30)


# 4. Prediction and Evaluation



y_pred = dt_model.predict(X_test)

#results
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
)

print(f"Accuracy: {accuracy:.4f}**")
print("\nConfusion Matrix:**")
print(conf_matrix)
print("\nDT Prediction:")
print(report)
print("-" * 30)