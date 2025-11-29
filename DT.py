import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # 결정 트리 모델 임포트
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ----------------------------------------------------
# 1. 데이터 로드 및 분리 (Data Loading and Separation)
# ----------------------------------------------------

FILE_PATH = 'migraine_data.csv'
TARGET_COLUMN = 'Type'

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"오류: {FILE_PATH} 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])

print("--- 데이터셋 로드 완료 ---")
print("-" * 30)

# ----------------------------------------------------
# 2. 데이터 전처리 (Data Preprocessing)
# ----------------------------------------------------

# 2.1 목표 변수 (y) 标签 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"인코딩된 분류 레이블 (총 {len(class_names)} 개): {class_names}")

# 2.2 특성 변수 (X) 独热 인코딩 (One-Hot Encoding)
X_processed = pd.get_dummies(X)

# 2.3 결측치 처리
X_processed = X_processed.fillna(0)

# 참고: 결정 트리는 거리에 민감하지 않으므로, 이 단계에서는 특성 축척 (StandardScaler)을 사용하지 않습니다.

print(f"전처리 후 특성 개수: {X_processed.shape[1]} 개")
print("-" * 30)

# ----------------------------------------------------
# 3. 데이터 분할 및 모델 훈련 (Splitting and Training)
# ----------------------------------------------------

# 3.1 훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, # 전처리된 데이터 사용
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print(f"훈련 데이터 샘플 수: {X_train.shape[0]}, 테스트 데이터 샘플 수: {X_test.shape[0]}")

# 3.2 결정 트리 모델 초기화 및 훈련
# criterion='gini'는 분할 기준을 Gini 불순도로 설정합니다.
# max_depth는 트리의 깊이를 제한하여 과적합을 방지할 수 있는 중요한 초매개변수입니다.
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)

print("--- 결정 트리 (DT) 모델 훈련 중... ---")
dt_model.fit(X_train, y_train)
print("모델 훈련 완료.")
print("-" * 30)

# ----------------------------------------------------
# 4. 모델 예측 및 평가 (Prediction and Evaluation)
# ----------------------------------------------------

# 4.1 테스트 데이터로 예측 수행
y_pred = dt_model.predict(X_test)

# 4.2 성능 지표 계산 및 출력
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