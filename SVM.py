import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # 引入 StandardScaler
from sklearn.svm import SVC # 引入 SVM 模型
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ----------------------------------------------------
# 1. 数据加载与分离 (Data Loading and Separation)
# ----------------------------------------------------

FILE_PATH = 'migraine_data.csv'
TARGET_COLUMN = 'Type'

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"false：can't find file {FILE_PATH}。Please check file path。")
    exit()

y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])


print(f"raw features: {X.shape[1]} ")
print("-" * 30)

# ----------------------------------------------------
# 2. 数据预处理 (Data Preprocessing) - 新增特征缩放
# ----------------------------------------------------

# 2.1 目标变量 (y) 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"already coded labels（ {len(class_names)} ）：{class_names}")

# 2.2 特征变量 (X) 独热编码 (One-Hot Encoding)
X_processed = pd.get_dummies(X)

# 2.3 缺失值处理 (使用 0 填充剩余的 NaN 值)
X_processed = X_processed.fillna(0)

# 2.4 **【为 SVM 新增】特征标准化/缩放 (Standard Scaling)**
# 将所有特征缩放到均值为 0，标准差为 1，确保每个特征同等重要。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed) # 对所有特征进行缩放

print(f": {X_processed.shape[1]} ")
print("-" * 30)

# ----------------------------------------------------
# 3. 数据集划分与模型训练 (Splitting and Training)
# ----------------------------------------------------

# 3.1 划分数据集
# 注意：这里我们使用缩放后的 X_scaled 数据进行划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print(f"training set: {X_train.shape[0]}，test set: {X_test.shape[0]}")

# 3.2 初始化并训练支持向量机 (SVM) 模型
# 使用 SVC (Support Vector Classifier)，默认使用 'rbf' 核函数。
# gamma='scale' 自动根据特征数量调整核函数的参数。
svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)


# SVM 模型的训练时间可能比随机森林稍长
svm_model.fit(X_train, y_train)
print("Training finish。")
print("-" * 30)

# ----------------------------------------------------
# 4. 模型预测与评估 (Prediction and Evaluation)
# ----------------------------------------------------

# 4.1 使用测试集进行预测
y_pred = svm_model.predict(X_test)

# 4.2 计算性能指标
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
)

print(f"Accuracy: {accuracy:.4f}**")
print("\nConfusion Matrix")
print(conf_matrix)
print("\nSVM Prediction")
print(report)
print("-" * 30)
