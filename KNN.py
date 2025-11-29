import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier # 引入 KNN 模型
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ----------------------------------------------------
# 1. 数据加载与分离 (Data Loading and Separation)
# ----------------------------------------------------

FILE_PATH = 'migraine_data.csv'
TARGET_COLUMN = 'Type'

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"错误：未能找到文件 {FILE_PATH}。请检查文件路径。")
    exit()

y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])

print("--- 数据集加载成功 ---")
print("-" * 30)

# ----------------------------------------------------
# 2. 数据预处理 (Data Preprocessing) - 包含特征缩放
# ----------------------------------------------------

# 2.1 目标变量 (y) 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"已编码的分类标签（共 {len(class_names)} 类）：{class_names}")

# 2.2 特征变量 (X) 独热编码 (One-Hot Encoding)
X_processed = pd.get_dummies(X)

# 2.3 缺失值处理
X_processed = X_processed.fillna(0)

# 2.4 **特征标准化/缩放 (Standard Scaling)**
# KNN 依赖距离计算，因此标准化是必要的。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

print(f"预处理后特征数量: {X_scaled.shape[1]} 列")
print("-" * 30)

# ----------------------------------------------------
# 3. 数据集划分与模型训练 (Splitting and Training)
# ----------------------------------------------------

# 3.1 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, # 使用缩放后的数据
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)


#print(f"训练集样本数: {X_train.shape[0]}，测试集样本数: {X_test.shape[0]}")

# 3.2 初始化并训练 K-近邻 (KNN) 模型
# n_neighbors=5：这是 K 的值，即模型在进行预测时考虑最近的 5 个邻居。
# K 的选择是 KNN 模型的关键超参数，通常需要通过交叉验证进行优化。
knn_model = KNeighborsClassifier(n_neighbors=5)

# KNN 实际上在 fit 阶段只是存储数据，真正的计算发生在 predict 阶段
knn_model.fit(X_train, y_train)
print("Training finish")
print("-" * 30)

# ----------------------------------------------------
# 4. 模型预测与评估 (Prediction and Evaluation)
# ----------------------------------------------------

# 4.1 使用测试集进行预测
y_pred = knn_model.predict(X_test)

# 4.2 计算性能指标
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
print("\nKNN Prediction")
print(report)
print("-" * 30)