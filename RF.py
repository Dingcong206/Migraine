import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# 1. 数据加载与分离 (Data Loading and Separation)

# 文件路径
FILE_PATH = 'migraine_data.csv'
TARGET_COLUMN = 'Type' # 目标变量列名

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"false：can't find {FILE_PATH}。Please check the file path。")
    exit()

# 分离目标变量 (y) 和特征变量 (X)
y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])

#print("--- 数据集加载成功 ---")
#print(f"原始特征数量: {X.shape[1]} 列")
#print("-" * 30)

# 2. 数据预处理 (Data Preprocessing)
# 2.1 目标变量 (y) 标签编码
# 将 'Type' 列的文本标签（如 'Typical aura with migraine'）转换为数字 (0, 1, 2, ...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 存储编码后的类别名称，用于结果解读
class_names = le.classes_
print(f"already coded labels（{len(class_names)} ）：{class_names}")

# 2.2  (X)  (One-Hot Encoding)
X_processed = pd.get_dummies(X)

# 2.3 缺失值处理 (使用 0 填充剩余的 NaN 值)
X_processed = X_processed.fillna(0)

print(f"features after preprocessing : {X_processed.shape[1]} ")
print("-" * 30)

# 3. 数据集划分与模型训练 (Splitting and Training)


# 3.1 划分数据集 (70% 训练, 30% 测试)
# stratify=y_encoded 确保训练集和测试集中的类别比例与原数据集保持一致。
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print(f"training set: {X_train.shape[0]}，test set: {X_test.shape[0]}")

# 3.2 初始化并训练随机森林模型
# n_estimators=100 表示构建 100 棵决策树。n_jobs=-1 表示使用所有可用的 CPU 核心进行并行计算。
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train)
print("Training finish。")
print("-" * 30)

# 4. 模型预测与评估 (Prediction and Evaluation)


# 4.1 使用测试集进行预测
y_pred = rf_model.predict(X_test)

# 4.2 计算性能指标
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 打印详细的分类报告，使用原始标签名称
report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
)
print(f"Accuracy: {accuracy:.4f}**")
print("\nConfusion Matrix")
print(conf_matrix)
print("\nRF Prediction")
print(report)
print("-" * 30)


# 这里我们使用测试集的第一行数据作为示例
new_patient_features = X_test.iloc[[0]]

# 进行预测
new_prediction_encoded = rf_model.predict(new_patient_features)

# 将数字结果转换回原始的偏头痛类型标签
new_prediction_label = le.inverse_transform(new_prediction_encoded)

