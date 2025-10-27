import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.family"] = "Arial"  # 设置全局字体为Times New Roman
plt.rcParams["font.size"] = 8  # 设置全局字体大小为10pt

"""
data = pd.read_csv("WBCD.csv")
y = data['Diagnosis'].ravel()
X = data.drop(['Diagnosis'], axis=1)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=29)

model = LinearRegression()

# 训练模型并获取特征权重
feature_weights = []
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    feature_weights.append(model.coef_)

# 对特征权重进行绝对值处理
abs_feature_weights = np.abs(np.mean(feature_weights, axis=0))

# 归一化处理
scaler = MinMaxScaler()
normalized_weights = scaler.fit_transform(abs_feature_weights.reshape(-1, 1)).flatten()

# 绘制一维热力图
plt.figure(figsize=(4, 4))
# 设置背景为灰色
sns.set_style("darkgrid")
sns.barplot(x=normalized_weights, y=X.columns, color='#E64B35',width=0.25)
plt.xlabel('Weight')
plt.ylabel('Features')
plt.title('WBCD',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})

# 设置y轴刻度标签
tick_labels = [str(i) for i in range(1, 10)]
plt.yticks(ticks=np.arange(9), labels=tick_labels)
plt.savefig("2-feature-1.svg",dpi=500)
plt.show()"""

"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

feature_count = []
accuracy_scores1 = []
accuracy_scores2 = []
accuracy_scores3 = []
accuracy_scores4 = []
accuracy_scores5 = []

precision_scores1 = []
precision_scores2 = []
precision_scores3 = []
precision_scores4 = []
precision_scores5 = []

recall_scores1 = []
recall_scores2 = []
recall_scores3 = []
recall_scores4 = []
recall_scores5 = []

f1_scores1 = []
f1_scores2 = []
f1_scores3 = []
f1_scores4 = []
f1_scores5 = []

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

for i in range(1, len(X.columns)+1):
    selected_features = normalized_weights.argsort()[-i:][::-1]
    X_selected = X.iloc[:, selected_features]
    print(X_selected)
    feature_count.append(i)

    model = SVC()
    scores1 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    print(scores1)
    accuracy_scores1.append(scores1['test_accuracy'].mean())
    precision_scores1.append(scores1['test_precision'].mean())
    recall_scores1.append(scores1['test_recall'].mean())
    f1_scores1.append(scores1['test_f1'].mean())

    model = MLPClassifier()
    scores2 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores2.append(scores2['test_accuracy'].mean())
    precision_scores2.append(scores2['test_precision'].mean())
    recall_scores2.append(scores2['test_recall'].mean())
    f1_scores2.append(scores2['test_f1'].mean())

    model = RandomForestClassifier()
    scores3 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores3.append(scores3['test_accuracy'].mean())
    precision_scores3.append(scores3['test_precision'].mean())
    recall_scores3.append(scores3['test_recall'].mean())
    f1_scores3.append(scores3['test_f1'].mean())

    model = XGBClassifier()
    scores4 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores4.append(scores4['test_accuracy'].mean())
    precision_scores4.append(scores4['test_precision'].mean())
    recall_scores4.append(scores4['test_recall'].mean())
    f1_scores4.append(scores4['test_f1'].mean())

    model = AdaBoostClassifier()
    scores5 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores5.append(scores5['test_accuracy'].mean())
    precision_scores5.append(scores5['test_precision'].mean())
    recall_scores5.append(scores5['test_recall'].mean())
    f1_scores5.append(scores5['test_f1'].mean())

plt.figure(figsize=(4, 3.1))
plt.plot(feature_count, accuracy_scores1, marker='o', markersize=3,label='SVM')
plt.plot(feature_count, accuracy_scores2, marker='s', markersize=3,label='ANN')
plt.plot(feature_count, accuracy_scores3, marker='^', markersize=3,label='RF')
plt.plot(feature_count, accuracy_scores4, marker='*', markersize=5,label='XGBoost')
plt.plot(feature_count, accuracy_scores5, marker='D', markersize=3,label='AdaBoost')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('WBCD',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("3-feature&acc-1.svg",dpi=500)
plt.show()


# 找到最大值的索引
print('特征选择前：')
print(accuracy_scores1[-1],precision_scores1[-1],recall_scores1[-1],f1_scores1[-1])
print(accuracy_scores2[-1],precision_scores2[-1],recall_scores2[-1],f1_scores2[-1])
print(accuracy_scores3[-1],precision_scores3[-1],recall_scores3[-1],f1_scores3[-1])
print(accuracy_scores4[-1],precision_scores4[-1],recall_scores4[-1],f1_scores4[-1])
print(accuracy_scores5[-1],precision_scores5[-1],recall_scores5[-1],f1_scores5[-1])



max_index1 = np.argmax(accuracy_scores1)
print('最佳选择特征数量:',max_index1)
print(accuracy_scores1[max_index1],precision_scores1[max_index1],recall_scores1[max_index1],f1_scores1[max_index1])


max_index2 = np.argmax(accuracy_scores2)
print('最佳选择特征数量:',max_index2)
print(accuracy_scores2[max_index2],precision_scores2[max_index2],recall_scores2[max_index2],f1_scores2[max_index2])


max_index3 = np.argmax(accuracy_scores3)
print('最佳选择特征数量:',max_index3)
print(accuracy_scores3[max_index3],precision_scores3[max_index3],recall_scores3[max_index3],f1_scores3[max_index3])


max_index4 = np.argmax(accuracy_scores4)
print('最佳选择特征数量:',max_index4)
print(accuracy_scores4[max_index4],precision_scores4[max_index4],recall_scores4[max_index4],f1_scores4[max_index4])


max_index5 = np.argmax(accuracy_scores5)
print('最佳选择特征数量:',max_index5)
print(accuracy_scores5[max_index5],precision_scores5[max_index5],recall_scores5[max_index5],f1_scores5[max_index5])
"""


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.family"] = "Arial"  # 设置全局字体为Times New Roman
plt.rcParams["font.size"] = 8  # 设置全局字体大小为10pt

data = pd.read_csv("breast_cancer_research/data/WDBC.CSV")
# 提取目标特征列（假设目标特征列是第一列之外的所有列）
target_features = data.iloc[:, 1:]

# 初始化StandardScaler对象
scaler = StandardScaler()

# 对目标特征列进行标准化
scaled_target_features = scaler.fit_transform(target_features)

# 将标准化后的目标特征列替换原始数据集中的目标特征列
data.iloc[:, 1:] = scaled_target_features

y = data['Diagnosis'].ravel()
X = data.drop(['Diagnosis'], axis=1)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=29)

model = LinearRegression()

# 训练模型并获取特征权重
feature_weights = []
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    feature_weights.append(model.coef_)

# 对特征权重进行绝对值处理
abs_feature_weights = np.abs(np.mean(feature_weights, axis=0))

# 归一化处理
scaler = MinMaxScaler()
normalized_weights = scaler.fit_transform(abs_feature_weights.reshape(-1, 1)).flatten()

# 绘制一维热力图
plt.figure(figsize=(4, 4))
# 设置背景为灰色
sns.set_style("darkgrid")
sns.barplot(x=normalized_weights, y=X.columns, color='#40B6C8')
plt.xlabel('Weight')
plt.ylabel('Features')
plt.title('WDBC',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})

# 设置y轴刻度标签
tick_labels = [str(i) for i in range(1, 31)]
plt.yticks(ticks=np.arange(30), labels=tick_labels)
plt.savefig("2-feature-2.svg",dpi=2000)
plt.show()


from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

feature_count = []
accuracy_scores1 = []
accuracy_scores2 = []
accuracy_scores3 = []
accuracy_scores4 = []
accuracy_scores5 = []

precision_scores1 = []
precision_scores2 = []
precision_scores3 = []
precision_scores4 = []
precision_scores5 = []

recall_scores1 = []
recall_scores2 = []
recall_scores3 = []
recall_scores4 = []
recall_scores5 = []

f1_scores1 = []
f1_scores2 = []
f1_scores3 = []
f1_scores4 = []
f1_scores5 = []

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

for i in range(1, len(X.columns)+1):
    selected_features = normalized_weights.argsort()[-i:][::-1]
    X_selected = X.iloc[:, selected_features]
    feature_count.append(i)

    model = SVC()
    scores1 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores1.append(scores1['test_accuracy'].mean())
    precision_scores1.append(scores1['test_precision'].mean())
    recall_scores1.append(scores1['test_recall'].mean())
    f1_scores1.append(scores1['test_f1'].mean())

    model = MLPClassifier()
    scores2 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores2.append(scores2['test_accuracy'].mean())
    precision_scores2.append(scores2['test_precision'].mean())
    recall_scores2.append(scores2['test_recall'].mean())
    f1_scores2.append(scores2['test_f1'].mean())

    model = RandomForestClassifier()
    scores3 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores3.append(scores3['test_accuracy'].mean())
    precision_scores3.append(scores3['test_precision'].mean())
    recall_scores3.append(scores3['test_recall'].mean())
    f1_scores3.append(scores3['test_f1'].mean())

    model = XGBClassifier()
    scores4 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores4.append(scores4['test_accuracy'].mean())
    precision_scores4.append(scores4['test_precision'].mean())
    recall_scores4.append(scores4['test_recall'].mean())
    f1_scores4.append(scores4['test_f1'].mean())

    model = AdaBoostClassifier()
    scores5 = cross_validate(model, X_selected, y, cv=10,scoring=scoring, n_jobs=-1)
    accuracy_scores5.append(scores5['test_accuracy'].mean())
    precision_scores5.append(scores5['test_precision'].mean())
    recall_scores5.append(scores5['test_recall'].mean())
    f1_scores5.append(scores5['test_f1'].mean())

plt.figure(figsize=(4, 3.1))
plt.plot(feature_count, accuracy_scores1, marker='o', markersize=3,label='SVM')
plt.plot(feature_count, accuracy_scores2, marker='s', markersize=3,label='ANN')
plt.plot(feature_count, accuracy_scores3, marker='^', markersize=3,label='RF')
plt.plot(feature_count, accuracy_scores4, marker='*', markersize=5,label='XGBoost')
plt.plot(feature_count, accuracy_scores5, marker='D', markersize=3,label='AdaBoost')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('WDBC',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("3-feature&acc-2.svg",dpi=500)
plt.show()


# 按照特征权重从大到小的顺序获取特征索引数组
sorted_idx = np.argsort(normalized_weights)[::-1]
print(sorted_idx)
# 找到最大值的索引
print('特征选择前：')
print(accuracy_scores1[-1],precision_scores1[-1],recall_scores1[-1],f1_scores1[-1])
print(accuracy_scores2[-1],precision_scores2[-1],recall_scores2[-1],f1_scores2[-1])
print(accuracy_scores3[-1],precision_scores3[-1],recall_scores3[-1],f1_scores3[-1])
print(accuracy_scores4[-1],precision_scores4[-1],recall_scores4[-1],f1_scores4[-1])
print(accuracy_scores5[-1],precision_scores5[-1],recall_scores5[-1],f1_scores5[-1])



max_index1 = np.argmax(accuracy_scores1)
print('最佳选择特征数量:',max_index1)
print(accuracy_scores1[max_index1],precision_scores1[max_index1],recall_scores1[max_index1],f1_scores1[max_index1])


max_index2 = np.argmax(accuracy_scores2)
print('最佳选择特征数量:',max_index2)
print(accuracy_scores2[max_index2],precision_scores2[max_index2],recall_scores2[max_index2],f1_scores2[max_index2])


max_index3 = np.argmax(accuracy_scores3)
print('最佳选择特征数量:',max_index3)
print(accuracy_scores3[max_index3],precision_scores3[max_index3],recall_scores3[max_index3],f1_scores3[max_index3])


max_index4 = np.argmax(accuracy_scores4)
print('最佳选择特征数量:',max_index4)
print(accuracy_scores4[max_index4],precision_scores4[max_index4],recall_scores4[max_index4],f1_scores4[max_index4])


max_index5 = np.argmax(accuracy_scores5)
print('最佳选择特征数量:',max_index5)
print(accuracy_scores5[max_index5],precision_scores5[max_index5],recall_scores5[max_index5],f1_scores5[max_index5])




"""
##########################################################################################
data = pd.read_csv("WDBC.csv")
# 提取目标特征列（假设目标特征列是第一列之外的所有列）
target_features = data.iloc[:, 1:]

# 初始化StandardScaler对象
scaler = StandardScaler()

# 对目标特征列进行标准化
scaled_target_features = scaler.fit_transform(target_features)

# 将标准化后的目标特征列替换原始数据集中的目标特征列
data.iloc[:, 1:] = scaled_target_features

y = data['Diagnosis'].ravel()
X = data.drop(['Diagnosis'], axis=1)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=29)

model = LinearRegression()

# 训练模型并获取特征权重
feature_weights = []
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    feature_weights.append(model.coef_)

# 对特征权重进行绝对值处理
abs_feature_weights = np.abs(np.mean(feature_weights, axis=0))

# 归一化处理
scaler = MinMaxScaler()
normalized_weights = scaler.fit_transform(abs_feature_weights.reshape(-1, 1)).flatten()

# 绘制一维热力图
plt.figure(figsize=(4, 4))
# 设置背景为灰色
sns.set_style("darkgrid")
sns.barplot(x=normalized_weights, y=X.columns, color='#40B6C8')
plt.xlabel('Weight')
plt.ylabel('Features')
plt.title('WDBC',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})

# 设置y轴刻度标签
tick_labels = [str(i) for i in range(1, 31)]
plt.yticks(ticks=np.arange(30), labels=tick_labels)
#plt.savefig("2-feature-2.svg",dpi=2000)
plt.show()

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

feature_count = []
accuracy_scores1 = []
accuracy_scores2 = []
accuracy_scores3 = []
accuracy_scores4 = []
accuracy_scores5 = []

for i in range(1, len(X.columns)+1):
    selected_features = normalized_weights.argsort()[-i:][::-1]
    X_selected = X.iloc[:, selected_features]
    feature_count.append(i)

    model = SVC()
    scores1 = cross_val_score(model, X_selected, y, cv=10)
    accuracy_scores1.append(scores1.mean())

    model = MLPClassifier()
    scores2 = cross_val_score(model, X_selected, y, cv=10)
    accuracy_scores2.append(scores2.mean())

    model = RandomForestClassifier()
    scores3 = cross_val_score(model, X_selected, y, cv=10)
    accuracy_scores3.append(scores3.mean())

    model = XGBClassifier()
    scores4 = cross_val_score(model, X_selected, y, cv=10)
    accuracy_scores4.append(scores4.mean())

    model = AdaBoostClassifier()
    scores5 = cross_val_score(model, X_selected, y, cv=10)
    accuracy_scores5.append(scores5.mean())

plt.figure(figsize=(4, 3.1))
plt.plot(feature_count, accuracy_scores1, marker='o', markersize=3,label='SVM')
plt.plot(feature_count, accuracy_scores2, marker='s', markersize=3,label='ANN')
plt.plot(feature_count, accuracy_scores3, marker='^', markersize=3,label='RF')
plt.plot(feature_count, accuracy_scores4, marker='*', markersize=5,label='XGBoost')
plt.plot(feature_count, accuracy_scores5, marker='D', markersize=3,label='AdaBoost')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('WDBC',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})
plt.grid(True)
plt.legend(loc='lower right')
#plt.savefig("3-feature&acc-2.svg",dpi=500)
plt.show()"""
