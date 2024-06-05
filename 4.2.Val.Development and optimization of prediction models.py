
#1.内验证集的roc曲线
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
# 加载数据
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/1.总P矫正.csv')

# 导入数据
data_feature = data[['Location',

'age','pretracheal.NLNM','pretracheal.LNMR',
'IPNLNM','TCLNMR',]]
data_target = data['LN.prRLNM']

# 数值变量标准化
data_featureNum = data[['age','pretracheal.NLNM','pretracheal.LNMR',
'IPNLNM','TCLNMR',]]
scaler = MinMaxScaler()
data_featureNum = scaler.fit_transform(data_featureNum)

data_featureCata = data[['Location',]]
data_featureCata = np.array(data_featureCata)

# 整合数据
data_feature = np.hstack((data_featureCata, data_featureNum))

# 分为训练集和验证集
class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.2, random_state=0)

# 定义模型和参数空间
model_param_grid= {
       'Logistic Regression': (LogisticRegression(), {'C': [0.01, 0.1, 1, 10, 100]}),
    'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}),
    'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}),
    'Gradient Boosting': (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}),
    'Support Vector Machine': (SVC(probability=True), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
    'Neural Network': (MLPClassifier(), {'hidden_layer_sizes': [(10,), (20,), (30,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate': ['constant', 'adaptive'], 'learning_rate_init': [0.001, 0.01, 0.1]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc'), {'n_estimators': [50, 100, 500], 'colsample_bytree': [0, 1, 11],
                                                                            'max_depth': [5, 7, 10], 'learning_rate': [0.01,0.05, 0.1], 'subsample': [0.7, 0.8, 0.9], 'gamma': [0, 0.1, 0.5]})
}

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan','orange']

# 初始化最佳AUC和最佳模型
best_auc = 0
best_model_name = ''
best_model = None

# 创建评价指标的空列表
Val_accuracy_scores = []
Val_auc_scores = []
Val_precision_scores = []
Val_specificity_scores = []
Val_sensitivity_scores = []
Val_npv_scores = []
Val_ppv_scores = []
Val_recall_scores = []
Val_f1_scores = []
Val_fpr_scores = []

# 拟合模型并绘制ROC曲线
plt.figure(figsize=(8, 6))

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
    grid_search.fit(class_x_tra, class_y_tra, )
    best_model_temp = grid_search.best_estimator_

    # 计算验证集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_Val_pred_prob = best_model_temp.predict_proba(class_x_val)[:, 1]
    else:
        y_Val_pred_prob = best_model_temp.decision_function(class_x_val)

    # 计算AUC值
    auc = roc_auc_score(class_y_val, y_Val_pred_prob)

    # 如果当前模型的AUC值是最高的，则更新最佳模型和最佳AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = best_model_temp

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(class_y_val, y_Val_pred_prob)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

    # 计算其他评价指标
    Val_y_pred = best_model_temp.predict(class_x_val)
    Val_accuracy = accuracy_score(class_y_val, Val_y_pred)
    Val_precision = precision_score(class_y_val, Val_y_pred)
    Val_cm = confusion_matrix(class_y_val, Val_y_pred)
    Val_tn, Val_fp, Val_fn, Val_tp = Val_cm.ravel()
    Val_specificity = Val_tn / (Val_tn + Val_fp)
    Val_sensitivity = recall_score(class_y_val, Val_y_pred)
    Val_npv = Val_tn / (Val_tn + Val_fn)
    Val_ppv = Val_tp / (Val_tp + Val_fp)
    Val_recall = Val_sensitivity
    Val_f1 = f1_score(class_y_val, Val_y_pred)
    Val_fpr = Val_fp / (Val_fp + Val_tn)

    # 将评价指标添加到列表中
    Val_accuracy_scores.append(Val_accuracy)
    Val_auc_scores.append(auc)
    Val_precision_scores.append(Val_precision)
    Val_specificity_scores.append(Val_specificity)
    Val_sensitivity_scores.append(Val_sensitivity)
    Val_npv_scores.append(Val_npv)
    Val_ppv_scores.append(Val_ppv)
    Val_recall_scores.append(Val_recall)
    Val_f1_scores.append(Val_f1)
    Val_fpr_scores.append(Val_fpr)

plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC For Metastasis of LN.prRLN (Validation set)')
plt.legend(loc='lower right')
plt.show()

# 打印最佳模型的名称和AUC值
print(f"最佳模型: {best_model_name} with AUC = {best_auc}")



# 创建训练集评价指标的DataFrame
Val_metrics_df = pd.DataFrame({
    'Model': list(model_param_grid.keys()),
    'Accuracy': Val_accuracy_scores,
    'AUC': Val_auc_scores,
    'Precision': Val_precision_scores,
    'Specificity': Val_specificity_scores,
    'Sensitivity': Val_sensitivity_scores,
    'Negative Predictive Value': Val_npv_scores,
    'Positive Predictive Value': Val_ppv_scores,
    'Recall': Val_recall_scores,
    'F1 Score': Val_f1_scores,
    'False Positive Rate': Val_fpr_scores
})

# 显示训练集评价指标DataFrame
print(Val_metrics_df)

# 将训练集评价指标DataFrame导出为CSV文件
Val_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/结果/2.p分析结果/3.筛选最佳模型/2.1.1验证集的评价指标.csv', index=False)

#2、内验证集的决策曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# 定义风险阈值
thresholds = np.linspace(0, 1, 100)
val_net_benefit = []



for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob = best_model_temp.predict_proba(class_x_val)[:, 1]
    else:
        y_val_pred_prob = best_model_temp.decision_function(class_x_val)

    val_model_net_benefit = []

    # 计算每个阈值下的净收益
    for threshold in thresholds:
        val_predictions = (y_val_pred_prob >= threshold).astype(int)
        val_net_benefit_value = (precision_score(class_y_val, val_predictions) - threshold * (1 - precision_score(class_y_val, val_predictions))) / (threshold + 1e-10)
        val_model_net_benefit.append(val_net_benefit_value)

    val_net_benefit.append(val_model_net_benefit)

# 转换为数组
val_net_benefit = np.array(val_net_benefit)

# 计算所有人都进行干预时的净收益
val_all_predictions = np.ones_like(class_y_val)  # 将所有预测标记为阳性（正类）
val_all_net_benefit = (precision_score(class_y_val, val_all_predictions) - thresholds * (1 - precision_score(class_y_val, val_all_predictions))) / (thresholds + 1e-10)

names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Gaussian Naive Bayes',
    'Neural Network',
    'XGBoost'
]

# 绘制DCA曲线
for i in range(val_net_benefit.shape[0]):
    plt.plot(thresholds, val_net_benefit[i], color=colors[i], label=names[i])

# 绘制"None"和"All"线
plt.plot(thresholds, np.zeros_like(thresholds), color='black', linestyle='-', label='None')
plt.plot(thresholds, val_all_net_benefit, color='gray', linestyle='--', label='All')

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
# 设置y轴的限制
plt.xlim(0, 0.6)
plt.ylim(-0.5,6)

# 设置图形属性
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis For Metastasis of LN.prRLN (Validation set)')
plt.legend(loc='upper right')

# 显示图形
plt.show()

#内验证集的校准曲线
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import ttest_ind

# 创建一个空列表来存储每个模型的校准曲线和Brier Score
val_calibration_curves = []
val_brier_scores = []

# 对每个模型进行循环
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob = best_model_temp.predict_proba(class_x_val)[:, 1]
    else:
        y_val_pred_prob = best_model_temp.decision_function(class_x_val)



    # 计算校准曲线
    val_fraction_of_positives, val_mean_predicted_value = calibration_curve(class_y_val, y_val_pred_prob, n_bins=10)
    val_calibration_curves.append((val_fraction_of_positives, val_mean_predicted_value, name, color))

    # 计算Brier分数
    val_brier_score = brier_score_loss(class_y_val, y_val_pred_prob)
    val_brier_scores.append((name, val_brier_score))

    # 打印Brier分数
    print(f'{name} - Brier Score: {val_brier_score:.3f}')

# 绘制校准曲线和Brier Score
fig, ax1 = plt.subplots(figsize=(10, 6))

for curve in val_calibration_curves:
    val_fraction_of_positives, val_mean_predicted_value, name, color = curve
    
    # 获取对应模型的Brier Score
    val_brier_score = next((score for model, score in val_brier_scores if model == name), None)
    
    # 将Brier Score赋予线颜色标注名称的后面
    if val_brier_score is not None:
        name += f' (Brier Score: {val_brier_score:.3f})'
    
    ax1.plot(val_mean_predicted_value, val_fraction_of_positives, "s-", label=name, color=color)
    
# 绘制"Perfectly calibrated"曲线
ax1.plot([0, 1], [0, 1], "k:",label="Perfectly calibrated")
    
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Calibration Curves For Metastasis of LN.prRLN (Validation set)")
plt.tight_layout()
plt.show()

#4、内验证集的精确召回曲线
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

# 初始化存储精确召回曲线和平均精确度的列表
val_precision_recall_curves = []
val_average_precision_scores = []

# 遍历每个模型
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_val_pred_prob = best_model_temp.predict_proba(class_x_val)[:, 1]
    else:
        y_val_pred_prob = best_model_temp.decision_function(class_x_val)

    # 计算精确召回曲线
    val_precision, val_recall, _ = precision_recall_curve(class_y_val, y_val_pred_prob)
    val_average_precision = average_precision_score(class_y_val, y_val_pred_prob)

    # 存储结果
    val_precision_recall_curves.append((val_precision, val_recall, f'{name} (AUPR: {val_average_precision:.3f})', color))
    val_average_precision_scores.append((f'{name} (AUPR: {val_average_precision:.3f})', val_average_precision))

    # 打印平均精确度
    print(f'{name} - Average Precision: {val_average_precision:.3f}')

# 绘制精确召回曲线
fig, ax2 = plt.subplots(figsize=(10, 6))

for curve in val_precision_recall_curves:
    val_precision, val_recall, name, color = curve
    ax2.plot(val_recall, val_precision, "-", color=color, label=name)

# 添加随机猜测曲线
plt.plot([0, 1], [class_y_val.mean(), class_y_val.mean()], linestyle='--', color='black', label='Random Guessing')

ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_ylim([0.0, 1.05])
ax2.set_xlim([0.0, 1.0])
ax2.legend(loc="lower left")
ax2.grid(True)

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Precision-Recall Curves For Metastasis of LN.prRLN (Validation set)")
plt.tight_layout()
plt.show()
