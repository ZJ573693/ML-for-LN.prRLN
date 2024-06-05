#外验证的roc曲线
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
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后.csv')

# 导入数据
data_feature = data[['prelaryngeal.NLNM','prelaryngeal.LNMR',
                     'Aspect.ratio','Internal.echo.homogeneous','Ingredients','T.stage','IPNLNM',


]]
data_target = data['LN.prRLNM']

# 数值变量标准化
data_featureNum = data[['prelaryngeal.NLNM','prelaryngeal.LNMR','T.stage','IPNLNM',]]
scaler = MinMaxScaler()
data_featureNum = scaler.fit_transform(data_featureNum)

data_featureCata = data[['Aspect.ratio','Internal.echo.homogeneous','Ingredients',]]
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
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='auc'), {'n_estimators': [50, 100, 200], 
                                                                            'max_depth': [5, 7, 10], 'learning_rate': [0.01,0.05, 0.1], 'subsample': [0.7, 0.8, 0.9], 'gamma': [0, 0.1, 0.5]})
}

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan','orange']

# 初始化最佳AUC和最佳模型
best_auc = 0
best_model_name = ''
best_model = None


# Load the external validation set
external_data = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/2.T总P矫正.csv')

#导入数据
external_feature = external_data[['prelaryngeal.NLNM','prelaryngeal.LNMR','IPNLNM',
                                  'Aspect.ratio','Internal.echo.homogeneous','Ingredients','T.stage',



]]
external_target=external_data['LN.prRLNM']
external_target.unique()#二分类


# Preprocess the external validation set
external_featureCata = external_data[['Aspect.ratio','Internal.echo.homogeneous','Ingredients',]]

external_featureNum = external_data[['prelaryngeal.NLNM','prelaryngeal.LNMR','T.stage','IPNLNM',

]]

external_featureNum = scaler.transform(external_featureNum)
external_feature = np.hstack((external_featureCata, external_featureNum))
external_target = external_data['LN.prRLNM']



# Lists for evaluation metrics
Ext_accuracy_scores = []
Ext_auc_scores = []
Ext_precision_scores = []
Ext_specificity_scores = []
Ext_sensitivity_scores = []
Ext_npv_scores = []
Ext_ppv_scores = []
Ext_recall_scores = []
Ext_f1_scores = []
Ext_fpr_scores = []

# Fit models and plot ROC curve for external validation set
plt.figure(figsize=(8, 6))

for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # Predict probabilities on external validation set
    if hasattr(best_model_temp, 'predict_proba'):
        y_test_pred_prob = best_model_temp.predict_proba(external_feature)[:, 1]
    else:
        y_test_pred_prob = best_model_temp.decision_function(external_feature)

    # Calculate AUC
    auc = roc_auc_score(external_target, y_test_pred_prob)

    # Update best model if current model has higher AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = best_model_temp

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(external_target, y_test_pred_prob)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

    # Calculate other evaluation metrics
    y_test_pred = best_model_temp.predict(external_feature)
    Ext_accuracy = accuracy_score(external_target, y_test_pred)
    Ext_precision = precision_score(external_target, y_test_pred)
    Ext_cm = confusion_matrix(external_target, y_test_pred)
    Ext_tn, Ext_fp, Ext_fn, Ext_tp = Ext_cm.ravel()
    Ext_specificity = Ext_tn / (Ext_tn + Ext_fp)
    Ext_sensitivity = recall_score(external_target, y_test_pred)
    Ext_npv = Ext_tn / (Ext_tn + Ext_fn)
    Ext_ppv = Ext_tp / (Ext_tp + Ext_fp)
    Ext_recall = Ext_sensitivity
    Ext_f1 = f1_score(external_target, y_test_pred)
    Ext_fpr = Ext_fp / (Ext_fp + Ext_tn)

    # Append evaluation metrics to lists
    Ext_accuracy_scores.append(Ext_accuracy)
    Ext_auc_scores.append(auc)
    Ext_precision_scores.append(Ext_precision)
    Ext_specificity_scores.append(Ext_specificity)
    Ext_sensitivity_scores.append(Ext_sensitivity)
    Ext_npv_scores.append(Ext_npv)
    Ext_ppv_scores.append(Ext_ppv)
    Ext_recall_scores.append(Ext_recall)
    Ext_f1_scores.append(Ext_f1)
    Ext_fpr_scores.append(Ext_fpr)

plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.grid(color='lightgray', linestyle='-', linewidth=1)  # Background grid lines
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Metastasis of LN.prRLN (Test Set)')
plt.legend(loc='lower right')
plt.show()

# Print best model name and AUC
print(f"Best model: {best_model_name} with AUC = {best_auc}")

# Create DataFrame for external validation metrics
Ext_metrics_df = pd.DataFrame({
    'Model': list(model_param_grid.keys()),
    'Accuracy': Ext_accuracy_scores,
    'AUC': Ext_auc_scores,
    'Precision': Ext_precision_scores,
    'Specificity': Ext_specificity_scores,
    'Sensitivity': Ext_sensitivity_scores,
    'Negative Predictive Value': Ext_npv_scores,
    'Positive Predictive Value': Ext_ppv_scores,
    'Recall': Ext_recall_scores,
    'F1 Score': Ext_f1_scores,
    'False Positive Rate': Ext_fpr_scores
})

# Display DataFrame
print(Ext_metrics_df)

# Export metrics to CSV
Ext_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/结果/2.p分析结果/3.筛选最佳模型/3.1.1测试集的评价指标.csv', index=False)

#2、外验证集的决策曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# 定义风险阈值
thresholds = np.linspace(0, 1, 100)
Ext_net_benefit = []



for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_Ext_pred_prob = best_model_temp.predict_proba(external_feature)[:, 1]
    else:
        y_Ext_pred_prob = best_model_temp.decision_function(external_feature)

    Ext_model_net_benefit = []

    # 计算每个阈值下的净收益
    for threshold in thresholds:
        Ext_predictions = (y_Ext_pred_prob >= threshold).astype(int)
        Ext_net_benefit_value = (precision_score(external_target, Ext_predictions) - threshold * (1 - precision_score(external_target, Ext_predictions))) / (threshold + 1e-10)
        Ext_model_net_benefit.append(Ext_net_benefit_value)

    Ext_net_benefit.append(Ext_model_net_benefit)

# 转换为数组
Ext_net_benefit = np.array(Ext_net_benefit)

# 计算所有人都进行干预时的净收益
Ext_all_predictions = np.ones_like(external_target)  # 将所有预测标记为阳性（正类）
Ext_all_net_benefit = (precision_score(external_target, Ext_all_predictions) - thresholds * (1 - precision_score(external_target, Ext_all_predictions))) / (thresholds + 1e-10)

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
for i in range(Ext_net_benefit.shape[0]):
    plt.plot(thresholds, Ext_net_benefit[i], color=colors[i], label=names[i])

# 绘制"None"和"All"线
plt.plot(thresholds, np.zeros_like(thresholds), color='black', linestyle='-', label='None')
plt.plot(thresholds, Ext_all_net_benefit, color='gray', linestyle='--', label='All')

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
# 设置y轴的限制
plt.xlim(0, 0.6)
plt.ylim(-0.5,6)

# 设置图形属性
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis For Metastasis of LN.prRLN (Test set)')
plt.legend(loc='upper right')

# 显示图形
plt.show()
#3、 外验证集的校准曲线
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import ttest_ind

# 创建一个空列表来存储每个模型的校准曲线和Brier Score
Ext_calibration_curves = []
Ext_brier_scores = []

# 对每个模型进行循环
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_Ext_pred_prob = best_model_temp.predict_proba(external_feature)[:, 1]
    else:
        y_Ext_pred_prob = best_model_temp.decision_function(external_feature)



    # 计算校准曲线
    Ext_fraction_of_positives, Ext_mean_predicted_value = calibration_curve(external_target, y_Ext_pred_prob, n_bins=10)
    Ext_calibration_curves.append((Ext_fraction_of_positives, Ext_mean_predicted_value, name, color))

    # 计算Brier分数
    Ext_brier_score = brier_score_loss(external_target, y_Ext_pred_prob)
    Ext_brier_scores.append((name, Ext_brier_score))

    # 打印Brier分数
    print(f'{name} - Brier Score: {Ext_brier_score:.3f}')

# 绘制校准曲线和Brier Score
fig, ax1 = plt.subplots(figsize=(10, 6))

for curve in Ext_calibration_curves:
    Ext_fraction_of_positives, Ext_mean_predicted_value, name, color = curve
    
    # 获取对应模型的Brier Score
    Ext_brier_score = next((score for model, score in Ext_brier_scores if model == name), None)
    
    # 将Brier Score赋予线颜色标注名称的后面
    if Ext_brier_score is not None:
        name += f' (Brier Score: {Ext_brier_score:.3f})'
    
    ax1.plot(Ext_mean_predicted_value, Ext_fraction_of_positives, "s-", label=name, color=color)
    
# 绘制"Perfectly calibrated"曲线
ax1.plot([0, 1], [0, 1], "k:",label="Perfectly calibrated")
    
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Calibration Curves For Metastasis of LN.prRLN (Test set)")
plt.tight_layout()
plt.show()
#4、外验证集的精确召回曲线
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

# 初始化存储精确召回曲线和平均精确度的列表
Ext_precision_recall_curves = []
Ext_average_precision_scores = []

# 遍历每个模型
for (name, (model, param_grid)), color in zip(model_param_grid.items(), colors):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(class_x_tra, class_y_tra)
    best_model_temp = grid_search.best_estimator_

    # 计算训练集上的预测概率
    if hasattr(best_model_temp, 'predict_proba'):
        y_Ext_pred_prob = best_model_temp.predict_proba(external_feature)[:, 1]
    else:
        y_Ext_pred_prob = best_model_temp.decision_function(external_feature)

    # 计算精确召回曲线
    Ext_precision, Ext_recall, _ = precision_recall_curve(external_target, y_Ext_pred_prob)
    Ext_average_precision = average_precision_score(external_target, y_Ext_pred_prob)

    # 存储结果
    Ext_precision_recall_curves.append((Ext_precision, Ext_recall, f'{name} (AUPR: {Ext_average_precision:.3f})', color))
    Ext_average_precision_scores.append((f'{name} (AUPR: {Ext_average_precision:.3f})', Ext_average_precision))

    # 打印平均精确度
    print(f'{name} - Average Precision: {Ext_average_precision:.3f}')

# 绘制精确召回曲线
fig, ax2 = plt.subplots(figsize=(10, 6))

for curve in Ext_precision_recall_curves:
    Ext_precision, Ext_recall, name, color = curve
    ax2.plot(Ext_recall, Ext_precision, "-", color=color, label=name)

# 添加随机猜测曲线
plt.plot([0, 1], [external_target.mean(), external_target.mean()], linestyle='--', color='black', label='Random Guessing')

ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_ylim([0.0, 1.05])
ax2.set_xlim([0.0, 1.0])
ax2.legend(loc="lower left")
ax2.grid(True)

# 设置背景灰色格子线
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.title("Precision-Recall Curves For Metastasis of LN.prRLN (Test set)")
plt.tight_layout()
plt.show()
