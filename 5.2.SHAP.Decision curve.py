
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
from sklearn.model_selection import KFold
import sklearn
# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')

# Select features and target variable
X = df[['Tumor.border','Hyperechoic','Location','IPLNM','TCLNM','age','size','pretracheal.LNMR',
'pretracheal.NLNM','IPLNMR','IPNLNM','TCLNMR','TCNLNM']]
y = df['LN.prRLNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0
}


# Initialize and train the model
model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Use SHAP to explain the model predictions
explainer = shap.TreeExplainer(model)


# 计算所有样本的SHAP值
shap_values = explainer.shap_values(X)
#条形图和散点图
shap.summary_plot(shap_values, X, plot_type="bar", color="green")
shap.summary_plot(shap_values, X, plot_type="dot")

#分类条形图
ndf =df.sort_values(by="LN.prRLNM")
print(ndf)
X2 = ndf[2:1472].drop(['LN.prRLNM'], axis=1)
X1 = ndf[1473:1713].drop(['LN.prRLNM'], axis=1)
shap_values = explainer.shap_values(X)

shap_values1 = explainer.shap_values(X1[1:224])
shap_values2 = explainer.shap_values(X2[1:224])
shap.summary_plot([shap_values1, shap_values2], X, plot_type="bar", class_names=["non-metastasis","metastasis"])
##因为只纳入了244个数据，所以输出的顺序不一致，需要手动更改，很重要！！！！
# 获取所有特征变量的名称
feature_names = X.columns.tolist()

# 手动指定特征变量名称的顺序
feature_names_order = ['Tumor.border', 'Hyperechoic','Location', 'age','TCNLNM','size','pretracheal.NLNM','IPLNM','TCLNMR','IPNLNM','TCLNM','pretracheal.LNMR','IPLNM',]


shap.summar
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
from sklearn.model_selection import KFold
import sklearn
# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')

# Select features and target variable
X = df[['Tumor.border','Hyperechoic','Location','IPLNM','TCLNM','age','size','pretracheal.LNMR',
'pretracheal.NLNM','IPLNMR','IPNLNM','TCLNMR','TCNLNM']]
y = df['LN.prRLNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0
}

# Initialize and train the model
model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Use SHAP to explain the model predictions
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.decision_plot(explainer.expected_value, shap_values[730, :], X.iloc[730, :], link='logit')
shap.bar_plot(shap_values[730, :], feature_names=X.columns)
shap.decision_plot(explainer.expected_value, shap_values, X, link='logit')
shap.decision_plot(explainer.expected_value, shap_values, X, feature_order="hclust")
shap.plots.force(explainer.expected_value, shap_values[730, :], X.iloc[730, :], link="logit", matplotlib=True)
###最终采用的这个版本以下所见为重要，以上为练习
import shap

# 计算所有样本的SHAP值
shap_values = explainer.shap_values(X)

# 计算每个特征的平均绝对SHAP值
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# 按照特征重要性排序
feature_order = np.argsort(mean_abs_shap)

# 绘制Decision Plot
shap.decision_plot(explainer.expected_value, shap_values, X, feature_order=feature_order)

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
from sklearn.model_selection import KFold
import sklearn
# Load data
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')

# Select features and target variable
X = df[['Tumor.border','Hyperechoic','Location','IPLNM','TCLNM','age','size','pretracheal.LNMR',
'pretracheal.NLNM','IPLNMR','IPNLNM','TCLNMR','TCNLNM']]
y = df['LN.prRLNM']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define XGBClassifier parameters
params = {'eta': 0.027825594022071243, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0
}


cv= KFold(n_splits=10, random_state=0, shuffle=True)



for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = XGBClassifier(**params).fit(X_train.iloc[train], y_train.iloc[train])
explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value

select = range(1000)
features = X.iloc[295:315]
features_display = X.loc[features.index]

shap_values = explainer.shap_values(features)

shap.decision_plot(expected_value, shap_values, features_display,feature_order=feature_order)
y_pred = (shap_values.sum(1) + expected_value) > 0
misclassified = y_pred != y[295:315]
shap.decision_plot(expected_value, shap_values, features_display, highlight=misclassified,feature_order=feature_order)
shap.decision_plot(expected_value, shap_values[misclassified], features_display[misclassified],
                   link='logit', highlight=0,feature_order=feature_order)
