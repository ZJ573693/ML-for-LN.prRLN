shap值的可视化热图及绝对值图展示
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

model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.plots.scatter(shap_values[:,'IPLNM'])
shap.plots.scatter(shap_values[:,'IPLNMR'])
shap.plots.scatter(shap_values[:,'TCLNMR'])
shap.plots.scatter(shap_values[:,'IPNLNM'])
shap.plots.scatter(shap_values[:,'pretracheal.NLNM'])
shap.plots.scatter(shap_values[:,'TCNLNM'])
shap.plots.scatter(shap_values[:,'Tumor.border'])
shap.plots.scatter(shap_values[:, 'age'])
shap.plots.scatter(shap_values[:,'pretracheal.LNMR'])
shap.plots.scatter(shap_values[:, 'Location'])
shap.plots.scatter(shap_values[:,'size'])
shap.plots.scatter(shap_values[:,'Hyperechoic'])
shap.plots.heatmap(shap_values[:1715])

shap.plots.bar(shap_values)

shap.plots.bar(shap_values.abs.max(0))
shap.plots.beeswarm(shap_values)
shap.plots.beeswarm(shap_values.abs, color="shap_red")
clustering = shap.utils.hclust(X, y)
shap.plots.bar(shap_values, clustering=clustering)
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.8)
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=1.8)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

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

model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
feature_names = X.columns

# Create a DataFrame for the feature importances
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean Abs SHAP': mean_abs_shap_values})

# Calculate contribution percentage
shap_importance_df['Contribution (%)'] = (shap_importance_df['Mean Abs SHAP'] / shap_importance_df['Mean Abs SHAP'].sum()) * 100

# Sort the DataFrame by contribution percentage
shap_importance_df = shap_importance_df.sort_values(by='Contribution (%)', ascending=False)

# Plot the feature contributions as a bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(shap_importance_df['Feature'], shap_importance_df['Contribution (%)'], color='skyblue')
plt.xlabel('Contribution (%)')
plt.ylabel('Feature')
plt.title('Metastasis of LN.prRLN Contribution Percentage based on SHAP values')
plt.gca().invert_yaxis()

# Add percentage values at the end of each bar
for bar, value in zip(bars, shap_importance_df['Contribution (%)']):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{value:.3f}%', va='center')

plt.show()

# Display the DataFrame
print(shap_importance_df)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

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


model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
feature_names = X.columns

# Create a DataFrame for the feature importances
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean Abs SHAP': mean_abs_shap_values})

# Sort the DataFrame by SHAP value and select top 10 features
shap_importance_df = shap_importance_df.sort_values(by='Mean Abs SHAP', ascending=False).head(10)

# Normalize the contribution to sum to 100%
shap_importance_df['Contribution (%)'] = (shap_importance_df['Mean Abs SHAP'] / shap_importance_df['Mean Abs SHAP'].sum()) * 100

# Plot the top 10 feature contributions as a bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(shap_importance_df['Feature'], shap_importance_df['Contribution (%)'], color='skyblue')
plt.xlabel('Contribution (%)')
plt.ylabel('Feature')
plt.title('Top 10 Metastasis of LN.prRLN Feature Contribution Percentage based on SHAP values')
plt.gca().invert_yaxis()

# Add percentage values at the end of each bar
for bar, value in zip(bars, shap_importance_df['Contribution (%)']):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{value:.3f}%', va='center')

plt.show()

# Display the DataFrame
print(shap_importance_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

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


model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
feature_names = X.columns

# Create a DataFrame for the feature importances
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean Abs SHAP': mean_abs_shap_values})

# Sort the DataFrame by SHAP value and select top 5 features
shap_importance_df = shap_importance_df.sort_values(by='Mean Abs SHAP', ascending=False).head(10)

# Normalize the contribution to sum to 100%
shap_importance_df['Contribution (%)'] = (shap_importance_df['Mean Abs SHAP'] / shap_importance_df['Mean Abs SHAP'].sum()) * 100

# Plot the top 5 feature contributions as a pie chart
plt.figure(figsize=(10, 7))
plt.pie(shap_importance_df['Contribution (%)'], labels=shap_importance_df['Feature'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Top 10 Metastasis of LN.prRLN Feature Contribution Percentage based on SHAP values')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Display the DataFrame
print(shap_importance_df)
