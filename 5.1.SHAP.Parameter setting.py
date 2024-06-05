pip install xgboost
pip install shap
#5.1.1
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'n_estimators': np.linspace(0, 3000, 10, dtype=int)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.2

other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'max_depth': np.linspace(1, 10, 15, dtype=int)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.3
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'min_child_weight': np.linspace(0, 10, 20, dtype=int)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.4
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'gamma': np.linspace(0.01, 1, 10)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.5
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'subsample': np.linspace(0, 1, 10)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.6
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'colsample_bytree': np.linspace(0, 1, 11)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.7
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'colsample_bylevel': np.linspace(0, 1, 11)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.8
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
other_params = {'eta': 0.025, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0,}
cv_params = {'reg_lambda': np.linspace(0, 1, 11)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.9
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
other_params = {'eta': 0.027825594022071243, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0}
cv_params = {'reg_alpha': np.linspace(0, 10, 10)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
#5.1.10
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/1.矫正后_副本.csv')
y = df['LN.prRLNM']
X = df.drop(['LN.prRLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
other_params = {'eta': 0.027825594022071243, 'n_estimators': 333, 'gamma': 0.89, 'max_depth': 3, 'min_child_weight': 0,
                'colsample_bytree': 0.3, 'colsample_bylevel': 0.0, 'subsample': 0.11111111111, 'reg_lambda': 0.2, 'reg_alpha': 0}
cv_params = {'eta': np.logspace(-2, 0, 10)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
