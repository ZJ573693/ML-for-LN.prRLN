
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/2.T总P矫正.csv')
#1、移除方差特性
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8))) 
data_sel = sel.fit_transform(data)
data_sel
a=sel.get_support(indices=True)
data.iloc[:,a]
data_sel=data.iloc[:,a]
data_sel.info()
#2、单变量特征选择
from sklearn.feature_selection import SelectKBest, chi2
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/1.总P矫正.csv')
data_feature = data[['Age','Sex','BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Hyperechoic',
                     'Tumor.internal.vascularization','Tumor.Peripheral.blood.flow','Size','location','Location','Mulifocality','Hashimoto','ETE','prelaryngeal.LNM',
                     'pretracheal.LNM','IPLNM','TCLNM','age','size','T.staging','prelaryngeal.LNMR','prelaryngeal.NLNM','pretracheal.LNMR',
                    'pretracheal.NLNM','IPLNMR','IPNLNM','TCLNMR','TCNLNM']]
data_feature.shape
data_target=data['LN.prRLNM']
data_target.unique()
set_kit=SelectKBest(chi2,k=10)#选取k值最高的10(5)个元素
data_sel=set_kit.fit_transform(data_feature,data_target)
data_sel.shape
a=set_kit.get_support(indices=True)
data_sel=data_feature.iloc[:,a]
data_sel.info()
#3、RFE
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR #知识向量回归模型
from sklearn.model_selection import cross_val_score #知识向量回归模型

data=pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/1.总P矫正.csv')
data_feature = data[['Age','Sex','BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Hyperechoic',
                     'Tumor.internal.vascularization','Tumor.Peripheral.blood.flow','Size','location','Location','Mulifocality','Hashimoto','ETE','prelaryngeal.LNM',
                     'pretracheal.LNM','IPLNM','TCLNM','age','size','T.staging','prelaryngeal.LNMR','prelaryngeal.NLNM','pretracheal.LNMR',
                    'pretracheal.NLNM','IPLNMR','IPNLNM','TCLNMR','TCNLNM']]
data_feature.shape
estimator=SVR(kernel='linear')
sel=RFE(estimator,n_features_to_select=10,step=1) 
data_target=data['LN.prRLNM']
data_target.unique()
sel.fit(data_feature,data_target)
a=sel.get_support(indices=True)
data_sel=data_feature.iloc[:,a]
data_sel.info()
#4、RFECV
RFC_ = RandomForestClassifier()  # 随机森林
RFC_.fit(data_sel, data_target)  # 拟合模型
c = RFC_.feature_importances_  # 特征重要性
print('重要性：')
print(c)
selector = RFECV(RFC_, step=1, cv=10,min_features_to_select=10)  # 采用交叉验证cv就是10倍交叉验证，每次排除一个特征，筛选出最优特征
selector.fit(data_sel, data_target)
X_wrapper = selector.transform(data_sel)  # 最优特征
score = cross_val_score(RFC_, X_wrapper, data_target, cv=5).mean()  # 最优特征分类结果
print(score)
print('最佳数量和排序')
print(selector.support_)
print(selector.n_features_)
print(selector.ranking_)
print(selector.support_)
feature_names = data_sel.columns
selected_features = feature_names[selector.support_]
print(selected_features)
print(selector.ranking_)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(selector.ranking_)), selector.ranking_)
plt.xticks(range(len(selector.ranking_)), feature_names, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Ranking')
plt.title('Feature Importance Ranking')
plt.show()
rfecv=RFECV(estimator=RFC_,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(data,data_target)
data.iloc[:,a]
data_sel=data.iloc[:,a]
#5、L1
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression #知识向量回归模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
clf = LogisticRegression()
clf.fit(data_feature, data_target)

model = SelectFromModel(clf, prefit=True)
data_new = model.transform(data_feature)
model.get_support(indices=True)
a=model.get_support(indices=True)
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns
data_featurenew=data_features.iloc[:,a]
data_featurenew.info()
#6、基于树模型
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/1.总P矫正.csv')
data_feature = data[['Age','Sex','BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Hyperechoic',
'Tumor.internal.vascularization','Tumor.Peripheral.blood.flow','Size','location','Location','Mulifocality','Hashimoto','ETE','prelaryngeal.LNM',
'pretracheal.LNM','IPLNM','TCLNM',

'age','size','T.staging','prelaryngeal.LNMR','prelaryngeal.NLNM','pretracheal.LNMR',
'pretracheal.NLNM','IPLNMR','IPNLNM','TCLNMR','TCNLNM']]
data_target=data['LN.prRLNM']
data_target.unique()#二分类
clf = ExtraTreesClassifier()
clf.fit(data_feature, data_target)
clf.feature_importances_
model=SelectFromModel(clf,prefit=True)
x_new=model.transform(data_feature)
model.get_support(indices=True)
a=model.get_support(indices=True)
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns
data_featurenew=data_features.iloc[:,a]
data_featurenew
data_featurenew.info()
