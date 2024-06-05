#指定pandas为pd方便后续数据的读取
import pandas as pd
#3、数据规范化
#指定pandas为pd方便后续数据的读取
import pandas as pd
#1、分类变量的编码
data.head
#1.1找出分类型的变量
data_category = data.select_dtypes(include=['object'])
#1.2查看
data_category
data_Number=data.select_dtypes(exclude=['object'])
data_Number
data_Number.columns.values
#1.6整合编码
from sklearn.preprocessing import OrdinalEncoder

# 创建并拟合编码器
encoder = OrdinalEncoder()
encoder.fit(data_category)

# 将分类变量进行编码转换
data_category_enc = pd.DataFrame(encoder.transform(data_category), columns=data_category.columns)
#1.7加载表头
data_category_enc
#1.10将表格拼回去
data_enc=pd.concat([data_category_enc,data_Number],axis=1)
#axis=0为纵向拼接 axis=1是按列拼接
#1.11编码完成
data_enc
#1.12将新的编码后的数据输入文件夹中
data_enc.to_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/结果/2.T总编码后.csv')
#2.1分类变量无法使用均质填补，因此使用众数填补（即出现频率最高的数进行填补）
#加载sklearn 的函数
from sklearn.impute import SimpleImputer
#2.2众数填补的缺失值
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# 创建并拟合填充器
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_encImpute = pd.DataFrame(imp.fit_transform(data_enc))

# 设置列名
data_encImpute.columns = data_enc.columns
#2.3整合
data_encImpute
#2.4看之前的变量名字
data_encImpute['prelaryngeal.LNM'].value_counts()
#2.5将插补后的数据保存下来
data_encImpute.to_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/结果/2.T总-仅仅用于r的dca2.csv')
#3数值数据校准和归一化
data_scale=data_encImpute
#3.1
target=data_encImpute['LN.prRLNM'].astype(int)
##3.2
target.value_counts()
from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(data_scale))
data_scaled.columns=data_scale.columns
data_scaled
#将矫正后的数据保存下来
data_scaled.to_csv('/Users/zj/Desktop/4.机器学习/三、喉返后/0.数据/2.T总P矫正.csv')
