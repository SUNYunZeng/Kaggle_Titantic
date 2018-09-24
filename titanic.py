# _*_ coding:utf-8 _*_
__author__ = 'Syz'

# 正则表达式模块
import re
# 导入线性回归模型跟逻辑回归模型
from sklearn.linear_model import LinearRegression, LogisticRegression

# 导入随机森林模型
from sklearn.ensemble import RandomForestClassifier
# 导入交叉验证模型
from sklearn.model_selection import KFold, cross_val_score

# 各特征属性的重要程度
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np
import pandas
from pandas import Series, DataFrame
# 导入图表函数
import matplotlib.pyplot as plt
from pylab import *
# 图表汉字正常显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 图表负值正常显示
matplotlib.rcParams['axes.unicode_minus'] = False

# 显示区域的宽度调整
pandas.set_option('display.width', 300)
# 显示最大行数设置
pandas.set_option('display.max_columns', None)
# 显示最大列数设置
pandas.set_option('display.max_rows', None)

# 读取泰坦尼克号数据
titanic = pandas.read_csv(r'C:\Users\12906\Desktop\Kaggle_Titanic-master\train.csv')

# 打印出泰坦尼克号部分数据，默认前五个
# print(titanic.head(3))
# 对于表中的数据进行统计
# print(titanic.describe())
# %% 数据预处理

# 由于年龄中有空值，需要先用平均值对年龄的缺失值进行填充，因为矩阵运算只能是数值型，不能是字符串
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
# 同理，由于Embarked（登船地点）里面也有空值，所以也需要用出现最多的类型对它进行一个填充
titanic['Embarked'] = titanic['Embarked'].fillna('S')

# 对于性别中的male与female，用0和1来表示。首先看性别是否只有两个值
# 对于登船地点的三个值S C Q，也用0 1 2分别表示
# print(titanic['Sex'].unique())
# print(titanic['Embarked'].unique())
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

# 加上其余的属性特性
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# 姓名的长度
titanic["NameLenght"] = titanic["Name"].apply(lambda x: len(x))


# 定义提取姓名中Mr以及Mrs等属性
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""


titles = titanic["Name"].apply(get_title)
# 对于姓名中的一些称呼赋予不同的数值
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Major': 7, 'Mlle': 8, 'Col': 9,
                 'Capt': 10, 'Ms': 11, 'Don': 12, 'Jonkheer': 13, 'Countess': '14', 'Lady': 15, 'Sir': 16, 'Mme': 17}
for k,v in title_mapping.items():
	titles[titles == k] = v
titanic['Titles'] = titles
# print(titanic.head())
# print(pandas.value_counts(titles))
presictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles", "FamilySize", "NameLenght"]
#%%% 查看各项属性的准确率
"""
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[presictors], titanic["Survived"])
# 获取每个数据的重要值
scores = -np.log10(selector.pvalues_)

# 画图表示，看看哪一些属性对结果影响较大，即重要值高
plt.bar(range(len(presictors)), scores)
plt.xticks(range(len(presictors)), presictors, rotation='vertical')

plt.show()
"""



# 选择其中比较重要的属性作为训练属性
# presictors = ["Pclass", "Sex", "Fare", "Titles",  "NameLenght"]
# %% 线性回归模型
"""
# 参与训练的属性
# presictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles", "FamilySize", "NameLenght"]
# 导入线性回归模型
alg = LinearRegression()
# 在训练集上进行三次交叉验证
# kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
kf = KFold(n_splits=2)
predictions = []
for train, test in kf.split(titanic):
	# 从train中取出分割后的训练数据
	train_predictors = titanic[presictors].iloc[train, :]
	# 取出存活数量作为训练目标
	train_target = titanic["Survived"].iloc[train]
	# 使用训练样本跟训练目标训练回归函数
	alg.fit(train_predictors, train_target)
	# 我们现在可以在测试集上做预测
	test_predictions = alg.predict(titanic[presictors].iloc[test, :])
	predictions.append(test_predictions)

# 检查回归的效果
predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions == titanic["Survived"])/len(predictions)
print(accuracy)
"""

# %% 逻辑回归模型
"""
presictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LogisticRegression(random_state=1)
scores = cross_val_score(alg, titanic[presictors], titanic["Survived"], cv=3)
print(scores.mean())
"""


# %% 随机森林模型
"""
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
# presictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles", "FamilySize", "NameLenght"]
kf = KFold(n_splits=3)
scores = cross_val_score(alg, titanic[presictors], titanic["Survived"], cv=kf.split(titanic))
print(scores.mean())
"""
#%% 模型融合

algorithms = [
	[RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2), ["Pclass", "Sex",
	                                 "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles", "FamilySize", "NameLenght"]],
	[LogisticRegression(random_state=1),  ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles",
	                                       "FamilySize", "NameLenght"]]
]
# 交叉验证
kf = KFold(n_splits=3)
predictions = []
for train, test in kf.split(titanic):
	train_target = titanic["Survived"].iloc[train]
	full_test_predictions = []
	# 对于每一个测试集都做出预测
	for alg, predictors in algorithms:
		alg.fit(titanic[predictors].iloc[train, :], train_target)
		test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
		full_test_predictions.append(test_predictions)
	test_predictions = (full_test_predictions[0]*3 + full_test_predictions[1])/4
	test_predictions[test_predictions <= .5] = 0
	test_predictions[test_predictions > .5] = 1
	predictions.append(test_predictions)

# 把所有的预测结果放到集合当中
predictions = np.concatenate(predictions, axis=0)

# 计算与训练数据真值比较的精度
accuracy = sum(predictions == titanic["Survived"])/len(predictions)
print(accuracy)


# %% 数据分析
# %%
"""
# 查看各等级乘客等级的获救情况
fig = plt.figure()
# 设置图表颜色的alpha参数
fig.set(alpha=0.2)

Suvived_0 = titanic.Pclass[titanic.Survived == 0].value_counts()
Suvived_1 = titanic.Pclass[titanic.Survived == 1].value_counts()
df = pandas.DataFrame({u"获救": Suvived_1, u"未获救": Suvived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
plt.show()
"""
# %%性别对获救结果的影像
"""
# 按性别分组
fig = plt.figure()
fig.set(alpha=0.2)

Survived_m = titanic.Survived[titanic.Sex == 0].value_counts()
Survived_f = titanic.Survived[titanic.Sex == 1].value_counts()
df = pandas.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u'不同性别获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')
plt.show()
"""


#%% 不同船舱等级的人数统计
"""
fig = plt.figure()
fig.set(alpha=0.2)
titanic.Pclass.value_counts().plot(kind='bar')
plt.ylabel(u'人数')
plt.title(u'乘员等级分布')
plt.show()
"""
# %% 年龄与获救之间的关系
"""
fig = plt.figure()
fig.set(alpha=0.2)
plt.scatter(titanic.Survived, titanic.Age)
plt.ylabel(u'年龄')
plt.grid(b=True, which='major', axis='y')
plt.title(u'不同年龄的获救情况（1为获救）')
plt.show()
"""
# %% 不同港口登录乘客获救情况
"""
fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = titanic.Embarked[titanic.Survived == 0].value_counts()
Survived_1 = titanic.Embarked[titanic.Survived == 1].value_counts()
df = pandas.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.xlabel(u'登录港口')
plt.ylabel(u'人数')
plt.title(u'不同港口登录乘客获救情况')

plt.show()
"""
# %% 有无家人获救情况
"""
fig = plt.figure()
fig.set(alpha=0.2)

Survived_cabin = titanic.Survived[pandas.notnull(titanic.Cabin)].value_counts()
Survived_nocabin = titanic.Survived[pandas.isnull(titanic.Cabin)].value_counts()
df = pandas.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
df.plot(kind='bar',stacked=True)
plt.title(u'有无Cabin属性的获救情况')
plt.xlabel(u'Cabin有无')
plt.ylabel(u'人数')
plt.show()
"""
#%% sigmoid函数绘制
"""
def Sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

x = np.arange(-10, 10, 0.1)
h = Sigmoid(x)  # Sigmoid函数
plt.plot(x, h)
plt.axvline(0.0, color='k')  # 坐标轴上加一条竖直的线（0位置）
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])  # y轴标度
plt.ylim(-0.1, 1.1)  # y轴范围
plt.show()
"""
