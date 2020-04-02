import numpy as np
import pandas as pd

'''trainData=pd.read_csv('train.csv',index_col=0)'''
#add the dataset
trainData = pd.read_csv('train.csv')
trainData.head()
trainData=trainData.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

trainData.count()

#missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imp=imp.fit(trainData.iloc[:,3:4])
trainData.iloc[:,3:4] = imp.transform(trainData.iloc[:,3:4])

trainData.dropna(axis=0,inplace=True)

y = trainData.Survived
X = trainData.drop(['Survived'],axis=1)

#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
X.iloc[:,1] = le.fit_transform(X.iloc[:,1])
a=[]
for i in X.iloc[:,6]:
    a.append(str(i))
X.iloc[:,6]=a
X.iloc[:,6] = le.fit_transform(X.iloc[:,6])

ohe = OneHotEncoder(categorical_features = [0,6])
X = ohe.fit_transform(X).toarray()
X = np.delete(X,[2,5],axis = 1)
#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)

#data visualization
import matplotlib.pyplot as plt
import seaborn as sb
plt.plot(xTrain[:,10],yTrain) #plotting lines through (x,y)
plt.scatter(xTrain[:,10],yTrain,c='red') #scatter plot
plt.hist(trainData['Sex'],bins=5) #frequency of ranges of values Histogram
sb.set(style='darkgrid')
sb.regplot(x = xTrain[:,10], y = yTrain, marker='.') #scatterplot with regression line
sb.lmplot(x = 'Fare', y = 'Pclass', data = trainData, hue = 'Sex', legend=True) #advanced scatter plot
sb.distplot(trainData['Fare'],bins=3,kde=False) #histogram
sb.countplot(x='Sex',data=trainData, hue = 'Survived') #frequency of each type of categorical value Barplot
sb.boxplot(x=trainData['Sex'], y=trainData['Fare']) #box and whiskers plot (x = is optional)
sb.boxplot(x='Sex', y='Fare', data=trainData, hue='Survived') #box and whiskers plot (x,hue = is optional)
sb.heatmap(trainData.isnull(),yticklabels=False,cbar=False,cmap='viridis') #heatmap to check missing values

f,(ax_box, ax_hist) = plt.subplots(2, gridspec_kw = {'height_ratios': (0.15 , 0.85)})
sb.boxplot(x=trainData['Fare'],ax=ax_box)
sb.distplot(trainData['Fare'],ax=ax_hist, kde=False) #box and whiskers plot & histogram

#classification model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xTrain , yTrain)
pred = lr.predict(xTest)

from sklearn.metrics import classification_report
print(classification_report(yTest,pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(yTest, pred))
