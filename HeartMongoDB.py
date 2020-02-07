# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pymongo as pm


#connecting to MongoDB
connection = pm.MongoClient()
heart = connection.heart
heart_collection = heart.heart_data

cursor = heart_collection.find()
data =  pd.DataFrame(list(cursor))  

data['target'] = data['target\n'].str.replace('\n', '')
df = data.drop(columns=["_id","target\n"])
df.rename(columns = {'ï»¿age':'age' }, inplace = True)
df.head() ##shows first 5 rows in dataset


y = df.target.values
x_data = df.drop(['target'], axis = 1)
# Normalize
x_data = x_data.astype(float)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

#Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


accuracies = {}
####Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
acc = lr.score(x_test.T,y_test.T)*100

accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.2f}%".format(acc))


# try to find best k value
from sklearn.neighbors import KNeighborsClassifier
scoreList = []
for i in range(1,10):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
plt.plot(range(1,10), scoreList)
plt.xticks(np.arange(1,10,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

# KNN Model
knn = KNeighborsClassifier(n_neighbors = 2)  # elbow accuracy is good for k=3
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)
print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
acc = max(scoreList)*100
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.2f}%".format(acc))

###SVC
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train.T, y_train.T)

acc = svm.score(x_test.T,y_test.T)*100
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))


####DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)

acc = dtc.score(x_test.T, y_test.T)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)

acc = rf.score(x_test.T,y_test.T)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))

##thalach and trestbps vs target
g = sns.catplot(x="thalach", y="trestbps", hue="target", kind="swarm", data=df)

##Cp vs target
temp = (df.groupby(['target']))['cp'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "cp", data = temp).set_title("Chest Pain vs Heart Disease")

###corr matrix
df.dtypes

plt.figure(figsize=(10,10))
palette = sns.diverging_palette(20, 220, n=256)

df["ca"]=pd.to_numeric(df["ca"])
df["chol"]=pd.to_numeric(df["chol"])
df["exang"]=pd.to_numeric(df["exang"])
df["fbs"]=pd.to_numeric(df["fbs"])
df["oldpeak"]=pd.to_numeric(df["oldpeak"])
df["restecg"]=pd.to_numeric(df["restecg"])
df["sex"]=pd.to_numeric(df["sex"])
df["thalach"]=pd.to_numeric(df["thalach"])
df["trestbps"]=pd.to_numeric(df["trestbps"])
df["age"]=pd.to_numeric(df["age"])
df["target"]=pd.to_numeric(df["target"])

cor=df.corr(method='pearson')
sns.heatmap(cor, annot=True,cmap=palette, vmax=.3).set(ylim=(10, 0))
plt.title("Correlation Matrix",size=10, weight='bold')

#People effected
plt.figure(figsize=(5, 5))
target_count = [len(df[df['target'] == 0]),len(df[df['target'] == 1])]
labels = ['No Disease', 'Disease']
colors = ['green', 'blue']
explode = (0.05, 0.1)
plt.pie(target_count, explode=explode, labels=labels, colors=colors,autopct='%4.2f%%',shadow=True, startangle=45)
plt.title('Target Percent')
plt.show()  
