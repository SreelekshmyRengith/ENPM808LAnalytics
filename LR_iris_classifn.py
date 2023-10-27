import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
df=pd.read_csv(r'C:\Fall 23\ENPM 808L\Week2 report\IrisNew.csv')
new_target = LabelEncoder()
df['target'] = new_target.fit_transform(df['Class'])
inputs = df.drop(['Class','target'],axis="columns")
inputs.columns = ['Sepal Length(cm)', 'Sepal Width(cm)', 'Petal Length(cm)', 'Petal Width(cm)']
target = df['target']
#splitting the data set into tarining and testing data
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(inputs,target,test_size=0.25,random_state=2)
lorg=LogisticRegression(random_state=0,max_iter=1000)
lorg.fit(x_train,y_train)
y_pred=lorg.predict(x_test)
#constructing a confusion matrix- to cal the accuracy of the model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix for LR:\n",cm)
#to calculate the accuracy
from sklearn.metrics import accuracy_score
print("Accuracy of Logistic Regression model: ",accuracy_score(y_test,y_pred))