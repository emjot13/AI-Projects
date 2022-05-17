import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.datasets import load_iris # visualization
from sklearn.neural_network import MLPClassifier # neural network
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("/home/emjot13/Desktop/studia/ai/lab8/iris.csv")

column_names = [column for column in data.columns[:4]]

df_norm = data[column_names].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


target = data[[data.columns[-1]]].replace(["setosa", "versicolor", "virginica"], [0, 1, 2])
df = pd.concat([df_norm, target], axis=1)
train, test = train_test_split(df, train_size = 0.7)
trainX = train[column_names]
trainY=train.Class
testX= test[column_names]
testY =test.Class  
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
clf.fit(trainX, trainY)
prediction = clf.predict(testX)
print(f"Accuracy: {round(metrics.accuracy_score(prediction,testY), 2) * 100}%")
