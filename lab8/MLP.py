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

accuracy = {}


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2), random_state=1, max_iter = 1000)
clf.fit(trainX, trainY)
clf_1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3), random_state=1, max_iter = 1000)
clf_1.fit(trainX, trainY)
clf_2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1, max_iter = 1000)
clf_2.fit(trainX, trainY)
prediction = clf.predict(testX)
prediction_1 = clf_1.predict(testX)
prediction_2 = clf_2.predict(testX)

accuracy["one layer, 2 neurons"], accuracy["one layer, 3 neurons"], accuracy["2 layers, 3 neurons each"] = prediction, prediction_1, prediction_2

for k, v in accuracy.items(): 
    score =  "{0:.2f}".format(metrics.accuracy_score(v, testY) * 100)
    print(f"Accuracy of {k}: {score}%")


# e) -> po przeprowadzeniu wszystkich koniecznych operacji MLP zaokrągla otrzymaną wartość do nabliższej odpowiadającej mu wartości neuronów warstwy wyjściowej




# Accuracy of one layer, 2 neurons: 33.33%
# Accuracy of one layer, 3 neurons: 97.78%
# Accuracy of 2 layers, 3 neurons each: 100.00%