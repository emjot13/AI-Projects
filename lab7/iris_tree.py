import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/emjot13/Desktop/studia/ai/lab7/iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=2)

train_inputs = train_set[:, :4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, :4]
test_classes = test_set[:, 4]



def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return "setosa"
    elif pl <= 5:
        return "virginica"
    else:
        return "versicolor"


def classify_iris_v2(sl, sw, pl, pw):
    if 4.3 <= sl <= 5.7 and  2.9 <= sw <= 4.2 and 1.1 <= pl <= 1.9:
        return "setosa"
    elif 4.9 <= sl <= 7.9 and 2.2 <= sw <= 3.8 and 4.5 <= pl <= 6.9:
        return "virginica"
    else:
        return "versicolor"





good_predictions = 0
length = test_set.shape[0]


for i in range(length):
    if classify_iris_v2(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_classes[i]:
        good_predictions = good_predictions + 1





print(good_predictions)
print(good_predictions/length*100, "%")


from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt



dtc = tree.DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes))   # about 96% accuracy


# train_inputs = train_set[:, :]
# train_classes = train_set[:, :]
# test_inputs = test_set[:, :]
# test_classes = test_set[:, :]

# code to make a graph

iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)
# plt.show()
plt.savefig('iris_tree.png', dpi=500)


from sklearn.metrics import confusion_matrix







