import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=275309)


train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

# print(train_inputs)
# print(train_classes)
# print(test_inputs)
# print(test_classes)


def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return "setosa"
    elif pl <= 5:
        return "virginica"
    else:
        return "versicolor"


def classify_iris_v2(sl, sw, pl, pw):
    if 4.4 <= sl <= 5.8 and sw >= 2.9 and pl >= 1 and pw >= 0.1:
        return "setosa"
    elif sl >= 5.6 and sw >= 2.5 and pl >= 4.5 and pw >= 1.4:
        return "virginica"
    else:
        return "versicolor"



good_predictions = 0
len = test_set.shape[0]


for i in range(len):
    if classify_iris_v2(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_classes[i]:
        good_predictions = good_predictions + 1
#
#

# train_set = list(train_set)
# for i in range(len(train_set)):
#     train_set[i] = list(train_set[i])
#
# print(type(train_set), train_set)

# print(good_predictions)
# print(good_predictions/len*100, "%")




