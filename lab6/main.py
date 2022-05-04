from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math 
import enchant

pd.set_option('display.max_rows', None, 'display.max_columns', None)
df = pd.read_csv("/home/emjot13/Desktop/studia/ai/lab6/iris_with_errors.csv")
pd.options.mode.chained_assignment = None

missing_values_count = 0
all_values_count = 0
bad_range_count = 0

indexes_to_correct = []
columns_averages = {}


for index in range(len(df.columns) - 1):
    column_sum = 0
    # print(len(df[df.columns[index]]), df[df.columns[index]])
    for ind, data in enumerate(df[df.columns[index]]):
        all_values_count += 1
        try:
             data = float(data)
             if math.isnan(data):
                missing_values_count += 1
             elif data < 0 or data > 15:
                 indexes_to_correct.append([index, ind])
                 bad_range_count += 1
             else:
                 column_sum += data
        except ValueError:
            missing_values_count += 1
    columns_averages[df.columns[index]] = column_sum / len(df[df.columns[index]])    



#there are 5 missing values and 5 incorrect values which corresponds to 1.67% of all the data


for k in indexes_to_correct:
    column = df.columns[k[0]]
    df[column][k[1]] = columns_averages[column]          # changing incorrect valeus to the average of their column
     

correct_varieties = ["Setosa", "Virginica", "Versicolor"]


for ind, variety in enumerate(df["variety"]):
    levensthein_distances = []
    for name in correct_varieties:
        levensthein_distances.append((name, enchant.utils.levenshtein(variety, name)))
    
    levensthein_distances.sort(key=lambda tup: tup[1])

    if levensthein_distances[0][1] > 4:     # if the levensthein distance is greater than 4, the input is marked as unrecognized
        df["variety"][ind] = "NA"
    else:
        df["variety"][ind] = levensthein_distances[0][0] # we set the name of variety as the one with the closest levensthein's distance


##########################################################################################################################################



df = pd.read_csv("/home/emjot13/Desktop/studia/ai/lab6/iris.data", names=['sepal length','sepal width','petal length','petal width','target'])


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features]
y = df.loc[:, ['target']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)



print(pca.explained_variance_ratio_)

# można zauważyć, że suma wariancji głównych składowych wynosi 95%, jednak żadna z obu nie ma ponad 80%, zatem minimalną ilością kolumn jakie musimy zostawić, żeby
# zachować 80% wariancji to 2 kolumny




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# wykres w 2D

iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 0].mean(),
        X[y == label, 1].mean() + 1.5,
        X[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

### wykres w 3D



