
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from io import StringIO
from sklearn.metrics import accuracy_score
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

df = pd.read_csv("/home/emjot13/Desktop/studia/ai/lab7/diabetes.csv")


X = df[['Pregnancies', 'Glucose', 'BloodPressure',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

f_name = ['Pregnancies', 'Glucose', 'BloodPressure',
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

dtc = tree.DecisionTreeClassifier(max_depth=4, criterion='entropy')
dtc = dtc.fit(X_train, y_train)
prediction = dtc.predict(X_test)
accuracy = accuracy_score(prediction, y_test) # about 76 % accuracy


# dot_data = StringIO()
# export_graphviz(dtc, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names=f_name, class_names=['0', '1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())
