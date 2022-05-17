# # Assigning features and label variables
# from sklearn.preprocessing import LabelEncoder
#
# weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny',
#            'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
#
# temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
#
# play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
#
# from sklearn import preprocessing
#
# # creating labelEncoder
# le = preprocessing.LabelEncoder()
# # Converting string labels into numbers.
# weather_encoded = le.fit_transform(weather)
# print(weather_encoded)
#
# temp_encoded = le.fit_transform(temp)
# label = le.fit_transform(play)
# print("Temp:", temp_encoded)
# print("Play:", label)
# features = zip(weather_encoded, temp_encoded)
#
# from sklearn.naive_bayes import GaussianNB
#
# # Create a Gaussian Classifier
# model = GaussianNB()
#
# # Train the model using the training sets
# model.fit(features, label)
#
# # Predict Output
# predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
# print("Predicted Value:", predicted)


# #Import scikit-learn dataset library
# from sklearn import datasets

# #Load dataset
# iris = datasets.load_iris()
# from sklearn.cross_validation import train_test_split

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=13)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(f"Accuracy: {round((y_test == y_pred).sum() / (X_test.shape[0])*100, 2)} %")

# aproximate accuracy: 95%