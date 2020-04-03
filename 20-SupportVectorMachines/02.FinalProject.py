# Support Vector Machines Project

# Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!

## The Data
# For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set).
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis.
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples.
# Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

# The iris dataset contains measurements for 150 iris flowers from three different species.

# The three classes in the Iris dataset:
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
#
# The four features of the Iris dataset:
#
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn to get the iris data by using: iris = sns.load_dataset('iris')

iris = sns.load_dataset('iris')

# print(iris.head())

#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa


# Exploratory Data Analysis

# Create a pairplot of the data set. Which flower species seems to be the most separable?

# sns.pairplot(iris, hue='species')
# plt.show()

# setosa seems most separable

# Create a kde plot of sepal_length versus sepal width for setosa species of flower

setosa = iris[iris['species'] == 'setosa']

# sns.jointplot(x='sepal_width', y='sepal_length', data=setosa, kind='kde', color='red')
# plt.show()

# Train Test Split
# Split your data into a training set and a testing set.

from sklearn.model_selection import train_test_split

X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train a Model

# Now its time to train a Support Vector Machine Classifier.
# Call the SVC() model from sklearn and fit the model to the training data.

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

# Model Evaluation
# Now get predictions from the model and create a confusion matrix and a classification report.

predictions = svc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

#               precision    recall  f1-score   support
#
#       setosa       1.00      1.00      1.00        13
#   versicolor       1.00      0.95      0.97        20
#    virginica       0.92      1.00      0.96        12
#
#     accuracy                           0.98        45
#    macro avg       0.97      0.98      0.98        45
# weighted avg       0.98      0.98      0.98        45


print(confusion_matrix(y_test, predictions))

# [[13  0  0]
#  [ 0 19  1]
#  [ 0  0 12]]
