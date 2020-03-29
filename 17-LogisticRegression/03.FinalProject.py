# In this project we will be working with a fake advertising data set,
# indicating whether or not a particular internet user clicked on an Advertisement.
# We will try to create a model that will predict whether or not they will
# click on an ad based off the features of that user.
#
# This data set contains the following features:
#
# 'Daily Time Spent on Site': consumer time on site in minutes
# 'Age': cutomer age in years
# 'Area Income': Avg. Income of geographical area of consumer
# 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# 'Ad Topic Line': Headline of the advertisement
# 'City': City of consumer
# 'Male': Whether or not consumer was male
# 'Country': Country of consumer
# 'Timestamp': Time at which consumer clicked on Ad or closed window
# 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# Import a few libraries you think you'll need (Or just import them as you go along!)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Utils to see all dataframe columns
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 100)


def separator():
    print('-' * 20)


# Read in the advertising.csv file and set it to a data frame called ad_data.
ad_data = pd.read_csv('advertising.csv')

# Check the head of ad_data
print(ad_data.head())

#    Daily Time Spent on Site  Age  Area Income  Daily Internet Usage                          Ad Topic Line            City  Male     Country            Timestamp  Clicked on Ad
# 0                     68.95   35     61833.90                256.09     Cloned 5thgeneration orchestration     Wrightburgh     0     Tunisia  2016-03-27 00:53:11              0
# 1                     80.23   31     68441.85                193.77     Monitored national standardization       West Jodi     1       Nauru  2016-04-04 01:39:02              0
# 2                     69.47   26     59785.94                236.50       Organic bottom-line service-desk        Davidton     0  San Marino  2016-03-13 20:35:42              0
# 3                     74.15   29     54806.18                245.89  Triple-buffered reciprocal time-frame  West Terrifurt     1       Italy  2016-01-10 02:31:19              0
# 4                     68.37   35     73889.99                225.58          Robust logistical utilization    South Manuel     0     Iceland  2016-06-03 03:36:18              0
separator()

# Use info and describe() on ad_data
print(ad_data.info())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 10 columns):
# Daily Time Spent on Site    1000 non-null float64
# Age                         1000 non-null int64
# Area Income                 1000 non-null float64
# Daily Internet Usage        1000 non-null float64
# Ad Topic Line               1000 non-null object
# City                        1000 non-null object
# Male                        1000 non-null int64
# Country                     1000 non-null object
# Timestamp                   1000 non-null object
# Clicked on Ad               1000 non-null int64
# dtypes: float64(3), int64(3), object(4)
# memory usage: 78.2+ KB
# None
separator()

print(ad_data.describe())

#        Daily Time Spent on Site          Age   Area Income  Daily Internet Usage         Male  Clicked on Ad
# count               1000.000000  1000.000000   1000.000000           1000.000000  1000.000000     1000.00000
# mean                  65.000200    36.009000  55000.000080            180.000100     0.481000        0.50000
# std                   15.853615     8.785562  13414.634022             43.902339     0.499889        0.50025
# min                   32.600000    19.000000  13996.500000            104.780000     0.000000        0.00000
# 25%                   51.360000    29.000000  47031.802500            138.830000     0.000000        0.00000
# 50%                   68.215000    35.000000  57012.300000            183.130000     0.000000        0.50000
# 75%                   78.547500    42.000000  65470.635000            218.792500     1.000000        1.00000
# max                   91.430000    61.000000  79484.800000            269.960000     1.000000        1.00000
separator()

## Exploratory Data Analysis

# Create a histogram of the Age

sns.distplot(ad_data['Age'], bins=30, kde=False)
plt.show()

# Create a jointplot showing Area Income versus Age

sns.jointplot(ad_data['Age'], ad_data['Area Income'])
plt.show()

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age

sns.jointplot(ad_data['Age'], ad_data['Daily Time Spent on Site'], kind='kde', color='red')
plt.show()

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage

sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Daily Internet Usage'], color='green')
plt.show()

# Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature

# sns.pairplot(ad_data, hue='Clicked on Ad') -> numpy issue
sns.pairplot(ad_data)
plt.show()


##  Regression

# Now it's time to do a train test split, and train our model!
# You'll have the freedom here to choose columns that you want to train on!

# Split the data into training set and testing set using train_test_split

from sklearn.model_selection import train_test_split

print(ad_data.columns)

# Index(['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country', 'Timestamp', 'Clicked on Ad'], dtype='object')
separator()

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train and fit a logistic regression model on the training set.

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

## Predictions and Evaluations

# Now predict values for the testing data

predictions = logmodel.predict(X_test)

# Create a classification report for the model

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predictions))
# [[149   8]
#  [ 14 129]]

print(classification_report(y_test, predictions))

#               precision    recall  f1-score   support
#
#            0       0.91      0.95      0.93       157
#            1       0.94      0.90      0.92       143
#
#     accuracy                           0.93       300
#    macro avg       0.93      0.93      0.93       300
# weighted avg       0.93      0.93      0.93       300
