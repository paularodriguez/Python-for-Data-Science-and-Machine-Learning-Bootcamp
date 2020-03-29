import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Utils to see all dataframe columns
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 100)


def separator():
    print('-' * 20)


train = pd.read_csv('titanic_train.csv')
print(train.head())

#    PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
# 0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
# 1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
# 2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
# 3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
# 4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S

# Step 1: Clean missing values

sns.boxplot(x='Pclass', y='Age', data=train)
plt.show()


# We see, in the previous boxplot, that passenger ages are bigger in first class than second class, and it also occurs for
# second and third classes.

# Now, we are going to infer missing passenger ages:

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    # 37 / 29 / 24 : values from boxplot
    if pd.isnull(Age):
        if Pclass == 1:
            return np.rint(train[train['Pclass'] == 1]['Age'].mean())
        elif Pclass == 2:
            return np.rint(train[train['Pclass'] == 2]['Age'].mean())
        else:
            return np.rint(train[train['Pclass'] == 3]['Age'].mean())
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
print(train)

# Take a look to the heatmap after fill missing ages
sns.heatmap(train.isnull())
plt.show()

# 10.heatmap-after-fill-ages.png - We see that there are no empty ages
separator()

# Drop cabin column

train.drop('Cabin', axis=1, inplace=True)
print(train.head())

#    PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Embarked
# 0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500        S
# 1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833        C
# 2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250        S
# 3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000        S
# 4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500        S


# Drop missing values
train.dropna(inplace=True)
separator()

# Step 2: Categorical features

# Method that converts categorical variables into dummies variables

# We start converting sex variable
# We have to remove one of them, because is redundant
sex = pd.get_dummies(train['Sex'], drop_first=True)
print(sex)

#      male
# 0       1
# 1       0
# 2       0
# 3       0
# 4       1

# The same for embarks
# Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton

embark = pd.get_dummies(train['Embarked'], drop_first=True)
separator()

# Concat all sets

train = pd.concat([train, sex, embark], axis=1)
print(train.head())

#    PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Embarked  male  Q  S
# 0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500        S     1  0  1
# 1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833        C     0  0  0
# 2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250        S     0  0  1
# 3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000        S     0  0  1
# 4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500        S     1  0  1

# Drop no longer needed columns

train.drop(['Name', 'Sex', 'Embarked', 'Ticket'], axis=1, inplace=True)
print(train.head())

#    PassengerId  Survived  Pclass   Age  SibSp  Parch     Fare  male  Q  S
# 0            1         0       3  22.0      1      0   7.2500     1  0  1
# 1            2         1       1  38.0      1      0  71.2833     0  0  0
# 2            3         1       3  26.0      0      0   7.9250     0  0  1
# 3            4         1       1  35.0      1      0  53.1000     0  0  1
# 4            5         0       3  35.0      0      0   8.0500     1  0  1
separator()

# Drop also PassengerId

train.drop(['PassengerId'], axis=1, inplace=True)
print(train.head())

#    Survived  Pclass   Age  SibSp  Parch     Fare  male  Q  S
# 0         0       3  22.0      1      0   7.2500     1  0  1
# 1         1       1  38.0      1      0  71.2833     0  0  0
# 2         1       3  26.0      0      0   7.9250     0  0  1
# 3         1       1  35.0      1      0  53.1000     0  0  1
# 4         0       3  35.0      0      0   8.0500     1  0  1
separator()


# STEP 3: Build and train the model

print(train.columns)
# Index(['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S'], dtype='object')

# features
X = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]
# column to predict
y = train['Survived']
separator()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print(X_train.head())

from sklearn.linear_model import LogisticRegression

# Create and train the model (Logistic Regression)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)


# Evaluate the result
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))


#               precision    recall  f1-score   support
#
#            0       0.82      0.92      0.87       163
#            1       0.85      0.69      0.76       104
#
#     accuracy                           0.83       267
#    macro avg       0.84      0.81      0.82       267
# weighted avg       0.83      0.83      0.83       267

# Confusion matrix

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))

# [[150  13]
#  [ 32  72]]

# TT: 150
# TN: 72
# FP: 32
# FN: 13
# Accuracy: (150 + 72) / (150 + 13 + 32 + 72) = 0.83


