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

booleans = train.isnull()
print(booleans)

# Booleans matrix

#      PassengerId  Survived  Pclass   Name    Sex    Age  SibSp  Parch  Ticket   Fare  Cabin  Embarked
# 0          False     False   False  False  False  False  False  False   False  False   True     False
# 1          False     False   False  False  False  False  False  False   False  False  False     False
# 2          False     False   False  False  False  False  False  False   False  False   True     False
# 3          False     False   False  False  False  False  False  False   False  False  False     False
# 4          False     False   False  False  False  False  False  False   False  False   True     False

sns.heatmap(booleans, yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# After analyze the result heatmap (01.heatmap.png) we see that we have missing data for age and cabin variable.

sns.set_style('whitegrid')

sns.countplot(x='Survived', data=train)
plt.show()
# 02.countplot-survived.png

sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
plt.show()
# 03.countplot-survived-bysex.png

sns.countplot(x='Survived', hue='Pclass', data=train)
plt.show()
# 04.countplot-survived-bypclass.png


# dropna -> removes missing values
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
plt.show()

# Same plot using another way
train['Age'].plot.hist(bins=30)
plt.show()

print(train.info())

# [891 rows x 12 columns]
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object
# Embarked       889 non-null object
# dtypes: float64(2), int64(5), object(5)
# memory usage: 83.7+ KB
# None

separator()

# SibSp: number of siblings or spouses on board

sns.countplot(x='SibSp', data=train)
plt.show()

# How much people pay as distribution
train['Fare'].hist(bins=40, figsize=(10,4))
plt.show()

# import cufflinks as cf
# cf.go_offline()
# train['Fare'].plot(kind='hist', bins=30)
