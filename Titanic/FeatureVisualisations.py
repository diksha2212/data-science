##IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline


##LOADING DATA
titanic_train=pd.read_csv("train.csv",dtype={"Age": np.float64},)
titanic_test=pd.read_csv("test.csv",dtype={"Age": np.float64},)

##Starting with embarked column

titanic_train['Embarked']=titanic_train['Embarked'].fillna('S')
sns.factorplot('Embarked','Survived', data=titanic_train,size=4,aspect=3)
sns.countplot(x='Survived', hue="Embarked", data=titanic_train, order=[1,0])
sns.countplot(x='Embarked', data=titanic_train)


##Fare Column

titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median(),inplace=True)

fare_not_survived=titanic_train['Fare'][titanic_train['Survived']==0]
fare_survived=titanic_train['Fare'][titanic_train['Survived']==1]

mean_fare_not_survived=fare_not_survived.mean()
mean_fare_survived=fare_survived.mean()
std_fare_not_survived=fare_not_survived.std()
std_fare_survived=fare_survived.std()

#plot

plt.hist(titanic_train['Fare'],bins=10)

#AGE
average_age_train=titanic_train['Age'].mean()
std_age_train=titanic_train['Age'].std()
count_nan_age_train=titanic_train['Age'].isnull().sum()

average_age_test=titanic_test['Age'].mean()
std_age_test=titanic_test['Age'].std()
count_nan_age_test=titanic_test['Age'].isnull().sum()

random_train=np.random.randint(average_age_train-std_age_train,average_age_train+std_age_train,size=count_nan_age_train)
random_test=np.random.randint(average_age_test-std_age_test,average_age_test+std_age_test,size=count_nan_age_test)

# fill NaN values in Age column with random values generated
titanic_train["Age"][np.isnan(titanic_train["Age"])] = random_train
titanic_test["Age"][np.isnan(titanic_test["Age"])] = random_test

plt.hist(titanic_train['Age'],bins=50)

##SibSp And Parch

titanic_train['Family']=titanic_train['SibSp']+titanic_train['Parch']
titanic_test['Family']=titanic_test['SibSp']+titanic_test['Parch']

titanic_train['Family'][titanic_train['Family']>0]=1
titanic_train['Family'][titanic_train['Family']==0]=0

titanic_test['Family'][titanic_test['Family']>0]=1
titanic_test['Family'][titanic_test['Family']==0]=0

# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Family', data=titanic_train, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = titanic_train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)

##sex
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=titanic_train, ax=axis1)

# average of survived for each Person(male, female, or child)
family_perc = titanic_train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=family_perc, ax=axis2, order=['male','female','child'])
