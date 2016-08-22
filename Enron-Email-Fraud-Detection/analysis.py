import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

## FETCHING ENRON DATA##
enron_data = pickle.load(open("final_project_dataset.pkl", "r"))

#creating numpy array for dictionary of dictionaries since sklearn cant read dictionaries

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)

#splitting target and features for prediction

def targetFeatureSplit( data ):

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features
    
#no of features in enron dataset
print len(enron_data['METTS MARK'].keys())
enron_data['METTS MARK'].keys()

#how many employees have  a quantified salary
count_salaried=0
for personname in enron_data.keys():
    if(enron_data[personname]['salary']!= 'NaN'):
        count_salaried+=1
print count_salaried

## email messages from Wesley Colwell to persons of interest

print enron_data["COLWELL WESLEY"]['from_this_person_to_poi']


## People who have NAN for total payments
count_nan_payments=0
for personname in enron_data.keys():
    if(enron_data[personname]['total_payments']== 'NaN'):
        count_nan_payments+=1
print count_nan_payments  


### no of POIs with nan total payments
count_poi_nan_payments=0
for personname in enron_data.keys():
    if(enron_data[personname]['poi'] == 1 and enron_data[personname]['total_payments']== 'NaN'):
        count_poi_nan_payments+=1
print count_poi_nan_payments 

## ALL the POIs have some payment.S
## So if we include the remaining POIs from names text file to dictionary, Total Payments for POI will become Null, And we cant 
#include total payments as feature 

##predicting bonus for employees from salary feature

features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

###regression goes here!

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

reg = LinearRegression()
reg.fit(feature_train, target_train)

#slope
print reg.coef_
print reg.intercept_

#score

print reg.score(feature_test,target_test)

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 
    
### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    print "Exception"
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

### Predicting bonus from the long_term_incentive

features_list = ["bonus", "long_term_incentive"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

reg = LinearRegression()
reg.fit(feature_train, target_train)

#slope
print reg.coef_
print reg.intercept_

#score

print reg.score(feature_test,target_test)

##removing outliers

# Remove the outlier - 'TOTAL' entry
enron_data.pop('TOTAL', 0)

features = ["salary", "bonus"]
data = featureFormat(enron_data, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

## Printing outliers

outliers = []
for key in enron_data:
    val = enron_data[key]['salary']
    if val == 'NaN':
        continue 
    
    outliers.append((key,int(val)))
    
outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:2])
outliers_final

## But we leave these 2 entries since they are Enron's Bosses and definitely POIs

### CLEANING OTHER UNWANTED RECORDS 

## checking for entries that donot represent a personname
for personname in enron_data.keys():
    if(len(personname.split())>4):
        print personname

enron_data['THE TRAVEL AGENCY IN THE PARK']
## Hence we need to pop this entry as it doesnot represent a peron 

enron_data.pop('THE TRAVEL AGENCY IN THE PARK', 0)


## Checking other records with no useful data

enron_data['LOCKHART EUGENE E'].values()

## No useful info for this user
enron_data.pop('LOCKHART EUGENE E', 0)

## FEATURE SELECTION ##

from sklearn.feature_selection import SelectKBest

features_list = ['poi',
                     'bonus',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'exercised_stock_options',
                     'expenses',
                     'loan_advances',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'total_payments',
                     'total_stock_value',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'to_messages']


def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features

best_features=get_k_best(enron_data,features_list,10)
my_feature_list = ['poi'] + best_features.keys()
my_feature_list

## ADDING A  NEW FEATURE FINANCIAL AGGREGATE TO CHECK CALUE OF TOTAL FINANCES ##
## finances = stocks + salary

def add_financial_aggregate(data_dict):
    fields = ['total_stock_value', 'exercised_stock_options', 'salary']
    for person in data_dict.keys():
        is_valid = True
        for field in fields:
            if data_dict[person][field] == 'NaN':
                is_valid = False
        if is_valid:
            data_dict[person]['financial_aggregate'] = sum([data_dict[person][field] for field in fields])
        else:
            data_dict[person]['financial_aggregate'] = 'NaN'
    

add_financial_aggregate(enron_data)

## To Analyse interaction of a peron with POI .add new feature Interaction with POI

def add_poi_interaction(data_dict):
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] + person['from_messages']
            poi_messages = person['from_poi_to_this_person'] + person['from_this_person_to_poi']
            person['poi_interaction'] = float(poi_messages) / total_messages
        else:
            person['poi_interaction'] = 'NaN'
            
add_poi_interaction(enron_data)

my_feature_list+=['poi_interaction']

# extract the features specified in features_list
data = featureFormat(enron_data, my_feature_list)

# split into labels and features (this line assumes that the first
# feature in the array is the label, which is why "poi" must always
# be first in the features list
labels, features = targetFeatureSplit(data)

## Code to evaluate a particular algorithm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import math

def evaluate_clf(clf, features, labels, test_size=0.3):
    accuracy = 0
    precision = 0
    recall = 0

    features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=test_size)
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    accuracy= accuracy_score(labels_test, predictions)
    precision = precision_score(labels_test, predictions)
    recall= recall_score(labels_test, predictions)

    print "precision: {}".format(precision)
    print "recall:    {}".format(recall)
    print accuracy
    return predictions
    
    
### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

l_clf = LogisticRegression(C=10**18, tol=10**-21)


### Selected Classifiers Evaluation
predictions= evaluate_clf(l_clf, features, labels)


### K-means Clustering
from sklearn.cluster import KMeans

k_clf = KMeans(n_clusters=2, tol=0.001)

predictions=evaluate_clf(k_clf, features, labels)

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm='SAMME')

predictions=evaluate_clf(a_clf, features, labels)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

predictions=evaluate_clf(rf_clf, features, labels)

### Stochastic Gradient Descent - Logistic Regression
from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log')

predictions=evaluate_clf(g_clf, features, labels)
