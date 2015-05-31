 
 

import pandas as pd
import numpy as np

import sys, math
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data




### Load the dictionary containing the original dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )



### Task 2: Remove outliers
# A quick view on data summary statistics visualized with boxplot shows very significant outlier.
# Our data contain the 'Total' sum of all records. Removed manually:
data_dict.pop( 'TOTAL', 0 )


### Engineer new features from existing one. No attempt to create features from email body text
for person in data_dict.keys():
    to = float(data_dict[person]['to_messages'])
    fr = float(data_dict[person]['from_messages'])
    topoi = float(data_dict[person]['from_poi_to_this_person'])
    frpoi = float(data_dict[person]['from_this_person_to_poi'])
    if math.isnan(topoi):
        topoi = 0
    if math.isnan(frpoi):
        frpoi = 0
    topoi_ratio = topoi/to
    if math.isnan(topoi_ratio):
        topoi_ratio = 0
    frpoi_ratio = frpoi/fr
    if math.isnan(frpoi_ratio):
        frpoi_ratio = 0
    
    data_dict[person]['to_ratio'] = topoi_ratio    
    data_dict[person]['from_ratio'] = frpoi_ratio
    data_dict[person]['comm'] = topoi>60 and frpoi>60
    data_dict[person]['maildata'] = to>0
    data_dict[person]['comm_sum'] = topoi + frpoi
    data_dict[person]['comm_ratio'] = topoi_ratio + frpoi_ratio
    data_dict[person]['comm_max'] = max(topoi, frpoi)
    data_dict[person]['comm_min'] = min(topoi, frpoi)
    if topoi>0 and frpoi>30:
        data_dict[person]['comm2'] = 2
    elif to:
        data_dict[person]['comm2'] = 0
    else:
        data_dict[person]['comm2'] = 1
#    for f in data_dict[person].keys():
#        if f>0 and type(f)==type(1.):
#            f = math.log(f)
#        elif  f<0:
#            f = math.log(-f)


# List all available features
features_listA = ['salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees']

features_listB = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 


features_listC = ['maildata', 'to_ratio', 'from_ratio',
                  'comm', 'comm_sum', 'comm_max', 'comm_min',
                  'comm_ratio', 'comm2']

features_label = ['poi']

features_list_full = features_label + features_listA + features_listB + features_listC

###########################################
##### Put selected features here #########

features_list = features_list_full
features_list = ['poi', 'bonus', 'total_stock_value', 'salary',
'exercised_stock_options', 'long_term_incentive',
'restricted_stock',
'comm2']
features_list = ['poi', 'bonus', 'total_stock_value', 'salary',
'exercised_stock_options',
'restricted_stock']
###########################################

### Save the modified dataset as my_dataset
my_dataset = data_dict

### Extract the labels and selected features from my dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Preprocess features (scale) if needed
from sklearn import preprocessing
features = preprocessing.scale(features)
#print 'Sample values of scaled features:'
# print features_scaled[1]
# print labels
features_scaled = (features - np.min(features, 0)) / (np.max(features, 0) + 0.0001)  # 0-1 scaling


### Split the dataset to train and test subsets
# We don't want to select features based on all data.
# Test set should remain hidden

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.5, random_state=40)
features_train_scaled, features_test_scaled, labels_train, labels_test = train_test_split(
    features_scaled, labels, test_size=0.5, random_state=40)

### Evaluate features
print '\nInitial features:'
print features_list

#### Random forests feature importance
from sklearn.ensemble import RandomForestClassifier
clfEval = RandomForestClassifier(n_estimators=10, max_features=None)
clfEval.fit(features_train, labels_train)
print '\nFeature importance for Trees'
print clfEval.feature_importances_ # for trees
###format feature importances for feature selection process
# https://github.com/alexbra/ud-nd-project4/blob/master/poi_id.py
feature_importances = clfEval.feature_importances_
sorted_idx = (-np.array(feature_importances)).argsort()
print "Rank of features"
for idx in sorted_idx:
    print "{:4f} : {}".format(feature_importances[idx], features_list[idx+1])

#### Linear regression coefficient significance
import numpy as np
import statsmodels.api as sm
features_train_sm = sm.add_constant(features_train_scaled)
results = sm.OLS(labels_train, features_train_sm).fit()
print '\nSummary for linear regression'
# print results.summary()
pval = results.pvalues
print "P-values ordered"
sorted_idx = (pval).argsort()
for idx in sorted_idx:
    print "{:4f} : {}".format(pval[idx], features_list[idx])

### SelectKBest feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
selector.fit(features_train, labels_train)
selected_features_index = selector.get_support(indices=True)
print selected_features_index
fl= [features_list[i+1] for i in selected_features_index]
features_list_KBest = ['poi'] + fl
print 'SelectKBest'
print features_list_KBest
#features_train = selector.transform(features_train)
#features_test  = selector.transform(features_test)
#print features_list

### Pick an algorithm

# Random Forests
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_features=None)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Logistic regression
from sklearn import linear_model
clf = linear_model.LogisticRegression()

# Linear regression (faulty)
from sklearn import linear_model
clf = linear_model.LinearRegression()

# MultinomialNB - if text features are added
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(class_prior=[1-1e-9, 1e-9])

# SVM
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)
clf = SVC(kernel="linear", C=10000)

# Lasso linear regression
from sklearn import linear_model
clf = linear_model.Lasso(alpha = 0.1)

# Combine multiple models, e.g. first feature scaling than LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import preprocessing
clf = make_pipeline(preprocessing.StandardScaler(), GaussianNB()) # no effect of normalization
clf = make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression()) # big effect of normalization


###########################################
##### Put selected algorithm here #########

clf = GaussianNB()
#clf = SVC(kernel="rbf", C=100)
#clf = make_pipeline(preprocessing.StandardScaler(), SVC())
#clf = make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
#clf = RandomForestClassifier(n_estimators=50, max_features=3, min_samples_split=4)
#clf = make_pipeline(preprocessing.StandardScaler(), SVC(kernel="rbf", C=1000000))
###########################################

### Test and evaluate the identifier

#### Fit the model
clf.fit(features_train, labels_train)

#### Evaluate performance
print '\nMy performance evaluation on small dataset'
y_pred = clf.predict(features_test)
print 'POI in train and test split: ', sum(labels_train), sum(labels_test)

from sklearn.metrics import classification_report
print labels_test
print y_pred
print(classification_report(labels_test, y_pred))


### Tune the algorithm

#from sklearn import grid_search
#parameters = {'n_estimators':[4, 10, 100, 200], 'min_samples_split':[2, 4, 8]}
#RF = RandomForestClassifier()
#clf = grid_search.GridSearchCV(RF, parameters)
#test_classifier(clf, my_dataset, features_list)
#clf.best_estimator_



# Check the performance presented to the reviewer
print '\nPerformance presented to the reviewer'
test_classifier(clf, my_dataset, features_list)

print features_list
print 'Number of features: ', len(features_list)

#print '\nTuned model'
#from sklearn import grid_search
##parameters = {'SVC__C':[10, 100, 10000, 1000000], 'SVC__gamma':[0, 1, 2, 4, 8, 100]}
##parameters = {'svc__C':[100, 100000, 10000000], 'svc__gamma':[0, 2, 10]}

##SVM = Pipeline([('scale', preprocessing.StandardScaler()), ('svc', SVC())])
##clf = grid_search.GridSearchCV(SVM, parameters, scoring='f1')

#parameters = {'n_estimators':[10, 30, 50, 70], 'max_features':[None, 2, 3, 4], 'min_samples_split':[2, 4, 6]}
#RF = RandomForestClassifier()
#clf = grid_search.GridSearchCV(RF, parameters, scoring='f1')


#test_classifier(clf, my_dataset, features_list)
#print clf.best_estimator_

dump_classifier_and_data(clf, my_dataset, features_list)