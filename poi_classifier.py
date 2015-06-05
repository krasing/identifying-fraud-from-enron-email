import pandas as pd
import numpy as np

import sys, math
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from loader import load_data, dump_feature_list



### Load modified dataset as my_dataset
my_dataset = load_data()


###########################################
##### Put selected features here #########

features_list = ['poi',
    'exercised_stock_options',
    'total_stock_value',
    'bonus']
#features_list=['poi', 'comm2']
#features_list=['poi', 'exercised_stock_options']
###########################################


### Extract the labels and selected features from my dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Preprocess features (scale) if needed
from sklearn import preprocessing
features_scaled = preprocessing.scale(features)



### Task 4.1:  Pick an algorithm

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
#clf = make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
#clf = linear_model.LogisticRegression()
#clf = RandomForestClassifier(n_estimators=50, max_features=2, min_samples_split=8)
#clf = RandomForestClassifier()
#clf = SVC(kernel="linear", C=1000000)
#clf = make_pipeline(preprocessing.StandardScaler(), SVC())
#clf = make_pipeline(preprocessing.StandardScaler(), SVC(kernel="rbf", C=100))
###########################################


###Task 5. Test and evaluate the identifier

print '\nPerformance:'
test_classifier(clf, my_dataset, features_list)



### Task 4.2:  Tune the algorithm
# Time consuming, uncomment to try

#print '\nTuned models: \n'
#from sklearn import grid_search


#parameters = {'svc__C':[100, 100000, 1000000, 10000000], 'svc__gamma':[0, 2, 10, 100], 'svc__kernel':["rbf","linear"]}
#SVM = Pipeline([('scale', preprocessing.StandardScaler()), ('svc', SVC())])
#clf = grid_search.GridSearchCV(SVM, parameters, scoring='f1')
#test_classifier(clf, my_dataset, features_list)
#print clf.best_estimator_

#parameters = {'n_estimators':[10, 100, 200], 'max_features':[None, 1, 2], 'min_samples_split':[2, 4, 8]}
#RF = RandomForestClassifier()
#clf = grid_search.GridSearchCV(RF, parameters, scoring='f1')
#test_classifier(clf, my_dataset, features_list)
#print clf.best_estimator_





dump_classifier_and_data(clf, my_dataset, features_list)
