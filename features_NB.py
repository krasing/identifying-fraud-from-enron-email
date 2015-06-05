 
 
 

import pandas as pd
import numpy as np

import sys, math
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from loader import load_data


### List all available features
features_listA = ['salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees']

features_listB = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 


features_listC = ['maildata', 'to_ratio', 'from_ratio',
                  'comm', 'comm_sum', 'comm_max', 'comm_min',
                  'comm_ratio', 'comm2', 'from_ratio_log']

features_label = ['poi']

features_list_full = features_label + features_listA + features_listB + features_listC

features_list = features_list_full


### Load modified dataset as my_dataset
my_dataset = load_data()

print 'initial features count: ', len(features_listA + features_listB)
print 'total datapoints count: ', len(my_dataset)

### Extract the labels and selected features from my dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




### Task 3.2:  Evaluate features
print '\nInitial features:'
print features_list


### SelectKBest feature selection
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=15)
selector.fit(features, labels)
selected_features_index = selector.get_support(indices=True)

print selected_features_index
fl= [features_list[i+1] for i in selected_features_index]
features_list = ['poi'] + fl
print 'SelectKBest'
print features_list
features = selector.transform(features)


featurescore = f_classif(features, labels)
sorted_idx = featurescore[1].argsort()
print "Rank of features"
for idx in sorted_idx:
    print "{:4f} : {}".format(featurescore[1][idx], features_list[idx+1])


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


# Evaluate the performance of multiple feature sets
fl= [features_list[i+1] for i in sorted_idx]
features_list = ['poi'] + fl
for i in range(1,15):
    features_list_i = features_list[0:i+1]
    print 'Selected, ', i, ' best:'
    print features_list_i
    test_classifier(clf, my_dataset, features_list_i)
