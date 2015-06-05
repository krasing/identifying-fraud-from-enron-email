 
 

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

### Preprocess features (scale) if needed
from sklearn import preprocessing
features = preprocessing.scale(features)
#print 'Sample values of scaled features:'
# print features_scaled[1]
# print labels
#features_scaled = (features - np.min(features, 0)) / (np.max(features, 0) + 0.0001)  # 0-1 scaling




### Task 3.2:  Evaluate features
print '\nInitial features:'
print features_list


import numpy as np
import statsmodels.api as sm
features = sm.add_constant(features)
results = sm.OLS(labels, features).fit()
print '\nSummary for logistic regression'
# print results.summary()
pval = results.pvalues
print "P-values ordered"
sorted_idx = (pval[1:]).argsort()
print "Rank of features"
for idx in sorted_idx:
    print "{:4f} : {}".format(pval[idx], features_list[idx+1])

fl= [features_list[indx+1] for indx in sorted_idx]
features_list = ['poi'] + fl

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

for i in range(1,11):
    features_list_i = features_list[0:i+1]
    print 'Selected, ', i, ' best:'
    print features_list_i
    test_classifier(clf, my_dataset, features_list_i)
    


 
