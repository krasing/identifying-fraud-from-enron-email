#!/usr/bin/pickle

""" a basic script for saving and loading intermediate
    results: dataset, feature_list, classifier to 
    my_dataset.pkl, my_feature_list.pkl, and
    my_classifier.pkl, respectively
"""

import pickle
import sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    pickle.dump(clf, open(CLF_PICKLE_FILENAME, "w") )
    pickle.dump(dataset, open(DATASET_PICKLE_FILENAME, "w") )
    pickle.dump(feature_list, open(FEATURE_LIST_FILENAME, "w") )

def load_classifier_and_data():
    clf = pickle.load(open(CLF_PICKLE_FILENAME, "r") )
    dataset = pickle.load(open(DATASET_PICKLE_FILENAME, "r") )
    feature_list = pickle.load(open(FEATURE_LIST_FILENAME, "r"))
    return clf, dataset, feature_list

def dump_data(dataset):
    pickle.dump(dataset, open(DATASET_PICKLE_FILENAME, "w") )

def dump_feature_list(feature_list):
    pickle.dump(feature_list, open(FEATURE_LIST_FILENAME, "w") )
    
def load_data():
    dataset = pickle.load(open(DATASET_PICKLE_FILENAME, "r") )
    return dataset
    
def load_feature_list():
    feature_list = pickle.load(open(FEATURE_LIST_FILENAME, "r"))
    return feature_list 
