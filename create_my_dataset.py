 
 

import pandas as pd
import numpy as np

import sys, math
import pickle
sys.path.append("../tools/")

from loader import dump_data, dump_feature_list


### Load the dictionary containing the original dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


### Task 2: Remove outliers
# A quick view on data summary statistics visualized with boxplot shows very significant outlier.
# Our data contain the 'Total' sum of all records. Removed manually:
data_dict.pop( 'TOTAL', 0 )
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


### Task 3.1:  Engineer new features from existing one. No attempt to create features from email body text
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
    data_dict[person]['from_ratio_log'] = frpoi_ratio
    if frpoi_ratio>0: data_dict[person]['from_ratio_log'] = math.log(frpoi_ratio)
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

### Save the modified dataset as my_dataset
my_dataset = data_dict

#print [data_dict[p]['comm2'] for p in  data_dict.keys()]
#print [data_dict[p]['poi'] for p in  data_dict.keys()]
### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_data(my_dataset)
