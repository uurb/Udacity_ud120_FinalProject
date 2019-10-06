#!/usr/bin/python

import sys
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Create a dataframe from the dictionary for manipulation and replace NaN strings with np.nan
enron_frame = pd.DataFrame.from_dict(data_dict, orient='index')
enron_frame = enron_frame.replace('NaN', np.nan)


# Data Cleaning
enron_frame = enron_frame.drop(columns=["email_address", "restricted_stock_deferred", "loan_advances", "director_fees", "deferral_payments"])
enron_frame = enron_frame.drop(["TOTAL", "THE TRAVEL AGENCY IN THE PARK"])


# Creating the new features possibly useful
enron_frame['to_poi_ratio'] = enron_frame['from_poi_to_this_person'] / enron_frame['to_messages']
enron_frame['from_poi_ratio'] = enron_frame['from_this_person_to_poi'] / enron_frame['from_messages']
enron_frame['bonus_to_total'] = enron_frame['bonus'] / enron_frame['total_payments']
enron_frame['bonus_to_salary'] = enron_frame['bonus'] / enron_frame['salary']
enron_frame['expenses_to_salary'] = enron_frame['expenses'] / enron_frame['salary']
enron_frame['stock_to_salary'] = enron_frame['total_stock_value'] / enron_frame['salary']


# The updated features list  
features_list = ['poi', 'to_messages', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'from_messages', 'other', 'from_this_person_to_poi',
'salary', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person', 'to_poi_ratio', 'from_poi_ratio',
'bonus_to_salary', 'bonus_to_total','expenses_to_salary','stock_to_salary']


# Handling the missing values 
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy = 'median', axis=0)

pois = enron_frame[enron_frame.poi.isin([True])]
nonpois = enron_frame[enron_frame.poi.isin([False])]

transformed_pois = imp.fit_transform(pois)
transformed_nonpois = imp.fit_transform(nonpois)
transformed_all = np.concatenate((transformed_pois, transformed_nonpois))
enron_frame[:] = transformed_all


# Scaling values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
enron_scaled = scaler.fit_transform(enron_frame)
enron_frame[:] = enron_scaled

# Pandas to Dictionary
my_dataset = enron_frame.to_dict(orient='index')



# Building the model
clf = Pipeline([
    ('select_features', SelectKBest(k=9)),
    ('classify', DecisionTreeClassifier(criterion="gini", min_samples_split=6, min_samples_leaf=2, max_depth=None))
])

dump_classifier_and_data(clf, my_dataset, features_list)
