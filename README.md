
# Udacity_ud120_FinalProject

## Project Goal: Identify Fraud from Enron Data

Enron corporation was one of the largest company's in America. This company went bankrupt in 2001 due to fraudulent business practices. For those who are interested about the story, further information can be found in [this wikipedia page](https://en.wikipedia.org/wiki/Enron_scandal).

The main focus of this project is to identify the person of interest (people who are related with fraud activities) as developing machine learning model. 

To achieve this goal, I applied the following procedure:
1. Investigating the dataset
2. Creating the new features possibly useful
3. Handling the missing values, Scaling values, Selecting features
4. Algorithm selection
5. Tuning the algorithm
6. Evaluation


## Dataset Analysis

I started to explore the data as asking some questions like how many people we have in total, how many of them identified as poi, how many features we have etc.

```python
print  "Number of data points:" , len(enron_frame)
```
```
Number of data points: 146
```
```python
pois = enron_frame[enron_frame.poi.isin([True])]
nonpois = enron_frame[enron_frame.poi.isin([False])]
print  "Number of pois:" , len(pois)
print  "Number of non-pois:" , len(nonpois)
```
```
Number of pois: 18
Number of non-pois: 128
```
In the data I see that number of pois and non-pois are imbalanced. This fact will effect numerous things such as choosing the right evaluation metrics.


```python
print enron_frame.info()
print pois.info()
```
```
<class 'pandas.core.frame.DataFrame'>
Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
Data columns (total 21 columns):
salary                       95 non-null float64
to_messages                  86 non-null float64
deferral_payments            39 non-null float64
total_payments               125 non-null float64
exercised_stock_options      102 non-null float64
bonus                        82 non-null float64
restricted_stock             110 non-null float64
shared_receipt_with_poi      86 non-null float64
restricted_stock_deferred    18 non-null float64
total_stock_value            126 non-null float64
expenses                     95 non-null float64
loan_advances                4 non-null float64
from_messages                86 non-null float64
other                        93 non-null float64
from_this_person_to_poi      86 non-null float64
poi                          146 non-null bool
director_fees                17 non-null float64
deferred_income              49 non-null float64
long_term_incentive          66 non-null float64
email_address                111 non-null object
from_poi_to_this_person      86 non-null float64
dtypes: bool(1), float64(19), object(1)
memory usage: 24.1+ KB

<class 'pandas.core.frame.DataFrame'>
Index: 18 entries, BELDEN TIMOTHY N to YEAGER F SCOTT
Data columns (total 21 columns):
salary                       17 non-null float64
to_messages                  14 non-null float64
deferral_payments            5 non-null float64
total_payments               18 non-null float64
exercised_stock_options      12 non-null float64
bonus                        16 non-null float64
restricted_stock             17 non-null float64
shared_receipt_with_poi      14 non-null float64
restricted_stock_deferred    0 non-null float64
total_stock_value            18 non-null float64
expenses                     18 non-null float64
loan_advances                1 non-null float64
from_messages                14 non-null float64
other                        18 non-null float64
from_this_person_to_poi      14 non-null float64
poi                          18 non-null bool
director_fees                0 non-null float64
deferred_income              11 non-null float64
long_term_incentive          12 non-null float64
email_address                18 non-null object
from_poi_to_this_person      14 non-null float64
dtypes: bool(1), float64(19), object(1)
memory usage: 3.0+ KB
```
After analyzing the data I find that some features have lots of missing values such as "restricted_stock_deferred", "loan_advances", "director_fees" and "deferral_payments". This features can be ignored for further steps because they are not helpful to interpreting the data.
Also email_address feature is completely useless because it doesn't make any sense for detecting fraud activities.
```python
enron_frame = enron_frame.drop(columns=["email_address","restricted_stock_deferred","loan_advances","director_fees","deferral_payments"])
```
After cleaning the data a bit i started to visualize the features.

```python
# visualizing salary feature
sns.boxplot(x='poi',y='salary',data=enron_frame)
plt.show()
```

![](https://github.com/uurb/Udacity_ud120_FinalProject/blob/master/outputs%26graphs/%231salary_with_total.png)

```python
#salary outlier detect
outlier = enron_frame[enron_frame["salary"]>  2e7]
print outlier.index.values
```
```
['TOTAL']
```
After visualizing the salary feature , I identified the one outlier, "TOTAL" which needs to be extracted from the data. Because it is not a real person therefore not a valid data.
Also there is a one more instance that is not a person. "THE TRAVEL AGENCY IN THE PARK" extracted from the data.
```python
enron_frame = enron_frame.drop(["TOTAL","THE TRAVEL AGENCY IN THE PARK"])
```


## Feature Engineering

### Creating New Features
The first thing I do in this section was to create new features which can helpful. I decided to add six new features to our dataset. The first two features were email ratios as "from_poi_to_this_person"/"to_messages" and "from_this_person_to_poi" / "from_messages". The thought that i have was finding ratio would be more helpful than numbers of corresponding messages. For instance, a person could send 1000 mails and 100 of them could be to poi. Also another person could send 100 mails and all of them could be to poi. In this situation comparing ratios are more meaningful than numbers.
Other 4 ones were "bonus" / "salary", "bonus" / "total_payments", "expenses" / "salary" and "total_stock_value" / "salary". The idea behind all these 4 was similar.  The ratio of bonus,expenses per salary could separate the people related with fraud and others. High bonuses, expenses over salary could be sign of fraud.

```python
enron_frame['to_poi_ratio'] = enron_frame['from_poi_to_this_person'] / enron_frame['to_messages']
enron_frame['from_poi_ratio'] = enron_frame['from_this_person_to_poi'] / enron_frame['from_messages']
enron_frame['bonus_to_total'] = enron_frame['bonus'] / enron_frame['total_payments']
enron_frame['bonus_to_salary'] = enron_frame['bonus'] / enron_frame['salary']
enron_frame['expenses_to_salary'] = enron_frame['expenses'] / enron_frame['salary']
enron_frame['stock_to_salary'] = enron_frame['total_stock_value'] / enron_frame['salary']
```

### Handling The Missing Values
There are lots of missing values in the dataset. To handle these missing values, I divided dataset to 2 section one is pois and the other one is non-pois. After that i filled this missing values with median of the features. The reason of using median instead of mean is that median is more robust to outliers compared to mean.
```
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy  =  'median', axis=0)

transformed_pois = imp.fit_transform(pois)
transformed_nonpois = imp.fit_transform(nonpois)
transformed_all = np.concatenate((transformed_pois, transformed_nonpois))
enron_frame[:] = transformed_all
```

### Scaling Values
The data needs to be scaled because they consist of different range of data units. Therefore I used [min_max scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) in order to prevent dominance of some features over others.
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
enron_scaled = scaler.fit_transform(enron_frame)
enron_frame[:] = enron_scaled
```

### Selecting Features
In order to select the most useful features in the dataset I used sklearn [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html). It gives the best k features defined by selected score function, automatically selected for use in the classifier. Also, I used [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) with SelectKBest to find the optimal number of features. Through using GridSearchCV, different number of k values were tried and the one that gives the highest value was selected according to performance metric.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys  =  True)
labels, features = targetFeatureSplit(data)
pipe1 = Pipeline([
('select_features', SelectKBest(chi2)),
('classify', DecisionTreeClassifier(random_state=0))
])
pipe2 = Pipeline([
('select_features', SelectKBest(chi2)),
('classify', AdaBoostClassifier(random_state=0))
])
pipe3 = Pipeline([
('select_features', SelectKBest(chi2)),
('classify', RandomForestClassifier(random_state=0))
])
pipe4 = Pipeline([
('select_features', SelectKBest(chi2)),
('classify', KNeighborsClassifier())
])

n_features = np.arange(1, len(features_list))
param_grid = [
{'select_features__k': n_features}
]

clf = GridSearchCV(pipe1, param_grid  = param_grid, scoring  =  "f1", cv  =  10)
clf.fit(features, labels)
print  "For DecisionTree: " , clf.best_params_

clf = GridSearchCV(pipe2, param_grid  = param_grid, scoring  =  "f1", cv  =  10)
clf.fit(features, labels)
print  "For AdaBoost: " , clf.best_params_

clf = GridSearchCV(pipe3, param_grid  = param_grid, scoring  =  "f1", cv  =  10)
clf.fit(features, labels)
print  "For RandomForest: " , clf.best_params_

clf = GridSearchCV(pipe4, param_grid  = param_grid, scoring  =  "f1", cv  =  10)
clf.fit(features, labels)
print  "For KNN: " , clf.best_params_
```

```
For DecisionTree:  {'select_features__k': 9}
For AdaBoost:  {'select_features__k': 18}
For RandomForest:  {'select_features__k': 16}
For KNN:  {'select_features__k': 2}

```

## Algorithm Selection
Because our dataset is small, running the models took short period of time so I tried as much as different number of algorithms.
Our dataset is imbalanced thus to evaluate performances of models, evaluating them with accuracy metric would be misleading. Therefore, I used different metrics in order to evaluate quality of the classifier.  [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) and [F1 scoring](https://en.wikipedia.org/wiki/F1_score) metrics are considered.

```python
clf1 = Pipeline([
('select_features', SelectKBest(k=9)),
('classify', DecisionTreeClassifier(random_state=0))
])
clf2 = Pipeline([
('select_features', SelectKBest(k=18)),
('classify', AdaBoostClassifier(random_state=0))
])
clf3 = Pipeline([
('select_features', SelectKBest(k=16)),
('classify', RandomForestClassifier(random_state=0))
])
clf4 = Pipeline([
('select_features', SelectKBest(k=2)),
('classify', KNeighborsClassifier())
])

print  "############### DecisionTree ###############"
dump_classifier_and_data(clf1, my_dataset, features_list)
tester.main()
print  "############### DecisionTree ###############"

print  "############### AdaBoost ###############"
dump_classifier_and_data(clf2, my_dataset, features_list)
tester.main()
print  "############### AdaBoost ###############"

print  "############### RandomForest ###############"
dump_classifier_and_data(clf3, my_dataset, features_list)
tester.main()
print  "############### RandomForest ###############"

print  "############### KNN ###############"
dump_classifier_and_data(clf4, my_dataset, features_list)
tester.main()
print  "############### KNN ###############"
```
```
############### DecisionTree ###############
Pipeline(memory=None,
     steps=[('select_features', SelectKBest(k=9, score_func=<function f_classif at 0x7fef6b026de8>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best'))])
	Accuracy: 0.91660	Precision: 0.68066	Recall: 0.70550	F1: 0.69286	F2: 0.70039
	Total predictions: 15000	True positives: 1411	False positives:  662	False negatives:  589	True negatives: 12338

############### DecisionTree ###############
############### AdaBoost ###############
Pipeline(memory=None,
     steps=[('select_features', SelectKBest(k=18, score_func=<function f_classif at 0x7fef6b026de8>)), ('classify', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=0))])
	Accuracy: 0.90933	Precision: 0.71136	Recall: 0.53850	F1: 0.61298	F2: 0.56601
	Total predictions: 15000	True positives: 1077	False positives:  437	False negatives:  923	True negatives: 12563

############### AdaBoost ###############
############### RandomForest ###############
Pipeline(memory=None,
     steps=[('select_features', SelectKBest(k=16, score_func=<function f_classif at 0x7fef6b026de8>)), ('classify', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_...estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False))])
	Accuracy: 0.91080	Precision: 0.80762	Recall: 0.43450	F1: 0.56502	F2: 0.47874
	Total predictions: 15000	True positives:  869	False positives:  207	False negatives: 1131	True negatives: 12793

############### RandomForest ###############
############### KNN ###############
Pipeline(memory=None,
     steps=[('select_features', SelectKBest(k=2, score_func=<function f_classif at 0x7fef6b026de8>)), ('classify', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform'))])
	Accuracy: 0.91287	Precision: 0.76350	Recall: 0.50200	F1: 0.60573	F2: 0.53892
	Total predictions: 15000	True positives: 1004	False positives:  311	False negatives:  996	True negatives: 12689

############### KNN ###############
```

DecisionTreeClassifier performs best according to f1 performance metric. RandomForest performs best if precision metric considered. I chose f1 performance metric because f1 performance metric considers both precision and recall. Also for the enron case, precision could be more important than the other metrics if we don't want to misidentify some people as poi. For the next step I chose the DecisionTree and RandomForest algorithms which gave the best score for those metrics.

## Tuning Algorithm & Evaluation
Until this point we run our algorithms with default parameters but the performance of the models can be improved by tuning the parameters. For this purpose, I used GridSearchCV. This takes grid of parameter and tries combination of these parameters and gives the combination which leads to the best result. I tried to tune "criterion", "max_depth", "min_samples_split" and "min_samples_leaf" parameters for DecisionTree algorithm. 
```python
pipe = Pipeline([
('select_features', SelectKBest(k=9)),
('classify', DecisionTreeClassifier(random_state=0))
])
param_grid = [{
"classify__criterion" : ['gini', 'entropy'],
"classify__min_samples_split" : [2, 4, 6, 8, 10],
"classify__min_samples_leaf" : [1, 2, 3],
"classify__max_depth" : [None, 2, 4, 6, 8, 10]
}]
clf = GridSearchCV(pipe, param_grid  = param_grid, scoring  =  "f1", cv  =  10)
clf.fit(features, labels)
print  "For DecisionTree: " , clf.best_params_
```
```
For DecisionTree:  {'classify__min_samples_split': 6, 'classify__min_samples_leaf': 2, 'classify__criterion': 'gini', 'classify__max_depth': None}
```
Running the model according to best parameters found by GridSearchCV.
```python
clf = Pipeline([
('select_features', SelectKBest(k=9)),
('classify', DecisionTreeClassifier(criterion="gini", min_samples_split=6, min_samples_leaf=2, max_depth=None))
])
print  "############### DecisionTree ###############"
dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()
print  "############### DecisionTree ###############"
```

```
############### DecisionTree ###############
Pipeline(memory=None,
     steps=[('select_features', SelectKBest(k=9, score_func=<function f_classif at 0x7fe3be956de8>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=6,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best'))])
	Accuracy: 0.92213	Precision: 0.71987	Recall: 0.68100	F1: 0.69990	F2: 0.68844
	Total predictions: 15000	True positives: 1362	False positives:  530	False negatives:  638	True negatives: 12470

############### DecisionTree ###############
```

Also i tuned the RandomForest algorithm with finding the best configuration of "criterion", "n_estimators", "max_depth", "min_samples_split" and "min_samples_leaf" parameters. 

```python
pipe = Pipeline([
('select_features', SelectKBest(k=16)),
('classify', RandomForestClassifier(random_state=0))
])
param_grid = [{
"classify__criterion" : ['gini', 'entropy'],
"classify__n_estimators" : [10, 25, 50],
"classify__min_samples_split" : [2, 3, 4, 5],
"classify__min_samples_leaf" : [1, 2, 3],
"classify__max_depth" : [None, 6, 8, 10, 12]
}]
clf = GridSearchCV(pipe, param_grid  = param_grid, scoring  =  "f1", cv  =  10)
clf.fit(features, labels)
print  "For RandomForest: " , clf.best_params_
```
```
For RandomForest:  {'classify__min_samples_split': 3, 'classify__n_estimators': 50, 'classify__min_samples_leaf': 1, 'classify__criterion': 'gini', 'classify__max_depth': None}
```
Running the model according to best parameters found by GridSearchCV.
```python
clf = Pipeline([
('select_features', SelectKBest(k=16)),
('classify', RandomForestClassifier(criterion="gini", n_estimators=50, min_samples_split=3, min_samples_leaf=1, max_depth=None, random_state=0))
])
print  "############### RandomForest ###############"
dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()
print  "############### RandomForest ###############"
```
```
############### RandomForest ###############
Pipeline(memory=None,
     steps=[('select_features', SelectKBest(k=16, score_func=<function f_classif at 0x7fe21a588de8>)), ('classify', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_...estimators=50, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False))])
	Accuracy: 0.91913	Precision: 0.85103	Recall: 0.47700	F1: 0.61134	F2: 0.52297
	Total predictions: 15000	True positives:  954	False positives:  167	False negatives: 1046	True negatives: 12833

############### RandomForest ###############
```

At the end of the tuning section, results didn't changed much. As a final decision I chose the DecisionTreeClassifier model because it has the highest f1 score.
For the evaluation I used "tester.py" which is provided by Udacity.

## Conclusion
The goal set by Udacity was to achieve scores at least 0.3 for precision and recall. I got scores for precision 0.71987, recall 0.68100, f1 0.69900 and accuracy 0.92213. Which means the persons I labeled as pois are actually pois with 71.987 percentage. Also model could identify %68.100 of the pois correctly. This result was the best one i could achieve. 

## Prerequisites
numpy, pandas, sklearn

## Usage
```
python poi_id.py
python tester.py
```
