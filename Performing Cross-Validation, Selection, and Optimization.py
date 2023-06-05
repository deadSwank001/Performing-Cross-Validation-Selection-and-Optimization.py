# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:23:18 2023

@author: swank
"""

#Pondering the Problem of Fitting a Model
#Dividing between training, test and validation

from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target
print(X.shape, y.shape)
---------------------------------------------------------------------------
"""ImportError                               Traceback (most recent call last)
Cell In[2], line 1
----> 1 from sklearn.datasets import load_boston
      2 boston = load_boston()
      3 X, y = boston.data, boston.target

File C:\ProgramData\anaconda3\lib\site-packages\sklearn\datasets\__init__.py:156, in __getattr__(name)
    105 if name == "load_boston":
    106     msg = textwrap.dedent(
    107         """
    108         `load_boston` has been removed from scikit-learn since version 1.2.
   (...)
    154         """
    155     )
--> 156     raise ImportError(msg)
    157 try:
    158     return globals()[name]

ImportError: 
`load_boston` has been removed from scikit-learn since version 1.2.

The Boston housing prices dataset has an ethical problem: as
investigated in [1], the authors of this dataset engineered a
non-invertible variable "B" assuming that racial self-segregation had a
positive impact on house prices [2]. Furthermore the goal of the
research that led to the creation of this dataset was to study the
impact of air quality but it did not give adequate demonstration of the
validity of this assumption.

The scikit-learn maintainers therefore strongly discourage the use of
this dataset unless the purpose of the code is to study and educate
about ethical issues in data science and machine learning.

In this special case, you can fetch the dataset from the original
source::

    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

Alternative datasets include the California housing dataset and the
Ames housing dataset. You can load the datasets as follows::

    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()

for the California housing dataset and::

    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name="house_prices", as_frame=True)

for the Ames housing dataset.

[1] M Carlisle.
"Racist data destruction?"
<https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>

[2] Harrison Jr, David, and Daniel L. Rubinfeld.
"Hedonic housing prices and the demand for clean air."
Journal of environmental economics and management 5.1 (1978): 81-102.
<https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air>"""
#######################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regression = LinearRegression()
regression.fit(X,y)

print('Mean squared error: %.2f' % mean_squared_error(
    y_true=y, y_pred=regression.predict(X)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=5)
print(X_train.shape, X_test.shape)
regression.fit(X_train,y_train)
print('Train mean squared error: %.2f' % mean_squared_error(
    y_true=y_train, y_pred=regression.predict(X_train)))
print('Test mean squared error: %.2f' % mean_squared_error(
    y_true=y_test, y_pred=regression.predict(X_test)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=6)
regression.fit(X_train,y_train)
print('Train mean squared error: %.2f' % mean_squared_error(
    y_true=y_train, y_pred=regression.predict(X_train)))
print('Test mean squared error: %.2f' % mean_squared_error(
    y_true=y_test, y_pred=regression.predict(X_test)))

#Cross-validating
#Using cross-validation on k folds

from sklearn.model_selection import cross_val_score, KFold
import numpy as np

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(regression, X, y, 
    scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
print('Folds: %i, mean squared error: %.2f std: %.2f' % 
      (len(scores),np.mean(np.abs(scores)),np.std(scores)))


#Sampling stratifications for complex data
%matplotlib inline
import pandas as pd
df = pd.DataFrame(X, columns=boston.feature_names)
df['target'] = y
df.boxplot('target', by='CHAS', return_type='axes');

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error

strata = StratifiedShuffleSplit(n_splits=3, 
                                test_size=0.35, 
                                random_state=0)
scores = list()
for train_index, test_index in strata.split(X, X[:,3]):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regression.fit(X_train, y_train)
    scores.append(mean_squared_error(y_true=y_test, 
                       y_pred=regression.predict(X_test)))
print('%i folds cv mean squared error: %.2f std: %.2f' % 
      (len(scores),np.mean(np.abs(scores)),np.std(scores)))
##########################################################################
###   ^^^ ALL errors above ---> Boston Dataset Corruption

#Selecting Variables Like a Pro

#Selecting on univariate measures

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

Selector_f = SelectPercentile(f_regression, percentile=25)
Selector_f.fit(X, y)
for n,s in zip(boston.feature_names,Selector_f.scores_):
    print('F-score: %3.2f\t for feature %s ' % (s,n))

#Using a greedy search

from sklearn.feature_selection import RFECV

selector = RFECV(estimator=regression, 
                 cv=10, 
                 scoring='neg_mean_squared_error')
selector.fit(X, y)
print("Optimal number of features : %d" 
      % selector.n_features_)
print(boston.feature_names[selector.support_])

#Pumping up your hyper-parameters
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

#Implementing a grid search

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, 
      weights='uniform', metric= 'minkowski', p=2)
grid = {'n_neighbors': range(1,11), 
        'weights': ['uniform', 'distance'], 'p': [1,2]}
print ('Number of tested models: %i' 
       % np.prod([len(grid[element]) for element in grid]))
score_metric = 'accuracy'

from sklearn.model_selection import cross_val_score

print('Baseline with default parameters: %.3f' 
      % np.mean(cross_val_score(classifier, X, y, 
                cv=10, scoring=score_metric, n_jobs=1)))

from sklearn.model_selection import GridSearchCV

search = GridSearchCV(estimator=classifier, 
                      param_grid=grid, 
                      scoring=score_metric, 
                      n_jobs=1, 
                      refit=True, 
                      return_train_score=True, 
                      cv=10)
search.fit(X,y)

print('Best parameters: %s' % search.best_params_)
print('CV Accuracy of best parameters: %.3f' % 
      search.best_score_)
print(search.cv_results_)

from sklearn.model_selection import validation_curve

model = KNeighborsClassifier(weights='uniform', 
                             metric= 'minkowski', p=1)
train, test = validation_curve(model, X, y, 
                               param_name='n_neighbors', 
                               param_range=range(1, 11), 
                               cv=10, scoring='accuracy', 
                               n_jobs=1)

import matplotlib.pyplot as plt

mean_train  = np.mean(train,axis=1)
mean_test   = np.mean(test,axis=1)
plt.plot(range(1,11), mean_train,'ro--', label='Training')
plt.plot(range(1,11), mean_test,'bD-.', label='CV')
plt.grid()
plt.xlabel('Number of neighbors')
plt.ylabel('accuracy')
plt.legend(loc='upper right', numpoints= 1)
plt.show()
​
#Trying a randomized search
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(estimator=classifier, 
                    param_distributions=grid, n_iter=10, 
    scoring=score_metric, n_jobs=1, refit=True, cv=10, )
random_search.fit(X, y)
print('Best parameters: %s' % random_search.best_params_)
print('CV Accuracy of best parameters: %.3f' % 
      random_search.best_score_)
