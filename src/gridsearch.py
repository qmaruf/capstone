from datetime import datetime
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import math
import numpy as np
import pandas as pd
import time


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['day'] = train.sample_time.apply(lambda x: x.split()[0].split('-')[2])
test['day'] = test.sample_time.apply(lambda x: x.split()[0].split('-')[2])

train['hour'] = train.sample_time.apply(lambda x: x.split()[1].split(':')[0])
test['hour'] = test.sample_time.apply(lambda x: x.split()[1].split(':')[0])

train['week_day'] = train['day'].apply(lambda x: int(x)%7)
test['week_day'] = test['day'].apply(lambda x: int(x)%7)

train.drop('sample_time', axis=1, inplace=True)
test.drop('sample_time', axis=1, inplace=True)



for f in train.columns:
    if train[f].dtype == 'object':        
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


cols_to_remove = []

for f in train.columns:
	corr = train.cpu_01_busy.corr(train[f])
	if math.isnan(corr) or corr == 0:
		cols_to_remove.append(f)

train.drop(cols_to_remove, axis=1, inplace=True)
test.drop(cols_to_remove, axis=1, inplace=True)


y = np.array(train.cpu_01_busy)

train.drop('cpu_01_busy', axis=1, inplace=True)
X = np.array(train)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

rfr =  RandomForestRegressor()
etr = ExtraTreesRegressor()
dtr = DecisionTreeRegressor()

tuned_parameters = [{'n_estimators': [200, 250, 300], 'verbose': [2], 'max_depth': [40, 50, 60]}]

def scorer(estimator, X, y):
	y_pred = estimator.predict(X)	
	score = math.sqrt(mean_squared_error(y, y_pred))
	return score

for obj in [dtr, rfr, etr]:
	clf = GridSearchCV(obj, tuned_parameters, cv=5, n_jobs=-1,scoring=scorer)
	clf.fit(X_train, y_train)

	print("Training report")
	print clf.best_params_
	print clf.grid_scores_
	print clf.best_estimator_
	print clf.best_score_

	for params, mean_score, scores in clf.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))


	print("Testing report")
	y_true, y_pred = y_test, clf.predict(X_test)
	score = math.sqrt(mean_squared_error(y_true, y_pred))
	print score

	print '\n\n\n\n'