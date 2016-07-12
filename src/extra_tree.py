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


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

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



clf = ExtraTreesRegressor(n_estimators=300, max_depth=60, verbose=2, n_jobs=-1)
clf.fit(X, y)

id_test = test.Id
test.drop('Id', axis=1, inplace=True)
test = np.array(test)
start = time.clock()
predictions = clf.predict(test)
end = time.clock()
# print end -start

preds = pd.DataFrame({'Id': id_test, 'Prediction': predictions})
outfile = 'submission_%s.csv'%str(datetime.now())
preds.to_csv(outfile, index=False, columns=['Id','Prediction'])






