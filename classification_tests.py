from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50).fit(training_final[:,:-1],training_final[:,-1])
print(perc(clf.score(testing_final[:,:-1],testing_final[:,-1])))

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators = 100,
                                 learning_rate = .6,
                                 max_depth = 1) \
    .fit(training_final[:,:-1],training_final[:,-1])
print(perc(clf.score(testing_final[:,:-1],testing_final[:,-1])))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 200,
                             min_samples_split = 2,
                             random_state = 0,
                             n_jobs = -1) \
    .fit(training_final[:,:-1],training_final[:,-1])
print(perc(clf.score(testing_final[:,:-1],testing_final[:,-1])))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes = (20,10)) \
    .fit(training_final[:,:-1],training_final[:,-1])
print(perc(clf.score(testing_final[:,:-1],testing_final[:,-1])))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes = (20,10)) \
    .fit(training_final[:,:-1],training_final[:,-1])
print(perc(clf.score(testing_final[:,:-1],testing_final[:,-1])))

from sklearn.gaussian_process import GaussianProcessClassifier
clf = GaussianProcessClassifier()
clf.fit(training_final[:,:-1],training_final[:,-1])
print(perc(clf.score(testing_final[:,:-1],testing_final[:,-1])))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=2)
clf.fit(training_final[:,:-1],training_final[:,-1])
print(perc(clf.score(testing_final[:,:-1],testing_final[:,-1])))