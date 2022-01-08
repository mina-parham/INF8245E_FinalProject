from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 0

# baseline - SVM
def train_svm(X_train, y_train, random_state=RANDOM_STATE, verbosity=True):
    clf = SVC(random_state=random_state, verbose=verbosity)
    clf.fit(X_train_flattened, y_train)
    return clf

# baseline - logistic regression
def train_logistic_regression(X_train, y_train, random_state=RANDOM_STATE, max_iter=100, verbosity=1):
    hyperparameters = []
    clf = LogisticRegression(max_iter=max_iter, verbose=verbosity, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf