import pandas as pd
import numpy as np
import time
import csv

SUBMISSION_HEADER = ['Id', 'class']

def open_pickled_file(file):
    with open(file, 'rb') as f:
        data = pd.read_pickle(file)
    return np.asarray(data)

def export_predictions(y_pred):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = './prediction-{}.csv'.format(timestr)
    file = open(filename, 'w+', newline ='')
    with file:    
        writer = csv.writer(file)
        writer.writerow(SUBMISSION_HEADER)
        writer.writerows(enumerate(y_pred))
    print('Predictions exported to {}'.format(filename))
	
def show_image(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()

def check_dataset(X, y, count=1):
    randomlist = random.sample(range(len(X)), count)
    for i in randomlist:
        show_image(X[i])
        print(y[i])
		
def predict_and_export(clf, X):
    print('Predicting..')
    y_pred = clf.predict(X)
    print('Exporting predictions..')
    export_predictions(y_pred)
    print('Export completed')  