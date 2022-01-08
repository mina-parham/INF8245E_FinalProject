from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler

from data_augmentation import *

def scale(train, test):
	scaler = StandardScaler()
	scaler.fit(train)
	scaled_train = scaler.transform(train)
	scaled_test = scaler.transform(test)
	return scaled_train, scaled_test

def encode(y):
    le = LabelEncoder()
    le.fit(y)
    return le.transform(y)

def encode_to_categorical(y, num_classes=11):
	y_encoded = encode(y)
	y_encoded_categorical = to_categorical(y_encoded, num_classes)
	return y_encoded_categorical

def flatten(dataframe):
    return dataframe.reshape(dataframe.shape[0], dataframe.shape[1] * dataframe.shape[2])

# Augment data
def balance(X, y):
    augmented_data_X = np.empty((0, X.shape[1], X.shape[2], X.shape[3]))
    augmented_data_Y = []
    freq = np.unique(y, return_counts=True)
    for index in range(len(freq[0])):
        label_index = np.where(y == freq[0][index])
        if freq[1][index] < 700:
            augmented_data = selective_augment(X[label_index], y[label_index], 3)
        elif freq[1][index] < 1000:
            augmented_data = selective_augment(X[label_index], y[label_index])
        else:
            augmented_data = (X[label_index], y[label_index])

        augmented_data_X = np.concatenate((augmented_data_X, augmented_data[0]))
        augmented_data_Y = np.concatenate((augmented_data_Y, augmented_data[1]))
    return augmented_data_X, augmented_data_Y