import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#The test_size parameter specifies that 20% of the data will be used for testing. 
# The shuffle parameter specifies that the data will be shuffled before splitting, and 
# The stratify parameter ensures that the data is split in a stratified manner, preserving the class distribution.

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

#This line calculates the accuracy of the model's predictions by comparing y_predict (the predicted labels) 
# with y_test (the actual labels) using the accuracy_score function. The result is stored in the score variable.

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
