import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
from sklearn.metrics import accuracy_score
style.use("ggplot")
red_training_data = [];red_training_labels = []
red_test_data = []; red_test_labels = []
blue_training_data = []; blue_training_labels = []
blue_test_data = []; blue_test_labels = []
red_blue_training_data = []; red_blue_test_data = []
red_blue_training_labels = []; red_blue_test_labels = []
red_blue_color = []
# data points in axis [0, 50] to [50, 0] are excluded due to low/high range conflict
#Create RED training and test data sets
#ALL RED DATA
for i in range(1, 151):
    a = np.random.randint(0, 50) # range 0-49
    b = np.random.randint(0, 50-(a-4)) # range 0-49
    red_training_data.append([a, b])
for i in range(0, 125, 5):
    remove = red_training_data.pop(i)
    red_test_data.append(remove)
#create red training_labels
for i in range(0, 125):
    red_training_labels.append(1)
#create red test_labels
for i in range(0, 25):
    red_test_labels.append(1)
#ALL BLUE DATA
for i in range(1, 151):
    a = np.random.randint(0, 50) # range 1-50
    b = np.random.randint((50-(a+4)), 51) # range 1-50
    blue_training_data.append([a, b])
for i in range(0, 125, 5):
    remove = blue_training_data.pop(i)
    blue_test_data.append(remove)
#create blue training_labels
for i in range(0, 125):
    blue_training_labels.append(2)
#create blue test_labels
for i in range(0, 25):
    blue_test_labels.append(2)
#JOIN RED-BLUE data sets
red_training_data.extend(blue_training_data)
red_blue_training_data = red_training_data # create final training data set
red_test_data.extend(blue_test_data)
red_blue_test_data = red_test_data # create final test data set
red_training_labels.extend(blue_training_labels)
red_blue_training_labels = red_training_labels # create final training labels set
red_test_labels.extend(blue_test_labels)
red_blue_test_labels = red_test_labels # create final test labels set
# make all data np.arrays
red_blue_training_data = np.array(red_blue_training_data)
red_blue_test_data = np.array(red_blue_test_data)
red_blue_training_labels = np.array(red_blue_training_labels)
red_blue_test_labels = np.array(red_blue_test_labels)
#SET PLOT DATA colors
for i in range(0, 125):
    red_blue_color.append('red')
for i in range(0, 125):
    red_blue_color.append('blue')
# Run ML SVM classification
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(red_blue_training_data, red_blue_training_labels)
w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(0, 51, num=250)
yy = a*xx - clf.intercept_[0]/w[1]
h0 = plt.plot(xx, yy, 'k-')
plt.scatter(red_blue_training_data[:, 0], red_blue_training_data[:, 1], s = 50, color = red_blue_color[:])
plt.scatter(red_blue_test_data[:, 0], red_blue_test_data[:, 1], s = 50, color = 'green')
plt.title('Support Vector Machine Classification')
plt.xlabel('Red/Blue Training data points')
plt.ylabel('Green Test data points')
#RUN Classification prediction and accuracy
pred = clf.predict(red_blue_test_data)
acc = accuracy_score(pred, red_blue_test_labels)
print('')
print("SVM run on the test data set shows an accuracy = %.2f" % acc)
plt.show()