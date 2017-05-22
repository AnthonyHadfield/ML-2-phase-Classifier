import numpy as np
import matplotlib.pyplot as plt
red_training_data = [];red_training_labels = []
red_test_data = []; red_test_labels = []
blue_training_data = []; blue_training_labels = []
blue_test_data = []; blue_test_labels = []
red_blue_training_data = []; red_blue_test_data = []
red_blue_training_labels = []; red_blue_test_labels = []
red_blue_color = []
# data points in axis [0, 50] to [50, 0] are excluded
#  due to low/high range conflict
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
for i in range(0, 125):
    red_test_labels.append(1)
#ALL BLUE DATA
for i in range(1, 150):
    a = np.random.randint(1, 51) # range 1-50
    b = np.random.randint((51-(a+4)), 51) # range 1-50
    blue_training_data.append([a, b])
for i in range(0, 125, 5):
    remove = blue_training_data.pop(i)
    blue_test_data.append(remove)
#create blue training_labels
for i in range(0, 125):
    blue_training_labels.append(1)
#create blue test_labels
for i in range(0, 125):
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
plt.figure()
plt.scatter(red_blue_training_data[:, 0], red_blue_training_data[:, 1], s = 50, color = red_blue_color[:])
plt.scatter(red_blue_test_data[:, 0], red_blue_test_data[:, 1], s = 50, color = 'green')
plt.title('Green Test data points')
plt.xlabel('Red/Blue Training data points')
plt.show()