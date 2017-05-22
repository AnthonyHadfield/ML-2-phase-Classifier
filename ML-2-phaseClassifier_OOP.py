# THIS IS ML-2-phase Classifier in OOP
import numpy as np
import matplotlib.pyplot as plt
rtrd = [] # red_training_data
rtrl = [] # red_training_labels
rted = [] # red_test_data
rtel = [] # red_test_labels
btrd = [] # blue_training_data
btrl = [] # blue_training_labels
bted = [] # blue_test_data
btel = [] # blue_test_labels
rbtrd = [] # red_blue_training_data
rbtrl = [] # red_blue_training_labels
rbted = [] # red_blue_test_data
rbtel = [] # red_blue_test_labels
red_blue_color = []
# data points in axis [0, 50] to [50, 0] are excluded due to low/high range conflict
class Classifier:
    def red_data(self,rtrd, rtrl, rted, rtel):                  #Create RED data
        self.rtrd = rtrd
        self.rtrl = rtrl
        self.rted = rted
        self.rtel = rtel
        for i in range(1, 151):
            a = np.random.randint(0, 50) # range 0-49
            b = np.random.randint(0, 50-(a-4)) # range 0-49
            rtrd.append([a, b])
        for i in range(0, 125, 5):                                #Split out RED TEST data set
            remove = rtrd.pop(i)
            rted.append(remove)
        for i in range(0, 125):                                   #Create RED labels
            rtrl.append(1)
        for i in range(0, 125):
            rtel.append(1)
        return rtrd, rtrl, rted, rtel
    def blue_data(self, btrd, btrl, bted, btel):                   #Create BLUE data
        self.btrd = btrd
        self.btrl = btrl
        self.bted = bted
        self.btel = btel
        for i in range(1, 150):
            a = np.random.randint(1, 51) # range 1-50
            b = np.random.randint((51-(a+4)), 51) # range 1-50
            btrd.append([a, b])
        for i in range(0, 125, 5):                              #Split out BLUE TEST data set
            remove = btrd.pop(i)
            bted.append(remove)
        for i in range(0, 125):                                 #Create BLUE labels
            btrl.append(2)
        for i in range(0, 125):
            btel.append(2)
        return btrd, btrl, bted, btel
    def red_blue_data(self):                                    # JOIN red/blue data sets
        Classifier.red_data(self, rtrd, rtrl, rted, rtel)
        Classifier.blue_data(self, btrd, btrl, bted, btel)
        rtrd.extend(btrd)
        rbtrd = rtrd                                            # create final training data set
        rted.extend(bted)
        rbted = rted                                            # create final test data set
        rtrl.extend(btrl)
        rbtrl = rtrl                                            # create final training labels set
        rtel.extend(btel)
        rbtel = rtel                                            # create final test labels set
        rbtrd = np.array(rbtrd)                                 #Set to arrays
        rbted = np.array(rbted)
        rbtrl = np.array(rbtrl)
        rbtel = np.array(rbtel)
        for i in range(0, 125):                                # Add in multiple color streams
            red_blue_color.append('red')
        for i in range(0, 125):
            red_blue_color.append('blue')
        plt.figure()                                            # PLOT DATA
        plt.scatter(rbtrd[:, 0], rbtrd[:, 1], s = 50, color = red_blue_color[:])
        plt.scatter(rbted[:, 0], rbted[:, 1], s = 50, color = 'green')
        plt.title('Green Test data points')
        plt.xlabel('Red/Blue Training data points')
        plt.show()
data = Classifier()
print(data.red_blue_data())                                     # Print data chart