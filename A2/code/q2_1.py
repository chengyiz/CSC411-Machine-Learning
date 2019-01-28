'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = None
        dists = self.l2_distance(test_point)
        # find the indices of the k nearest neighbours
        k_neighbours = dists.argsort()[:k]
        # find the label of the k-nn
        labels = self.train_labels[k_neighbours]
        # count the occurence of labels
        counts = [np.count_nonzero(labels==i) for i in range(10)]
        # find the label with max occurence
        maxoccur = max(counts)
        if counts.count(maxoccur) > 1: # tie exists
            # redo the classification by (k-1)-NN until 1NN
            digit = self.query_knn(test_point, k-1)
        else:
            digit = counts.index(maxoccur)
        #digit = counts.index(maxoccur)
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10)
    avg_accus = []
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        accus = []
        for train_index, test_index in kf.split(train_data):
            trains, tests  = train_data[train_index], train_data[test_index]
            label_train, label_test = train_labels[train_index], train_labels[test_index]
            knn = KNearestNeighbor(trains, label_train)
            accus.append(classification_accuracy(knn, k, tests, label_test))
        avg_accus.append(sum(accus)/float(len(accus)))
    k = avg_accus.index(max(avg_accus))+1
    return avg_accus, k

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    results = [knn.query_knn(eval_data[i],k) == eval_labels[i] for i in range(eval_labels.shape[0])]
    return sum(results) / float(len(results))

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    accu_train_1 = classification_accuracy(knn, 1, train_data, train_labels)
    accu_train_15 = classification_accuracy(knn, 15, train_data,train_labels)    
    accu_test_1 = classification_accuracy(knn, 1, test_data, test_labels)
    accu_test_15 = classification_accuracy(knn, 15, test_data,test_labels)
    print(accu_train_1, accu_test_1,accu_train_15,accu_test_15)
    accus, k = cross_validation(train_data, train_labels)
    for idx, val in enumerate(accus):
        print('Average accuracy for k='+str(idx+1)+' is '+str(val))
    print('The optimal value of k is '+str(k))
    train_accu = classification_accuracy(knn, k, train_data, train_labels)
    print('The training classification accuracy for k={0} is {1}'.format(k, train_accu))
    test_accu = classification_accuracy(knn, k, test_data, test_labels)
    print('The test classification accuracy for k={0} is {1}'.format(k, test_accu))


if __name__ == '__main__':
    main()