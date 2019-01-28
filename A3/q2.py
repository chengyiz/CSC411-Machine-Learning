import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = self.beta * self.vel - self.lr * grad
        return params + self.vel


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        pred = np.dot(X, self.w)
        minus = pred * y
        loss = 1- minus
        loss[loss<0] = 0
        return loss

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        loss = self.hinge_loss(X,y)
        products = -(X.T * y).T
        products[loss==0] = np.zeros(X.shape[1])
        no_bias = self.w
        no_bias[0] = 0
        return no_bias + self.c / float(X.shape[0]) * np.sum(products, axis=0)       

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return np.array([1 if np.dot(self.w, x) > 0 else -1 for x in X])

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        w_history.append(optimizer.update_params(w_history[-1], func_grad(w_history[-1])))
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    svm = SVM(penalty, train_data.shape[1])
    bs = BatchSampler(train_data, train_targets, batchsize)
    
    for i in range(iters):
        batch_data, batch_target = bs.get_batch()
        svm.w = optimizer.update_params(svm.w, svm.grad(batch_data, batch_target))
    return svm

def helper(beta):
    print('='*80)
    print('Beta = '+ str(beta))
    optimizer = GDOptimizer(0.05, beta)
    svm = optimize_svm(train_data, train_targets, 1.0, optimizer, 100, 500)
    pred_train = svm.classify(train_data)
    pred_test = svm.classify(test_data)
    train_loss = sum(svm.hinge_loss(train_data, train_targets)) / float(train_data.shape[0])
    test_loss = sum(svm.hinge_loss(test_data, test_targets)) / float(test_data.shape[0])
    print('training accuracy is {}'.format((pred_train == train_targets).mean()))
    print('test accuracy is {}'.format((pred_test == test_targets).mean()))
    print('average training hinge loss is {}'.format(train_loss))
    print('average test hinge loss is {}'.format(test_loss))
    
    plt.imshow(svm.w[1:].reshape((28,28)), cmap='gray')
    title = 'beta={}'.format(beta)
    plt.title(title)
    plt.show()
    
    
    
if __name__ == '__main__':
    gdopt0 = GDOptimizer(1.0)
    gdopt09 = GDOptimizer(1.0, 0.9)
    params0 = optimize_test_function(gdopt0)
    params09 = optimize_test_function(gdopt09)
        
    plt.plot(params0,'.r')
    plt.plot(params09,'.b')
    plt.legend(('beta=0', 'beta=0.9'))
    plt.show()
    
    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.concatenate((np.ones((train_data.shape[0], 1)), train_data), axis=1)
    test_data = np.concatenate((np.ones((test_data.shape[0], 1)), test_data), axis=1)
    
    helper(0)
    helper(0.1)