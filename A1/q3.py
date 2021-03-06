import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50
K = 500

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    # gradient = 2(XtXw - 2Xty)/n
    n = X.shape[0]
    Xw = np.dot(X, w)
    XtXw = np.dot(X.T, Xw)
    
    Xty = np.dot(X.T, y)
    
    return 2 * (XtXw - Xty) / float(n)

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    sum_grad = np.zeros(X.shape[1])
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        sum_grad += batch_grad
    avg_grad = sum_grad / K
    grad = lin_reg_gradient(X,y,w)
    cossimilarity = cosine_similarity(avg_grad, grad)
    print('cosine similarity:', cossimilarity)
    dist = np.linalg.norm(avg_grad-grad)
    print('squared distance:', dist)
    
    n = X.shape[1]
    j = np.random.choice(n)
    
    ms = range(1,401)
    
    sam_vars = []
    
    for m in ms:
        samples = []
        for i in range(K):
            X_b, y_b = batch_sampler.get_batch()
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            samples.append(batch_grad[j])
        mean = np.mean(samples)
        sv = sum([(sample - mean)**2 for sample in samples]) / (K-1)
        sam_vars.append(sv)
    
    print(sam_vars)
    logm = [np.log(m) for m in ms]
    logsv = [np.log(sv) for sv in sam_vars]
    
    plt.plot(logm, logsv)
    plt.ylabel('log sigma')
    plt.xlabel('log m')
    plt.show()


if __name__ == '__main__':
    np.random.seed(5)   
    main()
