'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        X = train_data[np.where(train_labels==i)]
        mean = np.average(X, axis=0)
        means[i] = mean
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for i in range(10):
        X = train_data[np.where(train_labels==i)].T
        covs = np.zeros((64,64))
        for j in range(64):
            for k in range(64):
                covs[j][k] = cov(X[j], X[k])
        covariances[i] = covs + np.eye(64)*0.01
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    vars=[]
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...
        log_diag = np.log(cov_diag)
        vars.append(log_diag.reshape((8,8)))
    all_var=np.concatenate(vars, 1)
    plt.imshow(all_var,cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    # log(2pi^(-64/2))
    c = -32*np.log(2*np.pi)
    # log(|sigma|^(-1/2))
    d = -0.5*np.log(np.linalg.det(covariances))
    n = digits.shape[0]
    # sigma^-1
    cov_invs = np.linalg.inv(covariances)
    result = np.zeros((n,10))
    for i in range(10):
        for j in range(n):
            # x - mu
            x = digits[j]-means[i]
            # (x-mu).T*sigma^-1*(x-mu)
            product = x.dot(cov_invs[i]).dot(x)
            # log(2pi^-32) - 0.5*log(|sigma|^-1/2) - 0.5(x-mu)*sigma^-1*(x-mu)
            result[j][i] = c+d[i]-0.5*product
    return result

def gen_likelihood(digits, means, covariances):
    '''
    Compute the p(x|y,mu,Sigma)
    Should return an n x 10 numpy array
    '''
    c = (2*np.pi)**-32
    d = np.linalg.det(covariances)**(-0.5)
    n = digits.shape[0]
    cov_invs = np.linalg.inv(covariances)
    result = np.zeros((n,10))
    for i in range(10):
        for j in range(n):
            x = digits[j]-means[i]
            product = x.dot(cov_invs[i]).dot(x)
            result[j][i] = c*d[i]*np.e**(-0.5*product)
    return result

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = digits.shape[0]
    p_x = np.array([0.1 * sum(probs) for probs in gen_likelihood(digits, means, covariances)])
    denom = np.log(p_x)
    result = np.zeros((n,10))
    c = np.log(0.1)
    numer = generative_likelihood(digits, means, covariances)
    for i in range(10):
        for j in range(n):
            result[j][i] = numer[j][i] + c - denom[j]
    return result

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    total = sum([cond_likelihood[i][int(labels[i])] for i in range(digits.shape[0])])
    return total / float(digits.shape[0])

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    result = cond_likelihood.argmax(axis=1)
    return result
    

def cov(a, b):
    '''
    compute the covariance of vector a and b
    '''
    assert a.shape[0] == b.shape[0]
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    cov = (a - mean_a).dot(b-mean_b) / (a.shape[0]-1)
    return cov

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    plot_cov_diagonal(covariances)
    avg_train = avg_conditional_likelihood(train_data,train_labels, means, covariances)
    avg_test = avg_conditional_likelihood(test_data,test_labels, means, covariances)
    print('The average conditional likelihood of train data is ' + \
          str(avg_train) + ', and the exponential of it is ' + str(np.e**avg_train))
    print('The average conditional likelihood of test data is ' + \
          str(avg_test) + ', and the exponential of it is ' + str(np.e**avg_test))
    train_accuracy = 1-np.count_nonzero((classify_data(\
        train_data, means,covariances)-train_labels))/float(train_data.shape[0])
    test_accuracy = 1-np.count_nonzero((classify_data(\
        test_data, means,covariances)-test_labels))/float(test_data.shape[0])    
    print('The accuracy of most likely posterior class on train data is ' + \
          str(train_accuracy))
    print('The accuracy of most likely posterior class on test data is ' + \
          str(test_accuracy))    

if __name__ == '__main__':
    main()