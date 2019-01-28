'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    train_data = np.append(train_data, np.zeros((10, 64)), axis=0)
    train_data = np.append(train_data, np.ones((10,64)), axis=0)
    train_labels = np.append(train_labels, np.arange(10).reshape((1,10)))
    train_labels = np.append(train_labels, np.arange(10).reshape((1,10)))
    for i in range(10):
        X = train_data[np.where(train_labels==i)].T
        n = X.shape[1]
        eta[i] = np.array([np.count_nonzero(X[j])/float(n) for j in range(X.shape[0])])
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    imgs = []
    for i in range(10):
        img_i = class_images[i]
        # ...
        imgs.append(img_i.reshape((8,8)))
    output = np.concatenate(imgs, 1)
    plt.imshow(output, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(10):
        for j in range(64):
            coin = np.random.uniform(0,1)
            if coin > 1-eta[i,j]:
                generated_data[i,j] = 1
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    n = bin_digits.shape[0]
    result = np.zeros((n,10))
    for i in range(n):
        digit = bin_digits[i]
        ones_indices = np.where(digit==1)
        zeros_indices = np.where(digit==0)
        assert ones_indices[0].shape[0] + zeros_indices[0].shape[0] == 64
        assert np.count_nonzero(digit[zeros_indices]) == 0
        assert np.count_nonzero(digit[ones_indices]) == ones_indices[0].shape[0]
        for j in range(10):
            result[i,j] = sum(np.log(eta[j][ones_indices])) + sum(np.log(1-eta[j][zeros_indices]))
    return result

def gen_likelihood(bin_digits, eta):
    '''
    Comput the generative likelihood:
        p(x|y, eta)
        
    Should return an n x 10 numpy array
    '''
    n = bin_digits.shape[0]
    result = np.zeros((n,10))
    for i in range(n):
        digit = bin_digits[i]
        ones_indices = np.where(digit==1)
        zeros_indices = np.where(digit==0)
        for j in range(10):
            result[i,j] = np.prod(eta[j][ones_indices]) * np.prod(1-eta[j][zeros_indices])
    return result

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = bin_digits.shape[0]
    denom = np.array([0.1*sum(probs) for probs in gen_likelihood(bin_digits, eta)])
    c = np.log(0.1)
    numer = generative_likelihood(bin_digits, eta)
    result = np.zeros((n,10))
    for i in range(n):
        for j in range(10):
            result[i,j] = numer[i,j] + c - np.log(denom[i])
    return result

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    int_labels = np.array([int(i) for i in labels])
    return np.average(cond_likelihood[np.arange(bin_digits.shape[0]), int_labels])

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    result = cond_likelihood.argmax(axis=1)
    return result    

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    plot_images(eta)

    # Evaluation

    generate_new_data(eta)
    avg_train = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_test = avg_conditional_likelihood(test_data, test_labels, eta)
    print('The average conditional likelihood of train data is ' + \
          str(avg_train) + ', and the exponential of it is ' + str(np.e**avg_train))
    print('The average conditional likelihood of test data is ' + \
          str(avg_test) + ', and the exponential of it is ' + str(np.e**avg_test))    
    train_accuracy = 1-np.count_nonzero((classify_data(\
        train_data, eta)-train_labels))/float(train_data.shape[0])
    test_accuracy = 1-np.count_nonzero((classify_data(\
        test_data, eta)-test_labels))/float(test_data.shape[0])    
    print('The accuracy of most likely posterior class on train data is ' + \
          str(train_accuracy))
    print('The accuracy of most likely posterior class on test data is ' + \
          str(test_accuracy))      

if __name__ == '__main__':
    main()
