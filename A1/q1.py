from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import tabulate

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(32, 8))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.plot(X.T[i,], y, '.')
        plt.xlabel(features[i])
        plt.ylabel('target')
        plt.title('target w.r.t '+ features[i])
        
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    
    a = np.dot(X.T, X)
    b = np.dot(X.T, Y)
    
    x = np.linalg.solve(a, b)
    return x

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)
    #TODO: Split data into train and test
    # find the number of test data
    test_num = int(round(0.2 * X.shape[0])) #101
    
    # make copies of datas
    train_X = X.copy()
    train_y = y.copy()
    
    # randomly choose indice among original data to use as test data
    tests = np.random.choice(X.shape[0], test_num, replace=False)
    test_X = [train_X[t_index] for t_index in tests]
    test_y = [train_y[t_index] for t_index in tests]
    
    trains = [i for i in range(X.shape[0]) if i not in tests]
    train_X = [train_X[t_index] for t_index in trains]
    train_y = [train_y[t_index] for t_index in trains]
    
    # add bias
    train_X = np.insert(train_X, 0, 1, axis=1)
    test_X = np.insert(test_X, 0, 1, axis=1)
    

    # Fit regression model
    w = fit_regression(train_X, train_y)
    features = np.insert(features, 0, 'bias')
    
    print(sum(w[1:])/13)
    
    for i in range(len(w)):
        print w[i], features[i]

    # Compute fitted values, MSE, etc.
    test_y_hat = np.dot(test_X, w)
    
    mse, mae, huber = 0, 0, 0
    for i in range(len(test_y_hat)):
        mae += abs(test_y_hat[i] - test_y[i])
        mse += (test_y_hat[i] - test_y[i]) ** 2
        if abs(test_y_hat[i] - test_y[i]) <= 1:
            huber += 0.5 * (test_y_hat[i] - test_y[i]) ** 2
        else:
            huber += abs(test_y_hat[i] - test_y[i]) - 0.5
    mse /= len(test_y_hat)
    mae /= len(test_y_hat)
    
    print 'mse: ', mse, '\n',\
          'mae: ', mae, '\n',\
          'huber loss: ', huber


if __name__ == "__main__":
    np.random.seed(7)
    main()

