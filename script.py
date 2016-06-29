import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import cPickle as pickle
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

"""
Function to compute theta for all training data rows
"""
def computeTheta(initialWeights,train_data_row):
    return sigmoid(np.dot(np.transpose(initialWeights), train_data_row))

"""
Function to compute {ln P(y|w)}
Parameters:
    outputLabels  : the matrix y
    train_data    : the matrix X (training data)
    initialWeights: matrix W (weight vector)
    theta_n       : theta_n matrix (see assignment specification)
Return:
    Value of Log-Likelihood for all training data
"""
def computeLogProbabilities(outputLabels,train_data,initialWeights,theta_n):

    n_data = train_data.shape[0] #no of training rows
    logLikelihood = 0.0
    i = 0

    log_theta_n     = np.zeros([n_data,1],dtype=float)
    theta_n_minus_1 = np.zeros([n_data,1],dtype=float)
    log_theta_n_minus_1 = np.zeros([n_data,1],dtype=float)
    theta_n_minus_1 = (1.0 - theta_n)
    log_theta_n     = np.log(theta_n)
    log_theta_n_minus_1=np.log(theta_n_minus_1)

    logLikelihood=np.dot(np.transpose(outputLabels),log_theta_n)+np.dot(np.transpose(1-outputLabels),log_theta_n_minus_1)
    """
    for i in range(n_data):
        logLikelihood = logLikelihood + np.dot(outputLabels[i],log_theta_n[i]) + np.dot((1 - outputLabels[i]), log_theta_n_minus_1[i]);
    """
    return logLikelihood

"""
Function to compute error_grad
Parameters:
    outputLabels  : the matrix y
    train_data    : the matrix X (training data)
    theta_n       : theta_n matrix (see assignment specification)
Return:
    Gradient of error function
"""
def computeErrorGrad(outputLabels,train_data,theta_n):
    n_data     = train_data.shape[0]
    error_grad = np.zeros((train_data.shape[1]+1, 1))
    P = 0.0
    P = 1.0/float(n_data)
    error_grad = (np.dot(P, np.dot(np.transpose(train_data),(theta_n - outputLabels))))
    return error_grad[:,0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0.0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias = np.ones((n_data,1))

    """
    Concatenating the bias to the training data
    """
    train_data = np.concatenate( (bias,train_data),axis=1)

    """
    Generate theta_n matrix
    """
    theta_n= np.zeros([n_data,1],dtype=float)
    W_transposeX=np.dot(train_data,initialWeights)

    for i in range(n_data):
        theta_n[i] = sigmoid(W_transposeX[i])

    """
    Compute log-likelihood value
    """
    logLikelihood = 0.0
    logLikelihood = computeLogProbabilities(labeli,train_data,initialWeights,theta_n)
    #print logLikelihood
    error         = (((-1.0) * float(logLikelihood)) / n_data)
    #print('error ',error)
    error_grad    = computeErrorGrad(labeli,train_data,theta_n)
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #print "blrPredict data ",data.shape
    """
    Add the bias term at the beginning
    """
    n_data = data.shape[0]
    bias = np.ones((n_data,1))
    """
    Concatenating the bias to the training data
    """
    data = np.concatenate( (bias,data),axis=1)
    outputs = np.zeros([n_data,W.shape[1]],dtype=float)
    outputs = np.dot(data,W)
    #print (outputs[0])
    i = 0
    for i in range(n_data):
        label[i][0]  = np.argmax(outputs[i],axis=0)
    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 10 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, Y = args

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    initialWeights_b = params.reshape(n_feature+1,10)
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias = np.ones((n_data,1))
    numerator = np.zeros([n_data,n_feature + 1],dtype=float)
    denominator = np.zeros([n_data,1],dtype=float)

    """
    Concatenating the bias to the training data
    """
    train_data = np.concatenate( (bias,train_data),axis=1)
    #print('train_data = ',train_data.shape)

    theta_nk= np.zeros([n_data,n_feature + 1],dtype=float)

    denominator = (np.sum(np.exp(np.dot(train_data,initialWeights_b)),axis=1)).reshape(n_data,1)
    numerator = np.exp(np.dot(train_data,initialWeights_b))

    theta_nk = np.divide(numerator,denominator)
    log_theta_nk = np.log(theta_nk)

    err = np.sum(np.sum(np.multiply(Y,log_theta_nk),0))
    P = 1.0/float(n_data)

    error = -1.0*P*err
    #print('error_mlr = ',error)

    theta_minus_y_transpose=np.transpose(theta_nk-Y)
    error_grad=(np.dot(P,np.transpose(np.dot(theta_minus_y_transpose,train_data)))).flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    """
    Add the bias term at the beginning
    """
    n_data = data.shape[0]
    bias = np.ones((n_data,1))
    """
    Concatenate the bias to the training data
    """
    data = np.concatenate( (bias,data),axis=1)
    outputs = np.zeros([n_data,W.shape[1]],dtype=float)
    outputs = np.dot(data,W)
    #print (outputs[0])
    i = 0
    for i in range(n_data):
        label[i][0]  = np.argmax(outputs[i],axis=0)
    return label

"""
Multiply function for SVN
"""
def multiples(m,count):
    arr = np.zeros(count,)
    for i in range(1,count):
        arr[i] = i*m
    arr[0] = 1
    return arr

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
    #print " W before ",W[i]

with open('params.pickle', 'wb') as f1:
    pickle.dump(W, f1)

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
print('--------------Linear Kernel-------------------')
clf = SVC(kernel="linear")
clf.fit(train_data, train_label.reshape(train_label.shape[0],))
print('\n Training set Accuracy:' + str(100 * np.mean((np.array([clf.predict(train_data)]) == train_label.T).astype(float))) + '%')
print('\n Validation set Accuracy:' + str(100 * np.mean((np.array([clf.predict(validation_data)]) == validation_label.T).astype(float))) + '%')
print('\n Testing set Accuracy:' + str(100 * np.mean((np.array([clf.predict(test_data)]) == test_label.T).astype(float))) + '%')

print('\n--------------RBF Kernel - gamma=1-------------------')
clf = SVC(kernel="rbf",gamma=1.0)
clf.fit(train_data, train_label.reshape(train_label.shape[0],))
print('\n Training set Accuracy:' + str(100 * np.mean((np.array([clf.predict(train_data)]) == train_label.T).astype(float))) + '%')
print('\n Validation set Accuracy:' + str(100 * np.mean((np.array([clf.predict(validation_data)]) == validation_label.T).astype(float))) + '%')
print('\n Testing set Accuracy:' + str(100 * np.mean((np.array([clf.predict(test_data)]) == test_label.T).astype(float))) + '%')

print('\n--------------RBF Kernel - gamma=auto-------------------')
clf = SVC(kernel="rbf",gamma="auto")
clf.fit(train_data, train_label.reshape(train_label.shape[0],))
print('\n Training set Accuracy:' + str(100 * np.mean((np.array([clf.predict(train_data)]) == train_label.T).astype(float))) + '%')
print('\n Validation set Accuracy:' + str(100 * np.mean((np.array([clf.predict(validation_data)]) == validation_label.T).astype(float))) + '%')
print('\n Testing set Accuracy:' + str(100 * np.mean((np.array([clf.predict(test_data)]) == test_label.T).astype(float))) + '%')

count = 11
arr = multiples(10.0,count)
training = np.zeros(count)
validation = np.zeros(count)
testing = np.zeros(count)
k = 0
for i in arr:
    print('\n--------------RBF Kernel - C='+str(i)+'-------------------')
    clf = SVC(C=i, kernel="rbf",gamma="auto")
    clf.fit(train_data, train_label.reshape(train_label.shape[0],))
    training[k] = 100 * np.mean((np.array([clf.predict(train_data)]) == train_label.T).astype(float))
    print('\n Training set Accuracy:' + str(training[k]) + '%')
    validation[k] = 100 * np.mean((np.array([clf.predict(validation_data)]) == validation_label.T).astype(float))
    print('\n Validation set Accuracy:' + str(validation[k]) + '%')
    testing[k] = 100 * np.mean((np.array([clf.predict(test_data)]) == test_label.T).astype(float))
    print('\n Testing set Accuracy:' + str(testing[k]) + '%')
    k = k+1

plt.plot(arr,training)
plt.plot(arr,validation)
plt.plot(arr,testing)
plt.legend(('Training Set Accuracy','Validation set Accuracy', 'Testing set Accuracy'))
plt.show()



"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

with open('params_bonus.pickle', 'wb') as f2:
    pickle.dump(W_b, f2)

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
