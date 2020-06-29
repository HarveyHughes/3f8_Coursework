import numpy as np
import matplotlib.pyplot as plt

# Load data
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

b1=np.array([[1],[1],[1]])
lrate = 0.001

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def plot_data_internal(X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()

##
# x: input to the logistic function
#
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# w: current parameter values
#
def compute_average_ll(X, y, w):
    output_prob = logistic(np.dot(X, w))
    return np.mean(y * np.log(output_prob)
                   + (1 - y) * np.log(1.0 - output_prob))

##
# ll: 1d array with the average likelihood per data point, for each training
# step. The dimension of this array should be equal to the number of training
# steps.
#

def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()

##
# X: 2d array with input features at which to compute predictions.
#
# (uses parameter vector w which is defined outside the function's scope)
#

def predict_for_plot(x):
    x_tilde = np.concatenate((x, np.ones((x.shape[ 0 ], 1 ))), 1)
    return logistic(np.dot(x_tilde, w))

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# predict: function that recives as input a feature matrix and returns a 1d
#          vector with the probability of class 1.

def plot_predictive_distribution(X, y, predict):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_predict = np.concatenate((xx.ravel().reshape((-1, 1)),
                                yy.ravel().reshape((-1, 1))), 1)
    Z = predict(X_predict)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()

##
# l: hyper-parameter for the width of the Gaussian basis functions
# Z: location of the Gaussian basis functions
# X: points at which to evaluate the basis functions

def expand_inputs(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)

##
# x: 2d array with input features at which to compute the predictions
# using the feature expansion
#
# (uses parameter vector w and the 2d array X with the centers of the basis
# functions for the feature expansion, which are defined outside the function's
# scope)
#

def predict_for_plot_expanded_features(x):
    x_expanded = expand_inputs(l, x, trainx)
    x_tilde = np.concatenate((x_expanded, np.ones((x_expanded.shape[ 0 ], 1 ))), 1)
    return logistic(np.dot(x_tilde, w))

def split_data(x,y,n): #x is the data, n is the number to be in the training set
    trainx= x[0:n-1,:]
    trainy=y[0:n-1]
    testx=x[n:1000,:]
    testy=y[n:1000]
    return trainx,testx,trainy,testy

def gradient_ascent(beta,x_tilde,y):
    sigma = 1/(1+np.exp(-np.dot(x_tilde, beta))) # creates a vextor of sigma values
    gradient=np.matmul((y-sigma.T),x_tilde).T   #creates a gradient vector
    return beta + lrate*gradient

def train_data(X,y,beta,n):
    aveLike=np.zeros(n)
    for i in range(n):
        beta = gradient_ascent(beta,X,y)
        aveLike[i]=compute_average_ll(X,y,beta)
    plot_ll(aveLike)
    return beta

def flip_numbers(x):
    return -x+1

def compute_confusion(testxtilde,testy,w):
    test_prob = logistic(np.dot(testxtilde, w))
    ytilde = test_prob //0.5
    confusion = np.array([[0,0],[0,0]])
    confusion[0,0] = np.dot(flip_numbers(testy),flip_numbers(ytilde)) # guess = 0 and value is 0
    confusion[0,1] = np.dot(flip_numbers(testy),ytilde) #guess = 1 , true = o
    confusion[1,0] = np.dot(testy,flip_numbers(ytilde))# guess = 0 , true value = 1
    confusion[1,1] = np.dot(testy,ytilde) #guess = 1 and true value = 1
    confusion = confusion / testy.shape[0]
    return confusion

def ll_per_point(X,y,w):
    output_prob = logistic(np.dot(X, w))
    p1 = np.log(output_prob)
    p0 = np.log(1.0 - output_prob)
    ll=np.multiply(y.T,p1.T).T + np.multiply((1-y).T,p0.T).T
    return ll


plot_data(X,y)

##part d
n=800
trainx,testx,trainy,testy=split_data(X,y,n)
trainxtilde = np.concatenate((trainx, np.ones((trainx.shape[0], 1))), 1)
testxtilde=np.concatenate((testx, np.ones((testx.shape[0], 1))), 1)
xtilde=np.concatenate((X, np.ones((X.shape[0], 1))), 1)

w=train_data(trainxtilde,trainy,b1,50)
print(w)

plot_predictive_distribution(trainx,trainy,predict_for_plot)




##pt e
train_ll= ll_per_point(trainxtilde,trainy,w)
test_ll=ll_per_point(testxtilde,testy,w)
train_ll_list=[train_ll]
test_ll_list=[test_ll]
c=compute_confusion(testxtilde,testy,w)
print(c)

##part f
l=10
b2=np.ones(trainx.shape[0]+1)*0.1
lrates=[0.0001,0.015,0.011]
steps = [200,600,1000]
for i in range(3): ##need to increate learning rate
    l = l/10
    lrate=lrates[i]
    testx_expanded = expand_inputs(l, testx, trainx)
    testxtilde = np.concatenate((testx_expanded, np.ones((testx_expanded.shape[0], 1))), 1)
    x_expanded = expand_inputs(l, trainx, trainx)
    x_tilde = np.concatenate((x_expanded, np.ones((x_expanded.shape[0], 1))), 1)
    w = train_data(x_tilde, trainy, b2, steps[i])  ##with l = 0.01 weights of magnitude 4 each therfore likely to overfit
    plot_predictive_distribution(trainx, trainy, predict_for_plot_expanded_features)
    c=compute_confusion(testxtilde,testy,w)
    print(c)
    train_ll = ll_per_point(x_tilde, trainy, w)
    test_ll = ll_per_point(testxtilde, testy, w)
    train_ll_list.append(train_ll)
    test_ll_list.append(test_ll)
    labels=['Linear','l=1','l=0.1', 'l=0.01']

def plot_ll_per_point(x,labels,title):
    position = np.arange(0,x[0].shape[0],1)
    for i in range (4) :
        plt.scatter(position,x[i], label = labels[i],s=1)
    plt.xlabel('Datapoint')
    plt.ylabel('Log-Likelihood')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.show()

plot_ll_per_point(train_ll_list,labels,"Training Data")
plot_ll_per_point(test_ll_list,labels,"Test Data")


def plot_data_ll(X, y,ll,labels,ll_min,title):
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        col = ll[i].reshape((y.shape[0],))
        plt.scatter(X[y == 0, 0], X[y == 0, 1],marker ='o', label = 'Class 1', c=col[y == 0], cmap='viridis' , vmin=ll_min, vmax=0)
        plt.scatter(X[y == 1, 0], X[y == 1, 1] ,marker = '+', label='Class 2', c=col[y == 1], cmap='viridis',vmin=ll_min, vmax=0)
        plt.title(labels[i])
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
        plt.colorbar(label='Log Likelihood')
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(title)
    plt.show()


plot_data_ll(trainx,trainy,train_ll_list,labels,-3,'Training Data')
plot_data_ll(testx,testy,test_ll_list,labels,-6 ,'Test Data')