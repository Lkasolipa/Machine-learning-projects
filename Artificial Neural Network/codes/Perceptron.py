######################################################################
###    Install Perceptron and use it to solve simple data          ###
###     A          B          C              Result                ###
###     0          0          0                  0                 ###
###     0          1          0                  0                 ###
###     1          0          0                  1                 ###
###     1          1          1                  1                 ###
######################################################################

%matplotlib inline
import matplotlib.pyplot as plt # For visualizing the error
import numpy as np # For initializing the matrix-vector representation

#The input matrix X[3x4]
X = np.array([[0, 0, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 1, 1]])
#The true output vector y[4x1]
y = np.array([[0, 0, 1, 1]]).T

######################################################################

class Perceptron(object):
    def __init__(self, eta = 0.01, epochs = 50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        """
        Train Perceptron
        
        Parameters
        ----------
        X : numpy_array
            The input matrix
            (In this exercise, X have 4 training samples and 3 features)
        y : numpy_array
            The true output
            (In this exercise, y is a vector 4x1 with 4 labels)
            
        Returns
        ----------
        self : object
        """
        np.random.seed(16)
        self.weight_ = np.random.uniform(-1, 1, 1 + X.shape[1])
        self.error_ = []
        
        for _ in range(self.epochs):
            error = 0
            for xi, yi in zip(X, y):
                ### print self.predict(xi)
                delta = self.eta * (yi - self.predict(xi))
                
                self.weight_[0] += delta
                self.weight_[1:] += delta * xi
                
                error += delta
            ### print
            self.error_.append(error)
        return self
        
    def net_input(self, X):
        """ Caculate the net input: z = w.T * x """
        return np.dot(X, self.weight_[1:]) + self.weight_[0]
    
    def predict(self, X):
        """ Mapping the net input using the Unit step function """
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        
####################################################################

#Train Perceptron with initial X and y
pct = Perceptron()
pct.train(X, y)

#Show the weight of Perceptron after learning
print 'The weight of Perceptron after learning is: '
print pct.weight_
print 'The output of Perceptron after learning is: '
print pct.predict(X)


#Plot the error every epoch
plt.plot(range(len(pct.error_)), pct.error_)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
