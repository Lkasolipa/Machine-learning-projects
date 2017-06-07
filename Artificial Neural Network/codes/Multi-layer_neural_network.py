### We will set the name of class is the same name with
### Neural Network for classification of scikit-learn library
class MLPClassifier(object):
    def __init__(self, hidden_neurons=100, random_state=None,
                 epochs = 1000, minibatches=10, shuffle=True,
                 eta=0.001, alpha=0.0, constant_value=0.0,
                 l1=0.0, l2=0.0):
        self.hidden_neurons = hidden_neurons
        self.random_state = np.random.seed(random_state)
        self.epochs = epochs
        self.minibatches = minibatches
        self.shuffle = shuffle
        self.eta = eta
        self.alpha = alpha
        self.constant_value=constant_value
        self.l1 = l1
        self.l2 = l2
    
    def fit(self, X, y):
        # Initialize cost attribute to track the cost in every epoch
        self.cost_ = []
        
        # Initialize the weight attributes
        self.w1_, self.w2_ = self.init_weights(X, y)
        
        # Encode label train using one-hot representation
        y_encode = self.encode_onehot(y)
        
        # Initialize momentum term
        prev_Delta2 = np.zeros((self.w2_.shape[0], self.w2_.shape[1]))
        prev_Delta1 = np.zeros((self.w1_.shape[0], self.w1_.shape[1]))
        
        for epoch in range(self.epochs):
            
            #Print current epoch / total epoch
            sys.stderr.write('\rEpoch: %d/%d' % (epoch+1, self.epochs))
            sys.stderr.flush()
            
            # Initialize Adaptive learning rate
            self.eta = self.eta / (1 + self.constant_value*epoch)
            
            # Shuffle the dataset before training
            if self.shuffle:
                index = np.random.permutation(y.shape[0])
                X, y_encode = X[index], y_encode[:, index]

            # Split data into minibatches
            minibatch = np.array_split(range(y_encode.shape[1]), self.minibatches)
            
            #Training on every minibatches
            for mini in minibatch:
                #Step 1: Feed-forward
                x1, a2, a3 = self.feed_forward(X[mini], self.w1_, self.w2_)
                
                #Step 2: Calculate cost function
                cost = self.calculate_cost(a3, y_encode[:, mini])
                self.cost_.append(cost)
                
                #Step 3: Back-propagation
                grad2, grad1 = self.backpropagation(x1, a2, a3,
                                                    y_encode[:,mini],
                                                    self.w1_, self.w2_)
                
                Delta2 = -self.eta * grad2
                Delta1 = -self.eta * grad1
                
                #Step 4: Plus the momentum alpha
                Delta2 += self.alpha * prev_Delta2
                Delta1 += self.alpha * prev_Delta1
                prev_Delta2, prev_Delta1 = Delta2, Delta1
                
                #Step 5: Update the weight
                self.w2_ += Delta2
                self.w1_ += Delta1
        
        return self
    
    def init_weights(self, X, y):
        w1 = np.random.uniform(-1, 1, (self.hidden_neurons, X.shape[1] + 1))
        w2 = np.random.uniform(-1, 1, (len(np.unique(y)), self.hidden_neurons + 1))
        
        return w1, w2
    
    def feed_forward(self, X, w1, w2, predict=False):
        x1 = self.add_bias_unit(X, axis_=1)
        
        z2 = w1.dot(x1.T)
        a2 = self.activation_function(z2)
        a2 = self.add_bias_unit(a2, axis_=0)
        
        z3 = w2.dot(a2)
        a3 = self.activation_function(z3)
        
        # Just for prediction
        if predict:
            return a3
        
        return x1, a2, a3
    
    def calculate_cost(self, a3, y):
        tmp = np.sum(((-y) * np.log(a3)) - ((1.0 - y) * np.log(1.0 - a3)))
        cost = tmp + (self.l1_regularization(self.w1_, self.w2_, self.l1) + \
                      self.l2_regularization(self.w1_, self.w2_, self.l2))
        
        return cost
    
    def backpropagation(self, x1, a2, a3, y, w1, w2):
        ### Calculate the delta and gradient descent
        delta3 = a3 - y
        
        delta2 = w2.T.dot(delta3) * self.derivative_of_sigmoid_function(a2)
        delta2 = delta2[1:,:]
        
        grad2 = delta3.dot(a2.T)
        grad1 = delta2.dot(x1)
        
        ### Plus extra regularization term
        grad2[:,1:] += w2[:,1:]*(self.l1 + self.l2)
        grad1[:,1:] += w1[:,1:]*(self.l1 + self.l2)
        
        return grad2, grad1
    
    def add_bias_unit(self, a, axis_):
        if axis_ == 1:
            a_new = np.concatenate((np.ones((len(a), 1)), a), axis=1)
        elif axis_ == 0:
            a_new = np.concatenate((np.ones((1, len(a[0,:]))), a), axis=0)
        else:
            print 'Axis must be "column(1)" or "row(0)"'
        return a_new
    
    def activation_function(self, z):
        return expit(z)
    
    def derivative_of_sigmoid_function(self, a):
        return a * (1.0 - a)
    
    def encode_onehot(self, y):
        y_encode = np.zeros((len(np.unique(y)), y.shape[0]))
        for index, value in enumerate(y):
            y_encode[value][index] = 1
        return y_encode
        
    def l1_regularization(self, w1, w2, lamda):
        return (lamda / 2.0) * (((w1[:,1:])**2).sum() + ((w2[:,1:])**2).sum())
    
    def l2_regularization(self, w1, w2, lamda):
        return (lamda / 2.0) * ((abs(w1[:,1:])).sum() + (abs(w2[:,1:])).sum())
    
    def predict(self, X):
        a3 = self.feed_forward(X, self.w1_, self.w2_, predict=True)
        pred = np.argmax(a3, axis=0)
        return pred
