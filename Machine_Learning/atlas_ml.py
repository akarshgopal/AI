import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display, Math, Latex
"""
 -----------------------------Utils-----------------------------
"""

def polynom_features(X,d):
    # (n,m) -> (n*o,m)
    stack_stack = []
    for i in range(X.shape[0]):
        Xi = X[i,:]
        stack = []
        for j in range(1,d+1):
            stack.append(Xi**j)
        stack_stack.append(np.vstack(stack))
    PX = np.vstack(stack_stack)
    return PX

def one_hot(Y,n_class):
    # Returns an (n_class,len(Y)) numpy array as the one hot representation of Y
    length = np.shape(Y)[1]
    O = np.zeros((n_class,length))
    for i in range(length):
        j = int(Y[0,i])
        O[j,i] = 1
    return O

def inv_one_hot(O):
    # Returns an (1,len(Y)) numpy array as the inverse one hot representation of Y
    n_class = np.shape(O)[0]
    length = np.shape(O)[1]
    Y = np.zeros((1,length))
    for i in range(length):
        j = np.argmax(O[:,i])
        Y[0,i] = j
    return Y

def normalize(X):
    # Returns (mean, std) normalized version of X
    mean = np.mean(X)
    std =  np.std(X)
    N_X = (X-mean)/(std)
    return N_X

def init_matrix(n1,n2,activation):
    # Initializes an (n2,n1) numpy array with a randomization optimized for 
    #particular activations
    
    if activation in [sigmoid,softmax]:
        M = np.random.randn(n2,n1)*np.sqrt(2./n1)
    elif activation in [relu,leaky_relu] :
        M = np.random.randn(n2,n1)*np.sqrt(1./n1)
    elif activation == tanh:
        M = np.random.randn(n2,n1)*np.sqrt(1./(n1+n2))
    else:
    	M = np.random.randn(n2,n1) 
    return M

"""
--------------------------------Model Metrics--------------------
"""
def model_accuracy(H,Y):
    n = np.shape(H)[1]
    err = 0
    O = inv_one_hot(H)
    L = inv_one_hot(Y)
    for i in range(n):
        if O[0,i]!=L[0,i]:
            err += 1
    accuracy = (1 - err/n)
    return accuracy

def RMSE(H,Y):
	n = H.shape[1]
	accuracy = 1- 1/n*(np.sqrt(np.dot((H-Y),(H-Y).T)))
	return accuracy.item()
"""
-----------------------------Activations ------------------------------
"""
class sigmoid:
    def activate(Z):
        A = 1/(1+np.exp(-Z))
        return A
    
    def diff(self,Z):
        dsig = np.multiply(self.activate(Z),(1-self.activate(Z)))
        return dsig
 
class relu:
    def activate(Z):
        A = Z*(Z>0)
        return A
    
    def diff(self,Z):
        d_rel = 1*(Z>0)
        return d_rel
    
class softmax:
    """Compute softmax values for each sets of scores in x."""
    def activate(Z):
        e_Z = np.exp(Z- np.max(Z,axis=0))
        return e_Z / e_Z.sum(axis=0)
    
    def diff(Z):
        return Z

    
# wrong implementation of leaky
class leaky_relu:
    def activate(Z):
        A = Z if (Z.all()>0.001*Z.all()) else 0.001*Z
        return A
    
    def diff(self,Z):
        d_lrel = 1 if (Z.all()>0.001*Z.all()) else 0.001
        return d_lrel
    
class tanh:
    def activate(Z):
        A = np.tanh(Z)
        return A

    def diff(self,Z):
        d_tanh = 1 - (np.multiply(self.activate(Z),self.activate(Z)))
        return d_tanh

#dummy activation
class no_op:
	def activate(Z):
		return Z

	def diff(self,Z):
		return 1

"""
----------------------------------Loss Functions ------------------
"""    
class CE_loss:
    def get_loss(H,Y):
        L = -np.mean(np.multiply(Y,np.log(H)))
        return L
    
    def diff(H,Y):
        dZ = H - Y 
        return dZ
    
class MSE:
    def get_loss(H,Y):
        L = 1/(2*H.shape[1])*(np.dot((H-Y),(H-Y).T))
        return L.item()
    
    def diff(H,Y):
        dZ = H-Y
        return dZ

"""
--------------------------------Optimizers--------------------------
Passes through dataset for one iteration and performs step 
updates on the model.
"""

def SGD(batch_size,X,Y,model,lr,beta,reg_lambda=0):
    m = np.shape(X)[1]
    for i in range(0,m,batch_size):
        X_batch = X[:,i:i+batch_size]
        Y_batch = Y[:,i:i+batch_size]
        model.f_pass(X_batch)
        model.back_prop(X_batch, Y_batch, batch_size, reg_lambda)
        model.optim(lr,beta)
    return model.loss

# Genetic optimizer
class genetic_optimizer:
    
    def __init__(self, population_size, num_genes, fitness_fn, \
        pass_thres=0.25, mutation_prob=0.2, mutation_ampl=1):
        self.population_size = population_size
        self.num_genes = num_genes
        self.fitness_fn = fitness_fn
        self.population = np.random.uniform(low = -4.0, high = 4.0,\
         size = (self.num_genes, self.population_size))
        self.mutation_prob = mutation_prob
        self.mutation_ampl = mutation_ampl
        self.fitness = self.fitness_fn(self.population)
        self.pass_thres = pass_thres
                
    def crossover(self, X1, X2):
        rand_list =  np.random.uniform(size = self.num_genes)< 0.5
        X3 = X1*rand_list + (1-rand_list)*X2
        return X3

    def mutate(self, X):
        rand_list =  np.random.uniform(size = self.num_genes)< self.mutation_prob
        rand_list2 =  np.random.uniform(-1, 1, size = self.num_genes)
        X2 = X + self.mutation_ampl * rand_list2 * rand_list
        return X2

    def next_generation(self):
        N = int(self.population_size*self.pass_thres)
        k = N
        while k < self.population_size:
            if (k < 2*N):
                self.population[:,k] = self.crossover(self.population[:,k - N],\
                 self.population[:,k - N + 1])
            
            if (k >= 2*N) and (k < 3*N):
                m = int(4 * N * np.random.uniform())
                self.population[:,k] = self.crossover(self.population[:,k - 2 *N],\
                 self.population[:,m])
            
            if (k >= 3*N):
                m1 = int( 4*N * np.random.uniform())
                m2 = int( 4*N * np.random.uniform())
                self.population[:,k] = self.crossover(self.population[:,m1],\
                 self.population[:,m2])
            
            self.population[:,k] = self.mutate(self.population[:,k])
            k = k + 1

    def optimize(self, num_generations = 10000):
        self.fitness = np.ndarray([self.population_size,1])
        for generation in range(num_generations):
            self.fitness = self.fitness_fn(self.population)
            qq = np.argsort(self.fitness)
            qq = qq[::-1]          
            self.population = self.population[:,qq]
            self.fitness = self.fitness[qq]  
            if (generation % (num_generations/10) == 0): 
                print(self.fitness[0]) 
                print(self.population[0])
            self.next_generation()
        return(self.fitness[0], self.population[:,0])

"""
------------------------------Trainer----------------------------
"""
def train(model, X, Y, X_test, Y_test, metric, n_epochs=100, \
    batch_size=4, lr=0.01, lr_decay=1, beta=0, reg_lambda=0):
    data_size = X.shape[1]
    for e in range(n_epochs):
        #shuffle dataset
        np.random.seed(138)
        shuffle_index = np.random.permutation(data_size)
        X, Y = X[:,shuffle_index], Y[:,shuffle_index]

        #SGD with momentum
        loss = SGD(batch_size,X,Y,model,lr,beta, reg_lambda)

        lr = lr*lr_decay

        H = model.f_pass(X)
        tr_acc = metric(H,Y)

        H = model.f_pass(X_test)
        acc = metric(H,Y_test)

        plt.plot(e,tr_acc, 'bo')
        plt.plot(e,acc,'ro')
        clear_output()
        print(f"epoch:{e+1}/{n_epochs} | Loss:{loss:.4f} | \
            Train Accuracy: {tr_acc:.4f} | Test_Accuracy:{acc:.4f}")
        
    #plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.show()

"""
------------------------------Layers---------------------------------
"""
class layer:
    def __init__(self, n_prev, n_next, activation):
        self.W = init_matrix(n_prev, n_next, activation)
        self.B = init_matrix(1, n_next, activation)
        self.activation = activation
        self.V_dW = np.zeros(self.W.shape)
        self.V_dB = np.zeros(self.B.shape)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)
        
    def forward(self, A0):
        self.Z = np.dot(self.W, A0) + self.B
        self.A = self.activation.activate(self.Z)
        return self.A
    
    def grad(self, dZ, W, A0, m, reg_lambda=0):
        dA = np.dot(W.T, dZ)
        dAdZ = self.activation.diff(self.activation, self.Z)
        self.dZ = np.multiply(dA, dAdZ)
        self.dW = (1./m)*(np.dot(self.dZ, A0.T) + reg_lambda*self.W)
        self.dB = (1./m)*(np.sum(self.dZ, axis=1, keepdims=True))
    
    def out_grad(self, dZ, A0, m, reg_lambda=0):
        self.dZ = dZ
        self.dW = (1./m)*(np.dot(self.dZ, A0.T) + reg_lambda*self.W)
        self.dB = (1./m)*(np.sum(self.dZ, axis=1, keepdims=True))
        
    def step(self, lr, beta):
        self.V_dW = (beta * self.V_dW + (1. - beta) * self.dW)
        self.V_dB = (beta * self.V_dB + (1. - beta) * self.dB)
        self.W = self.W - lr*self.V_dW
        self.B = self.B - lr*self.V_dB

"""
---------------------------Regression
"""
class Linear:
    def __init__(self, X_size, Y_size, lossfn):
        self.regressor = layer(X_size, Y_size, no_op)
        self.lossfn = lossfn
        
    def f_pass(self, X):
        self.H = self.regressor.forward(X)
        return self.H
    
    def back_prop(self,X,Y, batch_size, reg_lambda):
        m = batch_size
        self.loss = self.lossfn.get_loss(self.H,Y)
        dZ = self.lossfn.diff(self.H,Y)
        self.regressor.out_grad(dZ, X, m, reg_lambda)
        
    def optim(self, lr, beta=0):
        self.regressor.step(lr,beta)


class Logistic:
    def __init__(self, X_size, Y_size, lossfn):
        self.regressor = layer(X_size, Y_size, softmax)
        self.lossfn = lossfn
        
    def f_pass(self, X):
        self.H = self.regressor.forward(X)
        return self.H
    
    def back_prop(self, X, Y, batch_size, reg_lambda):
        m = batch_size
        self.loss = self.lossfn.get_loss(self.H,Y)
        dZ = self.lossfn.diff(self.H,Y)
        self.regressor.out_grad(dZ, X, m, reg_lambda)
    
    def optim(self, lr, beta=0):
        self.regressor.step(lr,beta)



def confusion_mat(H,Y):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    return None

def precision(H,Y):
    return None

def recall(H,Y):
    return None

def F1(H,Y):
    return None


#
#--------------------------------------------------------------------------  K Nearest Neighbours classifier ----------------------------------------------------

def KNN_classifier(X,Y,X_test,K):
    # X.shape = (n,m)  n = number of features , m = number of training examples
    # Y.shape = (1,m)
    # x.shape = (n,t) t = number of test examples
    # Euclidian dist only atm
    # Works properly only for k < m//2
    
    t = X_test.shape[1]
    y = []
    
    for i in range(t):
        x_i = np.expand_dims(X_test[:,i],axis=1)

        deltaV = x_i - X

        #Euclidian dist
        dist = np.sum(deltaV*deltaV,axis=0)
        Knn_indices = np.argpartition(dist,K)[0:K]
        Knn_labels = [Y[0,j].tolist() for j in Knn_indices]
        counts = np.bincount(Knn_labels)
        #find majority label
        y_i = np.argmax(counts)
        y.append(y_i)
    
    y = np.array([y])
    
    return y
