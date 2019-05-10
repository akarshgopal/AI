import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display, Math, Latex
import struct

'''
Data Format Expected : [m,n] 
                       [m,d,h,w] 
                    m = Number of examples, n = Number of features , d = depth, h = height, w = width
'''

"""
-------------------------------Data Processing---------------------------------------
"""
def normalize(X):
    # Returns (mean, std) normalized version of X
    mean = np.mean(X,axis=0, keepdims=True)
    std =  np.std(X,axis=0, keepdims=True)
    N_X = (X-mean)/(std)
    return N_X

def mat_normalize(X):
    mean = np.mean(X)
    std =  np.std(X)
    N_X = (X-mean)/(std)
    return N_X

def polynom_features(X,d):
    # (n,m) -> (n*o,m)
    #(m,n) -> (m,n*o)
    stack_stack = []
    for i in range(X.shape[1]):
        Xi = X[:,i]
        stack = []
        for j in range(1,d+1):
            stack.append(Xi**j)
        stack_stack.append(np.hstack(stack))
    PX = np.hstack(stack_stack)
    return PX

def one_hot(Y,n_class):
    # Returns an (len(Y), n_class) numpy array as the one hot representation of Y
    length = np.shape(Y)[0]
    O = np.zeros((length, n_class))
    for i in range(length):
        j = int(Y[i,0])
        O[i,j] = 1
    return O

def inv_one_hot(O):
    # Returns an (1,len(Y)) numpy array as the inverse one hot representation of Y
    n_class = np.shape(O)[1]
    length = np.shape(O)[0]
    Y = np.zeros((length,1))
    for i in range(length):
        j = np.argmax(O[i,:])
        Y[i,0] = j
    return Y

def zero_padding(img_matrix, padding):
    pad_width = [(0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])]
    padded_matrix = np.pad(img_matrix, pad_width=pad_width, mode='constant',)
    return padded_matrix
"""
-----------------------------Initialization----------------------------------------
"""
def init_matrix(n1,n2,activation):
    # Initializes an (n1,n2) numpy array with a randomization optimized for 
    #particular activations
    np.random.seed(1)
    if isinstance(activation,(sigmoid,softmax)):
        M = (np.random.randn(n1,n2) - 0.5)*np.sqrt(2./n1)
    elif isinstance(activation,(relu,leaky_relu)) :
        M = (np.random.randn(n1,n2)- 0.5)*np.sqrt(1./n1)
    elif isinstance(activation, tanh):
        M = (np.random.randn(n1,n2)- 0.5)*np.sqrt(1./(n1+n2))
    else:
    	M = (np.random.randn(n1,n2)- 0.5) 
    return M


def init_filters(K, D, kernel_size, activation):
        W = init_matrix(D*kernel_size[0]*kernel_size[1],K,activation)
        return W

"""
--------------------------------Data Loaders--------------------
"""   
def load_mnist_data(trainX_path,trainY_path,testX_path,testY_path):
    X = read_idx(trainX_path)
    X = mat_normalize(X)
    X = np.expand_dims(X,axis=1)

    Y = read_idx(trainY_path)
    Y = np.expand_dims(Y, axis=1)
    Y = one_hot(Y,10)

    X_test = read_idx(testX_path)
    X_test = mat_normalize(X_test)
    X_test = np.expand_dims(X_test,axis=1)

    Y_test = read_idx(testY_path)
    Y_test = np.expand_dims(Y_test, axis=1)
    Y_test = one_hot(Y_test,10)
    return X,Y, X_test, Y_test

"""
--------------------------------Model Metrics--------------------
"""
def model_accuracy(H,Y):
    E = (H-Y)== 0
    accuracy = E.sum()/H.shape[0]
    return accuracy

def RMSE(H,Y):
	accuracy = 1- 1/H.shape[0]*(np.sqrt(np.einsum('Mn,Mn->',(H-Y),(H-Y))))
	return accuracy.item()


"""
-----------------------------Activations ------------------------------
"""
class sigmoid:
    def activate(self,Z):
        A = 1/(1+np.exp(-Z))
        return A
    
    def diff(self,Z):
        dsig = np.multiply(self.activate(Z),(1-self.activate(Z)))
        return dsig

class relu:
    def activate(self,Z):
        A = Z*(Z>0)
        return A
    
    def diff(self,Z):
        d_rel = 1*(Z>0)
        return d_rel
    
class softmax:
    """Compute softmax values for each sets of scores in x."""
    def activate(self,Z):
        e_Z = np.exp(Z- np.max(Z,axis=1,keepdims=True))
        return e_Z / e_Z.sum(axis=1,keepdims=True)
    
    def diff(self,Z):
        return Z

class leaky_relu:
    def activate(self,Z):
        A = np.where(Z > 0, Z, Z * 0.01)
        return A
    
    def diff(self,Z):
        d_lrel = np.where(Z > 0, 1, 0.01)
        return d_lrel
    
class tanh:
    def activate(self,Z):
        A = np.tanh(Z)
        return A

    def diff(self,Z):
        d_tanh = 1 - (np.multiply(self.activate(Z),self.activate(Z)))
        return d_tanh

#dummy activation
class no_op:
	def activate(self,Z):
		return Z

	def diff(self,Z):
		return np.ones(Z.shape)

"""
----------------------------------Loss Functions ------------------
"""    
class CE_loss:
    def get_loss(self,H,Y):
        #adding  1e-21 to prevent log(0)
        L = -np.mean(np.multiply(Y,np.log(H+1e-21)))
        return L
    
    def diff(self,H,Y):
        # return dLdZ w.r.t softmax. This simplifies computation.
        n = Y.shape[1]
        dZ = 1/n*(H-Y) 
        return dZ
    
class MSE:
    def get_loss(self,H,Y):
        L = 1/(2*H.shape[0])*(np.einsum('mn,mn->',(H-Y),(H-Y)))
        return L.item()
    
    def diff(self,H,Y):
        dZ = H - Y 
        return dZ


"""        
--------------------------------Optimizers--------------------------
Passes through dataset for one iteration and performs step 
updates on the model.
"""

def SGD(batch_size,X,Y,model,lr,beta,reg_lambda=0):
    m = np.shape(X)[0]
    H = np.zeros(Y.shape)
    for i in range(0,m,batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        H[i:i+batch_size] = model.f_pass(X_batch)
        model.back_prop(X_batch, Y_batch, batch_size, reg_lambda)
        model.optim(lr,beta) 
    O = inv_one_hot(H)
    L = inv_one_hot(Y)
    tr_acc = model_accuracy(O,L)    
    return model.loss, tr_acc

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
        # half genes from X1, other half from X2
        # if number of genes n is odd, n-1 split between X1 and X2, 
        #     nth gene selected on random from either X1 or X2 
        N = X1.shape[0]
        X3 = np.ndarray(X1.shape)
        ind1 = np.random.choice(range(N), N//2, replace=False)
        for i in range(N):
            if i in ind1:
                X3[i] = X2[i]
            else:
                X3[i] = X1[i]    
        return X3

        #rand_list =  np.random.uniform(size = self.num_genes)< 0.5
        #X3 = X1*rand_list + (1-rand_list)*X2
        #return X3

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
def train(model, X, Y, X_test, Y_test, metric, n_epochs=10, \
    batch_size=4, lr=0.0003, lr_decay=1, beta=0, reg_lambda=0):
    data_size = X.shape[0]
    for e in range(n_epochs):
        #shuffle dataset
        np.random.seed(138)
        shuffle_index = np.random.permutation(data_size)
        X, Y = X[shuffle_index,:], Y[shuffle_index,:]

        #SGD with momentum
        loss, tr_acc = SGD(batch_size,X,Y,model,lr, beta, reg_lambda)
        lr = lr*lr_decay

        m = np.shape(X_test)[0]
        H = np.zeros(Y_test.shape)
        for i in range(0,m,batch_size):
            X_test_batch = X_test[i:i+batch_size]
            H[i:i+batch_size] = model.f_pass(X_test_batch)
        O = inv_one_hot(H)
        L = inv_one_hot(Y_test)
        acc = model_accuracy(O,L)
        
        plt.plot(e,tr_acc, 'bo')
        plt.plot(e,acc,'ro')
        clear_output()
        print(f"epoch:{e+1}/{n_epochs} | Loss:{loss:.4f} \
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
        self.activation = activation()
        
        self.V_dW = np.zeros(self.W.shape)
        self.V_dB = np.zeros(self.B.shape)
        
    def forward(self, A0):
        self.Z = np.einsum('ln,ml-> mn',self.W, A0) + self.B
        self.A = self.activation.activate(self.Z)
        return self.A
    
    def grad(self, dA, A0, m):
        dAdZ = self.activation.diff(self.Z)
        self.dZ = np.multiply(dA, dAdZ)
        self.dW = (1./m)*np.einsum('mn,ml->ln',self.dZ, A0)
        self.dB = (1./m)*(np.einsum('mn->n',self.dZ))
        dA_prev = np.einsum('ln,mn->ml',self.W, self.dZ) 
        return dA_prev
    
    def out_grad(self, dZ, A0, m):
        self.dZ = dZ
        self.dW = (1./m)*np.einsum('mn,ml->ln',self.dZ, A0)
        self.dB = (1./m)*(np.einsum('mn->n',self.dZ))
        dA_prev = np.einsum('ln,mn->ml',self.W, self.dZ) 
        return dA_prev
        
    def step(self, lr, beta):
        self.V_dW = (beta * self.V_dW + (1. - beta) * self.dW)
        self.V_dB = (beta * self.V_dB + (1. - beta) * self.dB)
        self.W = self.W - lr*self.V_dW
        self.B = self.B - lr*self.V_dB

class conv_layer:
    def __init__(self, kernel_size, n_channels, activation, \
        n_filters = 1, stride =[1,1],  padding = [0,0]):

        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding
        self.activation = activation()
        
        #intialize [n_filters,n_channels,kernel_size[0],kernel_size[1]] filters
        self.W = init_filters(self.n_filters, self.n_channels, \
            self.kernel_size, activation)
        self.B = init_matrix( 1, n_filters, activation)
        
        self.V_dW = np.zeros(self.W.shape)
        self.V_dB = np.zeros(self.B.shape)
        
    def get_output_shape(self, A_prev_shape):
        batch_size,D,W,H = A_prev_shape
        H_out = (H - self.kernel_size[0])//self.stride[0] + 1
        W_out = (W - self.kernel_size[1])//self.stride[1] + 1
        output_shape = (batch_size, self.n_filters, H_out, W_out)
        return output_shape
    
    def forward(self, A_prev):
        A_prev = zero_padding(A_prev, self.padding)        
        self.A_prev_shape = A_prev.shape
        self.im2cols, self.act_idx = im2col(A_prev,self.kernel_size, self.stride)
        output_shape = self.get_output_shape(A_prev.shape)
        
        self.Z = np.einsum('mfn,fk->mnk',self.im2cols,self.W) + self.B   
        self.A = self.activation.activate(self.Z)
        self.A = np.transpose(self.A,(0,2,1)).reshape(output_shape)
        return self.A
    
    def grad(self, dA):
        batch_size = dA.shape[0]
        dAdZ = self.activation.diff(self.Z)
        self.dZ = np.einsum('mjk,mjk->mjk', dA, dAdZ)
        
        self.dW = (1./batch_size)*np.einsum('mjk,mkl->jl', self.im2cols, self.dZ)
        self.dB = (1./batch_size)*np.einsum('mjk->k', self.dZ)
        
        dA_prev_cols = np.einsum('mnk,fk-> mfn',self.dZ, self.W) 

        dA_prev = np.zeros(self.A_prev_shape)        
        for i in range(self.act_idx.shape[-1]):
            dA_prev.ravel()[self.act_idx[:,:,i].ravel()] += dA_prev_cols[:,:,i].ravel()

        dA_prev = dA_prev.reshape(dA_prev.shape[0],dA_prev.shape[1],\
            dA_prev.shape[2]*dA_prev.shape[3])
        dA_prev = np.transpose(dA_prev,(0,2,1))
        
        return dA_prev
    
    def step(self, lr, beta, reg_lambda=0):
        self.V_dW = (beta * self.V_dW + (1. - beta) * self.dW)
        self.V_dB = (beta * self.V_dB + (1. - beta) * self.dB)
        self.W    =  self.W - lr*self.V_dW
        self.B    =  self.B - lr*self.V_dB


class max_pool_layer():
    def __init__(self, kernel_size, n_channels, stride =[1,1],  padding = [0,0]):

        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.stride = stride
        self.padding = padding
        
    def get_output_shape(self, A_prev_shape):
        batch_size,D,W,H = A_prev_shape
        H_out = (H - self.kernel_size[0])//self.stride[0] + 1
        W_out = (W - self.kernel_size[1])//self.stride[1] + 1
        output_shape = (batch_size, self.n_channels, H_out, W_out)
        return output_shape
    
    def forward(self, A_prev):

        A_prev = zero_padding(A_prev, self.padding)
        self.A_prev_shape = A_prev.shape
        self.im2cols, self.act_idx = im2col(A_prev, self.kernel_size, self.stride)
        
        output_shape = self.get_output_shape(A_prev.shape)
        
        self.Z = np.split(self.im2cols,self.n_channels,axis=1)
        self.Z = np.stack(self.Z, axis=1)

        self.A = np.max(self.Z,axis=2)  
        self.arg = np.argmax(self.A,axis=2)
        
        self.A = self.A.reshape(output_shape)
        return self.A
    
    def grad(self, dA):
        batch_size = dA.shape[0]
        dA = np.expand_dims(dA.transpose(0,2,1), axis=2)
        A = self.A.reshape(self.A.shape[0],self.A.shape[1],1,\
            self.A.shape[-2]*self.A.shape[-1])
        dAdZ = (self.Z - A)==0
        self.dZ = np.multiply(dA, dAdZ) 
        shape = self.dZ.shape
        self.dZ = self.dZ.reshape(shape[0],shape[1]*shape[2],shape[3])
        
        dA_prev = np.zeros(self.A_prev_shape)        
        
        for i in range(self.act_idx.shape[-1]):
            dA_prev.ravel()[self.act_idx[:,:,i].ravel()] += self.dZ[:,:,i].ravel()

        dA_prev = dA_prev.reshape(dA_prev.shape[0],dA_prev.shape[1],\
            dA_prev.shape[2]*dA_prev.shape[3])
        dA_prev = np.transpose(dA_prev,(0,2,1))
        
        return dA_prev
    
    def step(self, lr, beta, reg_lambda=0):
        return None


"""
---------------------------Regression
"""
class Linear:
    def __init__(self, X_size, Y_size, lossfn):
        self.regressor = layer(X_size, Y_size, no_op)
        self.lossfn = lossfn()
        
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
        self.lossfn = lossfn()
        
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



#
#------------  K Nearest Neighbours classifier ----------------------------

def KNN_classifier(X,Y,X_test,K):
    '''
    X.shape = (n,m)  n = number of features , m = number of training examples
    Y.shape = (1,m)
    x.shape = (n,t) t = number of test examples
    Euclidian dist only atm
    Works properly only for k < m//2
    '''
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



"""
 -----------------------------Utils-----------------------------
"""
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

