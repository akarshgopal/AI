import struct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display, Math, Latex


'''
Data Format Expected : [m,n] 
                       [m,d,h,w] 
                    m = Number of examples, n = Number of features , d = depth, h = height, w = width
'''

#-------------------------------Data Processing---------------------------------------
def flatten_imgs(X):
    n_img,d,h,w = X.shape
    size_arr = h*w
    return X.reshape(n_img,size_arr)

def normalize(X):
    """
    Normalize a numpy array with (mean,std) normalization w.r.t axis 0.

    Args:
        X (numpy array): The numpy array to be normalized.

    Returns:
        The (mean, std) normalized float numpy array of shape same as X. 
    
    Raises:
        No error checks included.
    """
    mean = np.mean(X,axis=0, keepdims=True)
    std =  np.std(X,axis=0, keepdims=True)
    N_X = (X-mean)/(std)
    return N_X

def mat_normalize(X):
    """
    Normalize a numpy array with (mean,std) normalization w.r.t all axes.

    Args:
        X(numpy array): The numpy array to be normalized.

    Returns:
        The (mean, std) normalized float numpy array of shape same as X. 
    
    Raises:
        No error checks included.
    """
    mean = np.mean(X)
    std =  np.std(X)
    N_X = (X-mean)/(std)
    return N_X

def polynom_features(X,d):
    """
    Adds polynomial features to a feature vector X upto degree d.
    
    Stacks d vectors of shape X horizontally, where each consecutive vector is a higher power of X.

    Args:
        X (numpy array):  The feature vector (A numpy array of shape (m,n), where m is 
           the number of examples and n is the number of features.
        d (int):  The degree of polynomial features required.
    Returns:
        A numpy float array PX which has the polynomial features with degree d, with shape (m,n*d). 
    
    Raises:
        No error checks included.
    """
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
    """
    Produces the one hot representation of Y in n classes.

    Args:
        Y (numpy array): The numpy array to be converted to one-hot representation. 
                         Should be shape (m,1) where m is number of examples. 
        n_class (int): The number of classes in the one-hot representation.

    Returns:
        A (len(Y), n_class) numpy array as the one hot representation of Y.
    
    Raises:
        No error checks included.
    """
    length = np.shape(Y)[0]
    O = np.zeros((length, n_class))
    for i in range(length):
        j = int(Y[i,0])
        O[i,j] = 1
    return O

def inv_one_hot(O):
    """
    Produces the inverse one hot representation of O from n classes.

    Args:
        O (numpy array): The numpy array of which inverse one-hot representation is to be taken. 
                         Should be shape (m, n_class) where m is number of examples. 

    Returns:
        A (m, 1) numpy array as the inverse one hot representation of O.
    
    Raises:
        No error checks included.
    """
    n_class = np.shape(O)[1]
    length = np.shape(O)[0]
    Y = np.zeros((length,1))
    for i in range(length):
        j = np.argmax(O[i,:])
        Y[i,0] = j
    return Y

def zero_padding(img_matrix, padding):
    """
    Pads img_matrix with padding height and width wise.

    Args:
        img_matrix (numpy array): The numpy array to be padded. Must be of shape (m,d,h,w),
                                  where m=number of examples, d = depth, h = height, w = width.
        padding (list or tuple of ints): The padding required. Must be of shape (hp,wp),
                                  where hp = padding height, wp = padding width.
    
    Returns:
        A (m, d, h+2hp, w+2wp) numpy array as the padded img_matrix.
    
    Raises:
        No error checks included.
    """
    pad_width = [(0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])]
    padded_matrix = np.pad(img_matrix, pad_width=pad_width, mode='constant',)
    return padded_matrix


#https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
def im2col(imgs, kernel_size, stride):
    """
    Converts a sequence of images to im2col representation.

    Args:
        imgs (numpy array): The numpy array to be padded. Must be of shape (m,d,h,w),
            where m=number of examples, d = depth, h = height, w = width.
        kernel_size (list or tuple of ints): The kernel or filter size. Must be of shape (hf,wf),
            where hf = kernel height, wf = kernel width.
        stride (list or tuple of ints): The strides required. Must be of shape (hs,ws),
            where hs = stride height, ws = stride width.
    
    Returns:
        A (m, hf*wf*d, ((h-hf)/hs+1)*((w-wf)/ws+1) ) numpy array as the im2col representation of imgs.
    
    Raises:
        No error checks included.
    """
    F = kernel_size
    batch_size, D,H,W = imgs.shape
    col_extent = (W - F[1]) + 1
    row_extent = (H - F[0]) + 1

    # Get batch block indices
    batch_idx = np.arange(batch_size)[:, None, None] * D * H * W
    # Get Starting block indices
    start_idx = np.arange(F[0])[None, :,None]*W + np.arange(F[1])
    # Generate Depth indices
    didx=H*W*np.arange(D)
    start_idx=(didx[None, :, None]+start_idx.ravel()).reshape((-1,F[0],F[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[None, :, None]*W + np.arange(col_extent)
    
    # Get all actual indices & index into input array for final output
    act_idx = (batch_idx + 
        start_idx.ravel()[None, :, None] + 
        offset_idx[:,::stride[0],::stride[1]].ravel())

    col_matrix = np.take (imgs, act_idx)
    return col_matrix, act_idx


#-----------------------------Initialization----------------------------------------


def init_matrix(n1,n2,activation):
    """
    Initializes a [n1,n2] numpy array with a randomization optimized for particular activations

    Args:
        n1 (int): Number of rows. In case of a Weight Matrix, this would be number of nodes in prev_layer.
        n2 (int): Number of columns. In case of a Weight Matrix, this would be number of nodes in next_layer. 
        activation (class name): The activation function relevant to the initialization.

    Returns:
        A [n1,n2] float numpy array initialized respectively for the activation function. 
    
    Raises:
        No error checks included.
    """
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
    """Initializes a filters or kernels for a CNN with a randomization optimized for particular activations

    Args:
        K (int): Number of filters.
        D (int): Number of channels. 
        kernel_size (list or tuple of ints): Kernel or filter size of shape (height, width).
        activation (class name): The activation function relevant to the initialization.

    Returns:
        A [D*kernel_height*kernel_width,K] float numpy array initialized respectively for the activation function. 
    
    Raises:
        No error checks included.
    """
    W = init_matrix(D*kernel_size[0]*kernel_size[1],K,activation)
    return W


#--------------------------------Data Loaders--------------------


def load_mnist_data(trainX_path,trainY_path,testX_path,testY_path):
    """Initializes a [n1,n2] numpy array with a randomization optimized for particular activations

    Args:
        trainX_path (string): The path to the train X dataset.
        trainY_path (string): The path to the train Y dataset. 
        testX_path (string): The path to the test X dataset.
        testY_path (string): The path to the test Y dataset.

    Returns:
        X,Y,X_test and Y_test respectively as numpy arrats of shape (m,d,h,w). 
    
    Raises:
        No error checks included.
    """
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

#--------------------------------Model Metrics--------------------

def model_accuracy(H,Y):
    """
    Computes the accuracy of the model by comparing output H w.r.t expected output Y.
    Accuracy is caclutated as the percentage of correct labels.

    Args:
        H (numpy array): The output, H, of the model.
        Y (numpy array): The label vector, Y, of the dataset.

    Returns:
        A float value, accuracy. 
    
    Raises:
        No error checks included.
    """    
    E = (H-Y)== 0
    accuracy = E.sum()/H.shape[0]
    return accuracy

def RMSE(H,Y):
    """
    Computes the accuracy of the model by comparing output H w.r.t expected output Y.
    Accuracy is caclutated as the root mean squared error in prediction w.r.t expected output.
    
    Args:
        H (numpy array): The output, H, of the model.
        Y (numpy array): The expected output vector, Y, of the dataset.

    Returns:
        A float value, accuracy. 
    
    Raises:
        No error checks included.
    """
    accuracy = 1/H.shape[0]*(np.sqrt(np.einsum('Mn,Mn->',(H-Y),(H-Y))))
    return accuracy


#-----------------------------Activations ------------------------------

class sigmoid:
    """
    Sigmoid activation
    """
    def activate(self,Z):
        """
        Performs activation on a numpy array, Z, and returns the same as numpy 
        array A of shape same as Z
        """
        A = 1/(1+np.exp(-Z))
        return A
    
    def diff(self,Z):
        """
        Calculates the differential of a numpy array, Z, w.r.t activation 
        and returns the same as numpy array dsig of shape same as Z
        """
        dsig = np.multiply(self.activate(Z),(1-self.activate(Z)))
        return dsig

class relu:
    """
    ReLU activation
    """
    def activate(self,Z):
        """
        Performs activation on a numpy array, Z, and returns the same as numpy 
        array A of shape same as Z
        """
        A = Z*(Z>0)
        return A
    
    def diff(self,Z):
        """
        Calculates the differential of a numpy array, Z, w.r.t activation 
        and returns the same as numpy array dsig of shape same as Z
        """
        d_rel = 1*(Z>0)
        return d_rel
    
class softmax:
    """
    Softmax activation
    """
    def activate(self,Z):
        """
        Performs activation on a numpy array, Z, and returns the same as numpy 
        array A of shape same as Z
        """
        e_Z = np.exp(Z- np.max(Z,axis=1,keepdims=True))
        return e_Z / e_Z.sum(axis=1,keepdims=True)
    
    def diff(self,Z):
        """
        Calculates the differential of a numpy array, Z, w.r.t activation 
        and returns the same as numpy array dsig of shape same as Z
        """
        return Z

class leaky_relu:
    """
    Leaky ReLU activation
    """
    def activate(self,Z):
        """
        Performs activation on a numpy array, Z, and returns the same as numpy 
        array A of shape same as Z
        """
        A = np.where(Z > 0, Z, Z * 0.01)
        return A
    
    def diff(self,Z):
        """
        Calculates the differential of a numpy array, Z, w.r.t activation 
        and returns the same as numpy array dsig of shape same as Z
        """
        d_lrel = np.where(Z > 0, 1, 0.01)
        return d_lrel
    
class tanh:
    """
    tanh activation
    """
    def activate(self,Z):
        """
        Performs activation on a numpy array, Z, and returns the same as numpy 
        array A of shape same as Z
        """
        A = np.tanh(Z)
        return A

    def diff(self,Z):
        """
        Calculates the differential of a numpy array, Z, w.r.t activation 
        and returns the same as numpy array dsig of shape same as Z
        """
        d_tanh = 1 - (np.multiply(self.activate(Z),self.activate(Z)))
        return d_tanh

#dummy activation
class no_op:
    """
    Dummy activation 
    """
    def activate(self,Z):
        """
        Performs activation on a numpy array, Z, and returns the same as numpy 
        array A of shape same as Z
        """
        return Z

    def diff(self,Z):
        """
        Calculates the differential of a numpy array, Z, w.r.t activation 
        and returns the same as numpy array dsig of shape same as Z
        """
        return np.ones(Z.shape)

"""
----------------------------------Loss Functions ------------------
"""    
class CE_loss:
    """
    The Cross-Entropy Loss Function
    """
    def get_loss(self,H,Y):
        """
        Computes the loss of the model using H, the output vector and Y, the expected output.
        """

        #adding  1e-21 to prevent log(0)
        L = -np.mean(np.multiply(Y,np.log(H+1e-21)))
        return L
    
    def diff(self,H,Y):
        """
        Returns dL/dZ w.r.t the softmax activation.
        """
        
        # return dLdZ w.r.t softmax. This simplifies computation.
        n = Y.shape[1]
        dZ = 1/n*(H-Y) 
        return dZ
    
class MSE:
    """
    The Mean-Square-Error Loss Function
    """
    def get_loss(self,H,Y):
        """
        Computes the loss of the model using H, the output vector and Y, the expected output.
        """

        L = 1/(2*H.shape[0])*(np.einsum('mn,mn->',(H-Y),(H-Y)))
        return L.item()
    
    def diff(self,H,Y):
        """
        Returns dL/dZ w.r.t the activation function.
        """

        dZ = H - Y 
        return dZ


#--------------------------------Optimizers--------------------------
#Passes through dataset for one iteration and performs step 
#updates on the model.


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


#------------------------------Trainer----------------------------

def train(model, X, Y, X_test, Y_test, metric, n_epochs=10, \
    batch_size=4, lr=0.0003, lr_decay=1, beta=0, reg_lambda=0,show_test_acc=True):
    data_size = X.shape[0]
    for e in range(n_epochs):
        #shuffle dataset
        np.random.seed(138)
        shuffle_index = np.random.permutation(data_size)
        X, Y = X[shuffle_index,:], Y[shuffle_index,:]

        #SGD with momentum
        loss, tr_acc = SGD(batch_size,X,Y,model,lr, beta, reg_lambda)
        lr = lr*lr_decay

        if show_test_acc:
            m = np.shape(X_test)[0]
            H = np.zeros(Y_test.shape)
            for i in range(0,m,batch_size):
                X_test_batch = X_test[i:i+batch_size]
                H[i:i+batch_size] = model.f_pass(X_test_batch)
            O = inv_one_hot(H)
            L = inv_one_hot(Y_test)
            acc = model_accuracy(O,L)
        else:
            acc = 0
            
        plt.plot(e,tr_acc, 'bo')
        plt.plot(e,acc,'ro')
        clear_output()
        print(f"epoch:{e+1}/{n_epochs} | Loss:{loss:.4f} \
Train Accuracy: {tr_acc:.4f} | Test_Accuracy:{acc:.4f}")
        
    #plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.show()

#------------------------------Layers---------------------------------

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


#---------------------------Regression

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




#------------  K Nearest Neighbours classifier ----------------------------

#TODO: Convert to (m,n) usage.
def KNN_classifier(X,Y,X_test,K):
    """
    Reads the MNIST dataset and returns it as anumpy array.
    Euclidian dist only atm
    Works properly only for k < m//2

    Args:
        X (numpy array): The training X dataset of shape = (n,m)  
            n = number of features , m = number of training examples
        Y (numpy array): The training Y dataset of shape = (1,m)
        X_test (numpy array): The data for which classification is to be performed. shape = (n,t)
        K (int): The number of nearest neighbours to consider
        
    Returns:
        y, the classified test examples as a numpy array of shape (1,t)  
    
    Raises:
        No error checks included.
    """ 

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


#-----------------------------Utils-----------------------------

def read_idx(filename):
    """
    Reads the MNIST dataset and returns it as anumpy array.
    Args:
        filename (string): The path to a MNIST dataset file.
    
    Returns:
        MNIST data corresponding to the file as a numpy array. 
    
    Raises:
        No error checks included.
    """    
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

