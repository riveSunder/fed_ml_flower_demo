import argparse

import sklearn
import sklearn.datasets
from autograd import numpy as np
from autograd import grad
import autograd

from functools import reduce
import sklearn.metrics

import flwr as fl


def get_data(split="all"):
    
    # x, y = sklearn.datasets.load_iris(return_X_y=True)
    x, y = sklearn.datasets.load_digits(return_X_y=True)

    np.random.seed(42); np.random.shuffle(x)
    np.random.seed(42); np.random.shuffle(y)


    val_split = int(0.2 * x.shape[0])
    train_split = (x.shape[0] - val_split) // 2

    eval_x, eval_y = x[:val_split], y[:val_split]

    alice_x, alice_y = x[val_split:val_split + train_split], y[val_split:val_split + train_split]
    bob_x, bob_y = x[val_split + train_split:], y[val_split + train_split:]

    train_x, train_y = x[val_split:], y[val_split:]
    
    if split == "all":
        return train_x, train_y
    
    elif split == "alice":
        return alice_x, alice_y
    
    elif split == "bob":
        return bob_x, bob_y
    
    elif split == "val":
        return eval_x, eval_y
    
    else: 
        print("error: split must be 'all', 'alice', 'bob', or 'val'. Default is 'all'")
        return 1


def cross_entropy(target, prediction):
    
    my_target = np.zeros_like(prediction)
    
    # convert to one-hot
    for ii in range(2):
        temp = np.zeros((1,3))
        temp[:, ii] = 1 
        my_target[np.where(target == ii)] = temp
    
    loss = np.mean( - (np.log( prediction ) * my_target ))
    #- np.mean(my_target * np.log(prediction) + (1-my_target) * np.log(1-prediction))
    
    return loss
    
def softmax(x):
    
    # subtract the max to increase numerical stability
    x = x - np.max(x)
    sm =  np.exp(x) / np.sum(np.exp(x + 1e-9), axis=-1, keepdims=True)
    
    return sm



class MLPClient(fl.client.NumPyClient):
    
    def __init__(self, dim_in=64, dim_h=64, num_classes=3, lr=3e-3, l2=1e-12, split="alice"):
        
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.num_classes = num_classes
        self.split = split
        self.l2 = l2
        
        self.w_xh = np.random.randn(self.dim_in, self.dim_h) / np.sqrt(self.dim_in * self.dim_h)
        self.w_hh = np.random.randn(self.dim_h, self.dim_h) / np.sqrt(self.dim_h * self.dim_h)
        self.w_hy = np.random.randn(self.dim_h, self.num_classes) / np.sqrt(self.dim_h * self.num_classes)

        self.lr = lr
    
    def set_parameters(self, parameters):
        
        parameters = np.array(parameters)
        total_params = reduce(lambda a,b: a*b, np.array(parameters).shape)
        expected_params = self.dim_in * self.dim_h + self.dim_h**2 + self.dim_h * self.num_classes
        
        assert total_params == expected_params, f"expected {expected_params} params, got {total_params} params"
        
        self.w_xh = parameters[0:self.dim_in * self.dim_h].reshape(self.dim_in, self.dim_h)
        self.w_hh = parameters[self.dim_in * self.dim_h:self.dim_in * self.dim_h + self.dim_h**2].reshape(self.dim_h, self.dim_h)
        self.w_hy = parameters[self.dim_in*self.dim_h + self.dim_h**2:].reshape(self.dim_h, self.num_classes)

        # relu activation
        #self.act = np.tanh 
        self.act = lambda x: x * (x > 0.0)
    
    def forward(self, x, w_xh=None, w_hh=None, w_hy=None):
        
        if w_xh is not None:
            x = self.act(np.matmul(x, w_xh))
            x = self.act(np.matmul(x, w_hh))
            x = softmax(np.matmul(x, w_hy))
        else:
            x = self.act(np.matmul(x, self.w_xh))
            x = self.act(np.matmul(x, self.w_hh))
            x = softmax(np.matmul(x, self.w_hy))
        
        return x
    
    def get_loss(self, x, y, w_xh, w_hh, w_hy):

        prediction = self.forward(x, w_xh, w_hh, w_hy)
        
        loss = cross_entropy(y, prediction)
        
        return loss
    
    def fit(self, parameters, config=None, epochs=3):
        """
        Set the model parameters with parameters,         
        """
        self.set_parameters(parameters)
        
        get_grad = autograd.grad(self.get_loss, argnum=(2,3,4))
        
        x, y = get_data(split=self.split)

    
        
        for ii in range(epochs):
            grad = get_grad(x, y, self.w_xh, self.w_hh, self.w_hy)

            for param, my_grad in zip([self.w_xh, self.w_hh, self.w_hy], grad):

                param -= self.lr * (my_grad ) #+ self.l2 * param)


        loss, _, accuracy_dict = self.evaluate(self.get_parameters())

        return self.get_parameters(), len(y), {"loss": loss, "accuracy": accuracy_dict["accuracy"]}

    
    def get_parameters(self):
        
        my_parameters = np.append(self.w_xh.reshape(-1), self.w_hh.reshape(-1)) 
        my_parameters = np.append(my_parameters, self.w_hy.reshape(-1)) 
        
        return my_parameters
    
    def evaluate(self, parameters, config=None):
        
        self.set_parameters(parameters)
        
        val_x, val_y = get_data(split="val")
        
        prediction = self.forward(val_x)
        loss = cross_entropy(val_y, prediction)
        
        prediction_class = np.argmax(prediction, axis=-1)
        
        accuracy = sklearn.metrics.accuracy_score(val_y, prediction_class)
        
        return float(loss), len(val_y), {"accuracy":float(accuracy)}
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--split", type=str, default="alice",\
            help="The training split to use, options are 'alice', 'bob', or 'all'")

    args = parser.parse_args()
    
    fl.client.start_numpy_client("localhost:8080", client=MLPClient()) #split = aggs.split))
