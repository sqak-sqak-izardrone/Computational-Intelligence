# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.

import numpy as np


class Perceptron:
    # this class defines a single perceptron, 
    # this include two method for the simple perceptron for the logic gate with step function as activation
    # and general perceptron equipped with a certain activation function
    
    
    # the perceptron is initialized with 
    # 1. its number of input features
    # 2. its initial weight , an array of length features+1 with the last term as bias
    # 3. its activation function, a string
    def __init__(self, features, init_weight, activation_function="sigmoid"):
        self.features=features
        self.weights=init_weight[0:features]
        self.bias=init_weight[features]
        self.activation_function=activation_function
        self.acti_value=0
        self.z_value=0
        self.gradient=np.ndarray(features+1)
        self.batch_gradient=np.ndarray(features+1)
        
        
        
    # a step function
    def step(s):
        if(s>0):
            return 1
        else:
            return 0
     
    
    # the general activation function, it switches to different concreate activation function 
    # depends on the activation function type of the perceptron 
    # currently, sigmoid and tanh is provided
    def activate(self,s):
        if(self.activation_function=="sigmoid"):
            self.value= self.sigmoid(s)
            return self.sigmoid(s)
        elif(self.activation_function=="tanh"):
            self.value = self.tanh(s)
            return self.tanh(s)
        else:
            raise ValueError("No such activation function!")
            
    
    # the general activation function's derivative, it switches to different concreate activation function derivatives
    # depends on the activation function type of the perceptron 
    # currently, sigmoid and tanh is provided
    def activation_deriv(self,s):
        if(self.activation_function=="sigmoid"):
            return self.sigmoid_deriv(s)
        elif(self.activation_function=="tanh"):
            return self.tanh_deriv(s)
        else:
            raise ValueError("No such activation function!")
     
    # sigmoid function and its derivative    
    def sigmoid(s):
        return 1/(1 + np.exp(-s))
    def sigmoid_deriv(self, s):
        return  self.sigmoid(s)*(1-self.sigmoid(s))
    
    #tanh function and its derivative
    def tanh(s):
        return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
    def tanh_deriv(s):
        return (np.exp(s)-np.exp(-s))/2*(np.exp(s)-np.exp(-s))/2
    
    # return the activation value for the given input
    def feed_forward(self, input):
        self.z_value=np.dot(input,self.weights)+self.bias
        return self.activate(self.z_value)        
     
    # the feed forward for step function activation
    # only used for logic gate case
    def simple_feed_forward(self,input):
        sum=np.dot(input,self.weights)+self.bias
        return self.step(sum)
    
    # update weight
    # the backward calculation for one perceptron logical gates
    def simple_back_propagation(self,input,labels,learning_rate):  
        result = []      
        for x in range(len(input)):
            result[x] = self.feed_forward(input[x])
        loss=np.sum(np.subtract(labels - result))
        self.weights=self.weights+learning_rate*input*loss
        self.bias=self.bias+learning_rate*loss  
        return
        
class Layer:
    # this class defines one layer (either hidden or output layer) of the ANN
    # storing the list of perceptrons inside this layer
    
    # construct the layer
    # specify the activation function , number of perceptrons, input feature number and initial weight for each perceptron
    # we give all perceptron the same initial weights
    def __init__(self, activation_function, perceptron_number, input_length, init_weight):
        self.input_length=input_length
        self.perceptron_number=perceptron_number
        self.activation_function=activation_function
        self.activation_values=np.ndarray(perceptron_number)
        self.perceptrons = []
        for x in range(perceptron_number):
            self.perceptrons.append(Perceptron(input_length,init_weight,activation_function))
            
    # feed forward for this layer
    # gather the feed forward results from each perceptron of this layer and 
    # return the results combined in an array of length len(self.perceptrons)
    def feed_forward(self,input):
        result = np.ndarray(len(self.perceptrons))
        for x in range(len(self.perceptrons)):
            self.activation_values[x]=self.perceptrons[x].activate(input)
            result[x]=self.activation_values[x]
            
    # this takes in 
    # 1. the partial derivative of the loss function L with respect to the activation value of the forward layer
    # 2. the forward layer, essentially z_value and weights of its perceptrons
    # 3. the backward layer, essentially activation value of its perceptrons
    # it updates all the gradients of the perceptron weights for this layer
    # returns the partial derivative of the loss function L with respect to the activation value of the current layer
    def back_propagation(self,forward_partial,forward_layer,backward_layer_activation):
        partial=np.ndarray(len(self.perceptrons))
        for x in range(len(self.perceptrons)):
            sum=0
            for y in range(len(forward_partial)):
                subsum1=forward_partial[y]
                subsum2=forward_layer.perceptrons[y].activation_deriv(forward_layer.perceptrons[y].z_value)
                subsum3=forward_layer.perceptrons[y].weights[x]
                sum+=subsum1*subsum2*subsum3
            partial[x]=sum
            subsum1=partial[x]
            subsum2=self.perceptrons[x].activation_deriv(self.perceptrons[x].z_value)
            for y in range(self.input_length):
                subsum3=backward_layer_activation[y]
                self.perceptrons[x].gradient[y]=subsum1*subsum2*subsum3
                self.perceptrons[x].batch_gradient[y]+=subsum1*subsum2*subsum3
            # the bias
            self.perceptrons[x].gradient[self.input_length]=subsum1*subsum2
            self.perceptrons[x].batch_gradient[self.input_length]+=subsum1*subsum2
        return partial
     
    
        

class ANN:
    # ANN artificial neural network model, initialize the model with a list of layers
    def __init__(self, layers):
        self.layer_number=len(layers)
        self.layers=[]
        for x in range(len(layers)):
            self.layers.append(layers[x])
    
    # predict the result from the current model, the prediction arise from iterating feed forward from each layer
    def predict(self, input):
        result = input
        for x in range(len(self.layers)):
            result=self.layers[x].feed_forward(result) # the result of the previous layer(or initial input) serves as the input for the curren layer
        return result
    
    # back propagation for one sample
    # update the gradients in each perceptron
    def back_propagation(self, loss_function_deriv, input, target):
        partial=np.ndarray(self.layer_number)
        result = self.predict(input)
        for x in range(len(target)):
            partial[x]=loss_function_deriv(result,target)
            subsum1=partial[x]
            subsum2=self.layers[len(self.layers)-1].perceptrons[x].activation_deriv(self.layers[len(self.layers)-1].perceptrons[x].z_value)
            for y in range(self.input_length):
                subsum3=self.layers[len(self.layers)-2].perceptrons[y].acti_value
                self.layers[len(self.layers)-1].perceptrons[x].gradient[y]=subsum1*subsum2*subsum3
                self.layers[len(self.layers)-1].perceptrons[x].batch_gradient[y]+=subsum1*subsum2*subsum3
            # the bias
            self.layers[len(self.layers)-1].perceptrons[x].gradient[self.input_length]=subsum1*subsum2
            self.layers[len(self.layers)-1].perceptrons[x].batch_gradient[self.input_length]+=subsum1*subsum2
        for x in reversed(range(1,len(self.layers)-1)):
            partial=self.layers[x].back_propagation(partial,self.layers[x+1],self.layers[x-1].activation_values)
        # for the first layer the "backward layer activation value" is just the input
        self.layers[0].back_propagation(partial,self.layers[1],input)
        
    
    # back propagation for a batch of records
    # the gradient for a batch of records is the average of the gradients for each record
    # update the gradients in each perceptron
    def back_propagation_batch(self,inputs,targets,loss_function_deriv):
        #initialize all batch gradient to 0
        for x in range(self.layer_number):
            for y in range(self.layers[x].peceptron_number):
                for z in range(self.layers[x].perceptrons[y].features+1):
                    self.layers[x].perceptrons[y].batch_gradient[z]=0
        input_length=len(inputs)
        # sum all gradients for different records together in batch_gradient
        for i in range(input_length):
            self.back_propagation(loss_function_deriv,inputs[i],targets[i])
        #average all gradients for the same parameters and update to the gradient
        for x in range(self.layer_number):
            for y in range(self.layers[x].peceptron_number):
                for z in range(self.layers[x].perceptrons[y].features+1):
                    self.layers[x].perceptrons[y].batch_gradient[z]=self.layers[x].perceptrons[y].batch_gradient[z]/input_length
                    self.layers[x].perceptrons[y].gradient[z]=self.layers[x].perceptrons[y].batch_gradient[z]
    
    def gradient_decent(self,learning_rate):
        for x in range(self.layer_number):
            for y in range(self.layers[x].perceptron_number):
                for z in range(self.layers[x].input_length+1):
                    self.layers[x].perceptrons[y].weights[z]-=learning_rate*self.layers[x].perceptrons[y].gradient[z]
                    
    def train(inputs, targets, learning_rate, threshold, loss_function, loss_function_deriv, batch_size=1):
        #do:
        #batch the inputs into different batches
        #back propagates on these batches to get the gradients
        #perform gradient decent
        #while(loss_function>threshold)
        pass
                    
        
        
    
