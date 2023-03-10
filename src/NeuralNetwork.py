# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.

import numpy as np


class Perceptron:
    # this class defines a single perceptron, 
    # this include two method for the simple perceptron for the logic gate with step function as activation
    # and general perceptron equipped with a certain activation function
    
    
    # the perceptron is initialized with 
    # 1. its number of input features
    # 2. its activation function, a string
    # 3. number of perceptrons in the layer
    # the weights and bias are initialized within a gaussian distribution of (0,2/(n_perceptrons - 1)) and (0,2) respectively
    def __init__(self, features, perceptrons, n_perceptrons, activation_function = "sigmoid"):
        self.features=features
        self.weights = []
        for i in range(features):
            if(n_perceptrons - 1 > 0):
                self.weights.append(np.random.normal(0, 2/(n_perceptrons - 1)))
            else:
                self.weights.append(np.random.normal(0, 2))
        self.bias = np.random.normal(0, 2/(n_perceptrons - 1)) if (n_perceptrons - 1 > 0) else np.random.normal(0, 2)
        self.activation_function=activation_function
        self.z_value=0
        self.gradient=np.ndarray(features+1)
        self.batch_gradient=np.ndarray(features+1)
        self.perceptrons = perceptrons
        
        
    
     
    
    # the general activation function, it switches to different concreate activation function 
    # depends on the activation function type of the perceptron
    # this takes in:
    # 1. input z value for the current perceptron
    # 2. all z value for the current layer(for softmax purpose) 
    # currently, sigmoid, tanh, ReLu and stable softmax are provided
    def activate(self,s, input):
        if(self.activation_function =="sigmoid"):
            return self.sigmoid(s)
        elif(self.activation_function =="tanh"):
            return self.tanh(s)
        elif(self.activation_function == "ReLu"):
            return self.ReLu(s)
        elif(self.activation_function == "softmax"):
            return self.stable_softmax(s, input)
        else:
            raise ValueError("No such activation function!")
            
    
    # the general activation function's derivative, it switches to different concreate activation function derivatives
    # depends on the activation function type of the perceptron 
    # currently, sigmoid, tanh, ReLu and stable softmax are provided
    def activation_deriv(self,s):
        if(self.activation_function=="sigmoid"):
            return self.sigmoid_deriv(s)
        elif(self.activation_function=="tanh"):
            return self.tanh_deriv(s)
        elif(self.activation_function == "ReLu"): 
            return self.ReLu_deriv(s)
        elif(self.activation_function == "softmax"):
            return self.stable_softmax_deriv()
        else:
            raise ValueError("No such activation function!")
     
    # sigmoid function and its derivative    
    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    def sigmoid_deriv(self, s):
        return  self.sigmoid(s)*(1-self.sigmoid(s))
    
    #tanh function and its derivative
    def tanh(self,s):
        return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
    def tanh_deriv(self,s):
        return 1-self.tanh(s)*self.tanh(s)
    
    #ReLu activation function and its derivative 
    def ReLu(self, s): 
        return max(0,s)
    def ReLu_deriv(self, s):
        return 1 if s > 0 else 0
    
    ##Softmax activation and its derivative 
    def stable_softmax(self,s, input):
        z_value_arr = []
        for p in self.perceptrons: 
            z_value_arr.append(np.dot(input, p.weights) + p.bias)
        max_value = np.max(z_value_arr)
        z_value_arr = [x - max_value for x in z_value_arr]
        self.softmax = np.exp(s - max_value)/ np.sum(np.exp(z_value_arr))
        return self.softmax
    
    def stable_softmax_deriv(self):   
        return self.softmax*(1 - self.softmax)
    
    # return the activation value for the given input, calculating z value and activation value
    def feed_forward(self, input):
        self.z_value=np.dot(input,self.weights)+self.bias
        return self.activate(self.z_value, input)        
    
    """ Legacy code
    # a step function
    def step(self,s):
        if(s>0):
            return 1
        else:
            return 0
    # the feed forward for step function activation
    # only used for logic gate case
    def simple_feed_forward(self,input):
        sum=np.dot(input,self.weights)+self.bias ## calculating the linear equation z = w^T*x + b 
        #print(sum)
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
    """
        
class Layer:
    # this class defines one layer (either hidden or output layer) of the ANN
    # storing the list of perceptrons inside this layer
    # construct the layer
    # this takes in:
    # 1. number of perceptrons
    # 2. input feature number 
    # 3. the activation function for each perceptron
    # we will randomize the initial weight for each perceptrons
    def __init__(self, perceptron_number, input_length, activation_function="sigmoid"):
        self.input_length=input_length
        self.perceptron_number=perceptron_number
        self.activation_function=activation_function
        self.activation_values=np.ndarray(perceptron_number)
        self.perceptrons = []
        for x in range(perceptron_number):
            self.perceptrons.append(Perceptron(input_length, self.perceptrons,perceptron_number,activation_function))

    # feed forward for this layer
    # gather the feed forward results from each perceptron of this layer and 
    # return the results combined as an numpy array of length len(self.perceptrons)
    def feed_forward(self,input):
        result = np.ndarray(len(self.perceptrons))
        ## store the activation values of previous layer to calculate the softmax functions 
        self.input = input 
        ##
        for x in range(len(self.perceptrons)):
            self.activation_values[x]=self.perceptrons[x].feed_forward(input)
            result[x]=self.activation_values[x]
        return result
            
    # this takes in: 
    # 1. the partial derivative of the loss function L with respect to the activation value of the forward layer
    # 2. the forward layer, essentially z_value and weights of its perceptrons
    # 3. the backward layer, essentially activation value of its perceptrons
    # it updates all the gradients of the perceptron weights for this layer
    # it add gradient to the batch gradient field for averaging in batch backpropagation
    # returns the partial derivative of the loss function L with respect to the activation value of the current layer

    # matrix multiplication is avoided here to reduce the space complexity of the algorithm
    def back_propagation(self,forward_partial,forward_layer,backward_layer_activation):
        # calculation for the derivative on the activation value
        partial=np.ndarray(len(self.perceptrons))
        for x in range(len(self.perceptrons)):
            sum=0
            for y in range(len(forward_partial)):
                subsum1=forward_partial[y]
                subsum2=forward_layer.perceptrons[y].activation_deriv(forward_layer.perceptrons[y].z_value)
                subsum3=forward_layer.perceptrons[y].weights[x]
                sum+=subsum1*subsum2*subsum3
            partial[x]=sum
        # calculation for the derivative on the weights （the gradient for one input）
            subsum1=partial[x]
            subsum2=self.perceptrons[x].activation_deriv(self.perceptrons[x].z_value)
            for y in range(self.input_length):
                subsum3=backward_layer_activation[y]
                self.perceptrons[x].gradient[y]=subsum1*subsum2*subsum3
                self.perceptrons[x].batch_gradient[y]+=subsum1*subsum2*subsum3
            # the bias
            self.perceptrons[x].gradient[self.input_length]=subsum1*subsum2
            self.perceptrons[x].batch_gradient[self.input_length]+=subsum1*subsum2
        #print("propagate through layer")
        #return the derivative on the activation value for the backward layer
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

    # predict the result for a batch of inputs
    def predictBatch(self, inputs):
        size = len(inputs)
        results = np.ndarray((size,self.layers[self.layer_number-1].perceptron_number))

        for i in range(size):
            results[i] = self.predict(inputs[i])
        return results
    
    # back propagation for one sample, essentially back propagation for stochastic gradient decent
    # this takes in:
    # 1. the derivative function for the loss function with respect to the activation value of the last layer
    # 2. the input features
    # 3. the ideal output label
    # update the gradients in each perceptron
    def back_propagation(self, loss_function_deriv, input, target):
        # back propagation for the last layer, essentially the same as back_propagation for previous layers, only difference is the partial derivative of loss with respect 
        # to the activation value is given directly from the loss_function_deriv
        partial=np.ndarray(len(target))
        result = self.predict(input) 
        for x in range(len(target)):
            partial[x]=loss_function_deriv(x,result,target)
            subsum1=partial[x]
            subsum2=self.layers[len(self.layers)-1].perceptrons[x].activation_deriv(self.layers[len(self.layers)-1].perceptrons[x].z_value)
            for y in range(self.layers[len(self.layers)-1].input_length):
                subsum3=self.layers[len(self.layers)-2].activation_values[y]
                self.layers[len(self.layers)-1].perceptrons[x].gradient[y]=subsum1*subsum2*subsum3
                self.layers[len(self.layers)-1].perceptrons[x].batch_gradient[y]+=subsum1*subsum2*subsum3
            # the bias
            self.layers[len(self.layers)-1].perceptrons[x].gradient[self.layers[len(self.layers)-1].input_length]=subsum1*subsum2
            self.layers[len(self.layers)-1].perceptrons[x].batch_gradient[self.layers[len(self.layers)-1].input_length]+=subsum1*subsum2
            #print(self.layers[len(self.layers)-1].perceptrons[x].gradient) 


        # back propagation for the hidden layers
        for x in reversed(range(1,len(self.layers)-1)):
            partial=self.layers[x].back_propagation(partial,self.layers[x+1],self.layers[x-1].activation_values)
        # back propagation for the first layer
        # for the first layer the "backward layer activation value" is just the input
        self.layers[0].back_propagation(partial,self.layers[1],input)
        #print(self.layers[0].perceptrons[x].gradient)
        
    
    # back propagation for a batch of records
    # the gradient for a batch of records is the average of the gradients for each record
    # this takes in:
    # 1. the loss function derivative with respect to the activation values of the last layer
    # 2. input batch
    # 3. idea output label batch
    # update the gradients in each perceptron
    def back_propagation_batch(self,loss_function_deriv,inputs,targets):
        #initialize all batch gradient to 0
        for x in range(self.layer_number):
            for y in range(self.layers[x].perceptron_number):
                for z in range(self.layers[x].perceptrons[y].features+1):
                    self.layers[x].perceptrons[y].batch_gradient[z]=0
        input_length=len(inputs)
        # sum all gradients for different records together in batch_gradient
        for i in range(input_length):
            self.back_propagation(loss_function_deriv,inputs[i],targets[i])
        #average all gradients for the same parameters and update to the gradient
        for x in range(self.layer_number):
            for y in range(self.layers[x].perceptron_number):
                for z in range(self.layers[x].perceptrons[y].features+1):
                    self.layers[x].perceptrons[y].batch_gradient[z]=self.layers[x].perceptrons[y].batch_gradient[z]/input_length
                    self.layers[x].perceptrons[y].gradient[z]=self.layers[x].perceptrons[y].batch_gradient[z]
        #for x in range(self.layer_number):
        #    for y in range(self.layers[x].perceptron_number):
        #        print(self.layers[x].perceptrons[y].gradient)
    
    # a gradient decent
    # this takes in:
    # 1. learning rate
    # 2. threshold for the gradient decent
    # returns true if every gradient in the model is lower or equal to the threshold
    def gradient_decent(self,learning_rate,threshold):
        converged=True
        for x in range(self.layer_number):
            for y in range(self.layers[x].perceptron_number):
                for z in range(self.layers[x].input_length):
                    self.layers[x].perceptrons[y].weights[z]-=learning_rate*self.layers[x].perceptrons[y].gradient[z]
                    if np.abs(self.layers[x].perceptrons[y].gradient[z])>threshold:
                        #print(self.layers[x].perceptrons[y].gradient[z])
                        converged=False
                #for the bias
                self.layers[x].perceptrons[y].bias-=learning_rate*self.layers[x].perceptrons[y].gradient[self.layers[x].input_length]
                if np.abs(self.layers[x].perceptrons[y].gradient[self.layers[x].input_length])>threshold:   
                    converged=False
        return converged

    # this is the training method for the overall model
    # this takes in:
    # 1. the inputs of the training data
    # 2. the target output of the training data
    # 3. the validation set(not necessarily the whole validation set)4
    # 4. the target for the validation set
    # 5. learning rate
    # 6. threshold for convergence
    # 7. the loss function
    # 8. the derivative for the loss function
    # 9. the batch size for mini_batching, SGD in batch_size=1 by default, no batching if batch_size=len(inputs)
    # updates the weights and bias for the model and trains it to fit the target
    # returns a (2,batch_iteration) shape array containing the loss value for the batch and the validation set for each iteration          
    def train(self, inputs, targets, val_inputs, val_targets, learning_rate, threshold, loss_function, loss_function_deriv, batch_size=1):
        #do:
        #randomly batch the inputs into different batches
        #back propagates on these batches to get the gradients
        #perform gradient decent
        #stores the loss value
        #while(gradient>threshold)  
        
        loss=[]
        val_loss=[]
        t_loss=[loss,val_loss]
        while True:
            #randomized batching
            batch_number=len(inputs)//batch_size
            indices=np.random.permutation(len(inputs))
            feature_batches=[]
            target_batches=[]
            for x in range(batch_number):
                feature_batches.append(inputs[indices[x*batch_size:(x+1)*batch_size], :])
                target_batches.append(targets[indices[x*batch_size:(x+1)*batch_size], :])

            #backward propagation and gradient decent
            converged=True
            for x in range(batch_number):
                self.back_propagation_batch(loss_function_deriv,feature_batches[x],target_batches[x])
                converged=self.gradient_decent(learning_rate,threshold) and converged 

                #stores the loss value for this iteration
                loss_value=0
                for y in range(batch_size):
                    loss_value+=loss_function(self.predict(feature_batches[x][y]),target_batches[x][y])
                loss.append(loss_value/batch_size)  
                #stores the loss value for validation set
                loss_value=0
                for y in range(len(val_inputs)):
                    loss_value+=loss_function(self.predict(val_inputs[y]),val_targets[y])
                val_loss.append(loss_value/len(val_inputs))
                if converged:
                    break
            if converged:
                break
        return t_loss

            
        
        
                    
        
        
    
