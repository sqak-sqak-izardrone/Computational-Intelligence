# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.




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
        self.weights=init_weight[0:len(features)]
        self.bias=init_weight[len(features)]
        self.activation_function=activation_function
        
        
    # a step function
    def step(s):
        if(s>0):
            return 1;
        else:
            return 0
     
    
    # the general activation function, it switches to different concreate activation function 
    # depends on the activation function type of the perceptron 
    # currently, sigmoid and tanh is provided
    def activate(self,s):
        rt = np.ndarray(len(s))
        for x in range(len(s)):
            if(self.activation_function=="sigmoid"):
                rt[x]=sigmoid(s)
            else if(self.activation_function=="tanh"):
                rt[x]=tanh(s)
            else:
                raise ValueError("No such activation function!")
        return rt
            
    
    # the general activation function's derivative, it switches to different concreate activation function derivatives
    # depends on the activation function type of the perceptron 
    # currently, sigmoid and tanh is provided
    def activation_deriv(self.s):
        rt = np.ndarray(len(s))
        for x in range(len(s)):
            if(self.activation_function=="sigmoid"):
                rt[x]=sigmoid_deriv(s)
            else if(self.activation_function=="tanh"):
                rt[x]=tanh_deriv(s)
            else:
                raise ValueError("No such activation function!")
        return rt
     
    # sigmoid function and its derivative    
    def sigmoid(s):
        return 1/(1+np.exp(-s))
    def sigmoid_deriv(s):
        return  sigmoid(s)*(1-sigmoid(s))
    
    #tanh function and its derivative
    def tanh(s):
        return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
    def tanh_deriv(s):
        return (np.exp(s)-np.exp(-s))/2*(np.exp(s)-np.exp(-s))/2
    
    def feed_forward(self, input):
        sum  = np.ndarray(len(input))
        for x in range(len(input)):
            sum[x]=np.dot(input[x],self.weights)+self.bias
        return activate(sum)        
        
    def simple_feed_forward(self,input):
        sum=np.dot(input,self.weights)+self.bias
        return step(sum)
    
    #update weight
    def simple_back_propagation(self,input,labels,learning_rate):        
        for x in range(len(input)):
            result[x]=feed_forward(input[x])
        loss=np.sum(labels-result)
        self.weights=self.weights+learning_rate*input*loss
        self.bias=self.bias+learning_rate*loss  
        return
        
class Layer:
    def __init__(self, activation_function, node_number, input_length, init_weight):
        self.activation_function=activation_function
        self.perceptrons = []
        for x in range(node_number):
            self.perceptrons.append(Perceptron(input_length,init_weight,activation_function))
            
    def feed_forward(self,input):
        result = np.ndarray((len(input),len(self.perceptrons)))
        for x in range(len(self.perceptrons)):
            for y in range(len(input)):
                result[y][x]=self.perceptrons[x].activate(input[y],self.activation_function)
    
    def num_perceptrons: 
        return len(self.perceptrons) 
        

class ANN:
    def __init__(self, layers):
        self.layers=[]
        for x in range(len(layers)):
            self.layers.append(layers[x])
            
    def predict(self, input):
        result = input
        for x in range(len(input)):
            for y in range(len(self.layers)):
                result=self.layers[y].feed_forward(result)
        return result
    
    def back_propagation(self):
        pass
        
                    
        
        
    
