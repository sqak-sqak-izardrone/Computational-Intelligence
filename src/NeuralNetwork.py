# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.



class Perceptron:
    def __init__(self, features, init_weight, activation_function="sigmoid"):
        self.features=features
        self.weights=init_weight[0:len(features)]
        self.bias=init_weight[len(features)]
        self.activation_function=activation_function
        
    def step(s):
        if(s>0):
            return 1;
        else:
            return 0
        
    def activate(s,activation_function):
        rt = np.ndarray(len(s))
        for x in range(len(s)):
            if(activation_function=="sigmoid"):
                rt[x]=sigmoid(s)
            else if(activation_function=="tanh"):
                rt[x]=tanh(s)
            else:
                raise ValueError("No such activation function!")
        return rt
            
            
    def activation_deriv(s,activation_function):
        rt = np.ndarray(len(s))
        for x in range(len(s)):
            if(activation_function=="sigmoid"):
                rt[x]=sigmoid_deriv(s)
            else if(activation_function=="tanh"):
                rt[x]=tanh_deriv(s)
            else:
                raise ValueError("No such activation function!")
        return rt
            
    def sigmoid(s):
        return 1/(1+np.exp(-s))
    def sigmoid_deriv(s):
        return  sigmoid(s)*(1-sigmoid(s))
    
    def tanh(s):
        return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
    def tanh_deriv(s):
        return (np.exp(s)-np.exp(-s))/2*(np.exp(s)-np.exp(-s))/2
    
    def feed_forward(self, input):
        sum  = np.ndarray(len(input))
        for x in range(len(input)):
            sum[x]=np.dot(input[x],self.weights)+self.bias
        return activate(sum,self.activation_function)        
        
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
        
        

class ANN:
    def __init__(self):
        pass
    
