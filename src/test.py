from NeuralNetwork import Layer 

layer1 = Layer(3,2,"sigmoid")

layer2 = Layer(1,3,"softmax")

result1 = layer1.feed_forward([0,1])

print(result1)

result2 = layer2.feed_forward(result1)

print(result2)