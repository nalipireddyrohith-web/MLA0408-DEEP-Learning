import numpy as np
x=np.array([1,2,3])
w=np.array([0.2,0.4,0.6])
b=0.5
z=np.dot(x,w)+b
sigmoid=1/(1+np.exp(-z))
relu=max(0,z)
print("Neuron Output:",z)
print("Sigmoid:",sigmoid)
print("ReLU:",relu)
