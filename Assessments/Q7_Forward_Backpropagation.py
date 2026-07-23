import numpy as np

x=np.array([1,2])
w=np.array([[0.5,0.3],[0.4,0.7]])
hidden=np.dot(x,w)
output=1/(1+np.exp(-hidden))
print("Forward Output:",output)

x=1.0
target=1.0
weight=0.5
lr=0.1
net=x*weight
out=1/(1+np.exp(-net))
error=target-out
gradient=error*out*(1-out)*x
weight=weight+lr*gradient
print("Updated Weight:",weight)
