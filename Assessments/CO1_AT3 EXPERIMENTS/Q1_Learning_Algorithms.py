import numpy as np
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5],dtype=float)
y=np.array([2,4,6,8,10],dtype=float)

m,b=0,0
lr=0.01
epochs=500
loss=[]
n=len(X)

for _ in range(epochs):
    y_pred=m*X+b
    error=y_pred-y
    dm=(2/n)*np.sum(error*X)
    db=(2/n)*np.sum(error)
    m-=lr*dm
    b-=lr*db
    loss.append(np.mean(error**2))

print("Slope:",m)
print("Intercept:",b)

plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.show()
