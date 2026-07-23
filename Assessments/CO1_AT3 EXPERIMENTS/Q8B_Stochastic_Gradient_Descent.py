import numpy as np
from sklearn.linear_model import SGDRegressor

x=5
lr=0.1
for i in range(20):
    grad=2*x
    x=x-lr*grad
    print(i,x)

X=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,6,8,10])
model=SGDRegressor(max_iter=5000,learning_rate="constant",eta0=0.01,random_state=42)
model.fit(X,y)
print("Prediction:",model.predict([[6]])[0])
