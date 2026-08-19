from sklearn.neural_network import MLPRegressor
import numpy as np

X=np.random.rand(100,2)
y=X[:,0]+X[:,1]
model=MLPRegressor(batch_size=16,max_iter=500,random_state=42)
model.fit(X,y)
print("Score:",model.score(X,y))
