import numpy as np

for d in [2,10,50,100,500]:
    X=np.random.rand(100,d)
    dist=[]
    for i in range(99):
        dist.append(np.linalg.norm(X[i]-X[i+1]))
    print("Dimension:",d)
    print("Average Distance:",np.mean(dist))
