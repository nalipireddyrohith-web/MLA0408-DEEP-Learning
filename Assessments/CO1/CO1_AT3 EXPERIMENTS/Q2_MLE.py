import numpy as np
np.random.seed(0)
true_mean=5
true_std=2
data=np.random.normal(true_mean,true_std,1000)
mle_mean=np.mean(data)
mle_variance=np.mean((data-mle_mean)**2)
print("Actual Mean:",true_mean)
print("Estimated Mean:",mle_mean)
print("Actual Variance:",true_std**2)
print("Estimated Variance:",mle_variance)
