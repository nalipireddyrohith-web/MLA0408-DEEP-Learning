from sklearn.datasets import make_classification,load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X,y=make_classification(n_samples=100,n_features=2,n_redundant=0,n_clusters_per_class=1,random_state=0)
p=Perceptron()
p.fit(X,y)
print("Perceptron Accuracy:",p.score(X,y))

iris=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
mlp=MLPClassifier(hidden_layer_sizes=(20,),max_iter=3000,random_state=42)
mlp.fit(X_train,y_train)
print("MLP Accuracy:",accuracy_score(y_test,mlp.predict(X_test)))
