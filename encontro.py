
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

df = pd.read_csv('Dados_Treinamento_Sinal2.csv', header = None)

X = df.iloc[0:14, [0, 1, 2, 3]].values
y = df.iloc[0:14, 4].values


print('vetor x', X)

print('vetor y', y)

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
fig = plt.figure(figsize=(7,5))
plt.scatter(X[:,2],X[:,3], c=y, cmap=cm_bright)
plt.xlabel('atributo 1')
plt.ylabel('atributo 2')
plt.title('visualizacao dos dados')
plt.scatter(None, None, color = 'b', label = 'Classe 1')
plt.scatter(None, None, color = 'r', label = 'Classe 5')
plt.legend()
plt.show()


#espaco p teste:

np.random.seed(16)
weight_ = np.random.uniform(-1,1, X.shape[1]+1)
error_ = []

print('vetor de pesos', weight_)
print('vetor de erro', error_)



class Rede(object):
    def __init__(self, eta = 0.001, epoch = 1000):
        self.eta = eta
        self.epoch = epoch

    def fit(self, X, y):
        np.random.seed(16)
        self.weight_ = np.random.uniform(-1,1, X.shape[1]+1)
        self.error_ = []

        cost = 0

        for _ in range(self.epoch):
            output = self.activation_function(X)
            error = y - output

            self.weight_[0] += self.eta*sum(error)
            self.weight_[1:] += self.eta*X.T.dot(error)

            cost = 1./2*sum((error**2))
            self.error_.append(cost)
            print(self.error_)

        return self

    def net_input(self,X):
        return np.dot(X, self.weight_[1:]+self.weight_[0])

    def activation_function(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation_function(X)>=0.0, 'classe 1', 'classe 5')



names = ['Taxa de Aprendizado = 0.001', 'Taxa de Aprendizado = 0.1']
classifiers = [Rede(), Rede(eta=0.01)]
step = 1
plt.figure(figsize=(14, 5))
for name, classifier in zip(names, classifiers):
    ax = plt.subplot(1, 2, step)
    clf = classifier.fit(X, y)
    ax.plot(range(len(clf.error_)), clf.error_)
    ax.set_ylabel('Erro')
    ax.set_xlabel('itaracoes')
    ax.set_title(name)

    step += 1

plt.show()

clf = Rede()

clf.fit(X,y)

A = [1, 1.7, 1.96, .97] #classe1

B = [5, 5.7, 5.96, 4.97] #classe5

print('A= ',clf.predict(A))

print('B= ',clf.predict(B))











