import numpy as np

class Perceptron(object):
    def __init__(self, learning_rate=0.01, x_shape=2):
        self.shape = x_shape
        self.learning_rate = learning_rate
        self.weights = np.random.rand(self.shape)
        self.bias = np.random.rand()
        self.errors = []

    def gradient_descent(self,X, y, epoch=1):
        for i in range(epoch):
            y_hat = self.predict(X)
            error = y - y_hat
            self.weights += self.learning_rate * np.dot(error, X.T)
            self.bias += self.learning_rate * np.sum(error)
            self.errors.append(error)
    def predict(self,X):
        return np.matmul(self.weights, X) + self.bias

X = np.array([[0,0,1,1], [0,1,0,1]])
y = np.array([0,1,1,1])

np.random.seed(42)
model = Perceptron(learning_rate=0.01, x_shape=X.shape[0])
print(f"Initial parameters : Weights -> {model.weights} - Bias ->{model.bias}")
model.gradient_descent(X, y, epoch=14)
print(f"\nFinal parameters : Weights -> {model.weights} - Bias ->{model.bias}")

predictions = model.predict(X)
int_pred = []

for elt in predictions:
    if elt < 0.5:
        elt = 0
    else :
        elt = 1
    int_pred.append(elt)

print(f"\nBased Predictions : {predictions}\nPredictions : {int_pred}")