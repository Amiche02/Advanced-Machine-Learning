import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def print_prams(model):
    for i, layer in enumerate(model.layers):
        print(f"Layer {i + 1} weights:\n{layer.weights}")
        print(f"Layer {i + 1} bias:\n{layer.bias}\n")

class Layers(object):
    def __init__(self, nb_neuron, x_shape):
        self.weights = np.random.rand(nb_neuron, x_shape)
        self.bias = np.random.rand(nb_neuron, 1)

    def forward(self, X):
        self.input = X
        self.output = relu(np.dot(self.weights, self.input) + self.bias)
        return self.output

    def backward(self, d_output):
        d_input = np.dot(self.weights.T, d_output * relu_derivative(self.output))
        self.d_weights = np.dot(d_output * relu_derivative(self.output), self.input.T)
        self.d_bias = np.sum(d_output * relu_derivative(self.output), axis=1, keepdims=True)
        return d_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias


class Network(object):
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred):
        d_output = y_pred - y_true
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(y, y_pred)
            self.update(learning_rate)


# Exemple d'utilisation :
np.random.seed(42)
X = np.array([[0,0,1,1], [0,1,0,1]])
y = np.array([[0,1,1,0]])

# Créer et ajouter des couches au réseau
network = Network()
network.add_layer(Layers(10, 2))
network.add_layer(Layers(1, 10))

print("\nInitial parameters :")
print_prams(network)

# Entraînement du réseau
network.train(X, y, epochs=1000, learning_rate=0.01)

# Vérifier les résultats
y_pred = network.forward(X)
print(f"\nTargets : {y}")
print(f"Predictions: {y_pred}")

int_pred = []
for elt in y_pred[0]:
    if elt < 0.5:
        elt = 0
    else :
        elt = 1
    int_pred.append(elt)

print(f"Integer Predictions: {int_pred}")

# Afficher les poids et les biais de chaque couche du réseau
print("\nUpdated parameters :")
print_prams(network)