import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        self.activation_name = activation
        self.activation = np.tanh if activation == 'tanh' else lambda x: 1 / (1 + np.exp(-x))
        self.activation_derivative = (
            lambda x: 1 - np.tanh(x)**2 if activation == 'tanh' else x * (1 - x)
        )

        self.weights = []
        for i in range(1, len(layers)):
            w = 2 * np.random.random((layers[i-1] + 1, layers[i])) - 1
            self.weights.append(w)

    # -----------------------
    # MÉTODO FIT
    # -----------------------
    def fit(self, X, y, learning_rate=0.03, epochs=40000):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Agregar bias a las entradas

        for epoch in range(epochs):
            idx = np.random.randint(X.shape[0])
            a = [X[idx]]

            # Forward pass
            for w in self.weights:
                a[-1] = np.atleast_2d(a[-1])
                z = np.dot(a[-1], w)
                a.append(self.activation(z))
                if w is not self.weights[-1]:
                    a[-1] = np.hstack([a[-1], np.ones((a[-1].shape[0], 1))])

            # Backward pass
            error = y[idx] - a[-1]
            deltas = [error * self.activation_derivative(a[-1])]

            for i in range(len(a) - 2, 0, -1):
                delta = deltas[-1].dot(self.weights[i].T) * self.activation_derivative(a[i])
                if i > 1:
                    delta = delta[:, :-1]
                deltas.append(delta)
            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])

                grad = layer.T.dot(delta)
                if grad.shape != self.weights[i].shape:
                    min_rows = min(grad.shape[0], self.weights[i].shape[0])
                    min_cols = min(grad.shape[1], self.weights[i].shape[1])
                    grad = grad[:min_rows, :min_cols]

                self.weights[i][:grad.shape[0], :grad.shape[1]] += learning_rate * grad

    # -----------------------
    # MÉTODO PREDICT
    # -----------------------
    def predict(self, x):
        a = np.concatenate((x, [1]))  # agregar bias
        for w in self.weights:
            a = self.activation(np.dot(a, w))
            if w is not self.weights[-1]:
                a = np.concatenate((a, [1]))
        return a 
