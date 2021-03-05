import numpy as np
from relu import Relu

class Linear:
    def __init__(self, input_dim, hidden_dim=1, loss=None):
        self.weights = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(hidden_dim)
        self.loss = loss

    def __call__(self, x):
        self.x = x
        output = x @ self.weights + self.bias 
        return output

    def backward(self, gradient):
        self.weights_gradient = self.x.T @ gradient
        self.bias_gradient = gradient.sum(axis=0)
        self.x_gradient = gradient @ self.weights.T
        return self.x_gradient

    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.weights_gradient
        self.bias = self.bias - learning_rate * self.bias_gradient

    def fit(self, x: np.ndarray, y: np.ndarray, lr: float, num_epochs: int):
        for epoch in range(num_epochs):
            y_pred = self(x)
            loss_value = self.loss(y_pred, y)
            print(f'Epoch {epoch} - loss {loss_value}')
            loss_gradient = self.loss.backward()
            self.backward(loss_gradient)
            self.update(lr)
    

class NonLinear:
    def __init__(self, input_dim, hidden_dim, loss=None):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.relu = Relu()
        self.output = Linear(hidden_dim, 1)
        self.loss = loss

    def __call__(self, x):
        l1 = self.linear1(x)
        r = self.relu(l1)
        out = self.output(r)
        return out

    def backward(self, output_gradient):
        out_gradient = self.output.backward(output_gradient)
        relu_gradient = self.relu.backward(out_gradient)
        l1_gradient = self.linear1.backward(relu_gradient)
        return l1_gradient

    def update(self, learning_rate: float):
        self.linear1.update(learning_rate)
        self.output.update(learning_rate)


    def fit(self, x: np.ndarray, y: np.ndarray, lr: float, num_epochs: int):
        for epoch in range(num_epochs):
            y_pred = self(x)
            loss_value = self.loss(y_pred, y)
            print(f'Epoch {epoch} - loss {loss_value}')
            loss_gradient = self.loss.backward()
            self.backward(loss_gradient)
            self.update(lr)
        return y_pred