from loss import MSE
from models import NonLinear
import numpy as np
from utils import plot_3d


def main():
    input_dim = 2
    hidden_dim = 10
    x = np.random.uniform(-1, 1, (200, input_dim))
    weights_true = np.array([[5, 1],]).T
    bias_true = np.array([1])
    y = (x ** 2) @ weights_true + x @ weights_true + bias_true
    loss = MSE()
    model = NonLinear(input_dim, hidden_dim, loss)
    y_pred = model.fit(x, y, 0.1, 40)
    plot_3d(x, y, y_pred)
    



if __name__ == '__main__':
    main()