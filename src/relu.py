import numpy as np

class Relu:
    def __call__(self, input):
        self.input = input
        self.output = np.clip(self.input, 0, None)
        return self.output

    def backward(self, output_gradient):
        self.input_gradient = (self.input > 0) * output_gradient
        return self.input_gradient