import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(x, y, y_pred=None):  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x[:, 0], x[:, 1], y, label='underlying function')
  if y_pred is not None:
    ax.scatter(x[:, 0], x[:, 1], y_pred, label='our function')
  plt.legend()
  plt.show()