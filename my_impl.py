import numpy as np
from matplotlib import pyplot as plt


def forward_solve(x, f, bc1, bc2):
  """Defines a forward solve for a 1D steady-state heat equation"""
  h = (x[1] - x[0])
  num_space_points = len(x) - 2

  d = np.empty(num_space_points)
  d.fill(2)
  subd = np.empty(num_space_points - 1); subd.fill(-1)
  supd = subd

  D_matrix = (np.diag(d) + np.diag(subd, -1) + np.diag(supd, 1)) / h**2

  f_trimmed = f[1:-1]

  f_trimmed[0] -= bc1 / h**2
  f_trimmed[1] -= bc2 / h**2

  u_trimmed = np.linalg.solve(D_matrix, f_trimmed)
  u = np.concatenate(([bc1], u_trimmed, [bc2]))
  return u


def plot_solution():
  bc1 = 0
  bc2 = 0
  x = np.linspace(0,1, 50)
  f = -x**2 + 1
  plt.plot(x, f)
  plt.show()
  u = forward_solve(x, f, bc1, bc2)
  plt.plot(x, u)
  plt.show()





plot_solution()
