import numpy as np
import scipy.optimize as optim
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

  u_trimmed = np.linalg.solve(-D_matrix, f_trimmed)
  u = np.concatenate(([bc1], u_trimmed, [bc2]))
  return u

def create_f(a, x):
  return a * x**2 + 1

def func_to_minimize(f_try, u_tilde, x, bc1, bc2):
  u = forward_solve(x, f_try, bc1, bc2)
  return np.sum((u - u_tilde)**2)

def primal_minimization():
  bc1 = 0
  bc2 = 0
  x = np.linspace(0,1, 100)
  f = np.sin(5* np.pi * x)
  f_try =  np.random.rand(100)
  u_tilde = forward_solve(x, f, bc1, bc2)
  result = optim.minimize(func_to_minimize, f_try, args=(u_tilde, x, bc1, bc2), method='CG', options={'disp':True})

  # now we test it out
  u = forward_solve(x, result.x, bc1, bc2)
  plt.plot(x, u, label='u')
  plt.plot(x, u_tilde, label='u_tilde')
  plt.legend()
  plt.show()

  plt.plot(x, result.x, label='computed f')
  plt.plot(x, f, label='real f')
  plt.legend()
  plt.show()

  print("max difference between solutions is {}".format(np.abs(np.max(u - u_tilde))))


def plot_solution():
  primal_minimization()





plot_solution()