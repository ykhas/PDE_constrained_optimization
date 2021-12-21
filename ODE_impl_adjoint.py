from typing import overload
import numpy as np
import scipy.optimize as optim
from matplotlib import pyplot as plt

Beta = 0.000025 * 0
num_of_times = 0
def create_D_matrix_full_size(x):
  D_matrix, h = create_D_matrix(x)
  first_col = np.empty(len(x) - 2); first_col.fill(0); first_col[0] = -1/h**2
  last_col = np.empty(len(x) - 2); last_col.fill(0); last_col[-1] = -1/h**2
  first_row = np.empty(len(x)); first_row.fill(0); first_row[0] = -h**2
  last_row = np.empty(len(x)); last_row.fill(0); last_row[-1] = -h**2
  return np.concatenate(([first_row], np.concatenate(([first_col], D_matrix.T, [last_col])).T, [last_row])), h

def create_D_matrix(x):
  h = (x[1] - x[0])
  num_space_points = len(x) - 2

  d = np.empty(num_space_points)
  d.fill(2)
  subd = np.empty(num_space_points - 1); subd.fill(-1)
  supd = subd

  D_matrix = (np.diag(d) + np.diag(subd, -1) + np.diag(supd, 1)) / h**2
  return D_matrix, h

def forward_solve(f, bc1, bc2, D_matrix, h):
  """Defines a forward solve for a 1D steady-state heat equation"""

  f_trimmed = f[1:-1]

  f_trimmed[0] -= bc1 / h**2
  f_trimmed[1] -= bc2 / h**2

  u_trimmed = np.linalg.solve(-D_matrix, f_trimmed)
  u = np.concatenate(([bc1], u_trimmed, [bc2]))
  return u

def func_to_minimize(f_try, u_tilde, bc1, bc2, D_matrix, h, x):
  global Beta
  u = forward_solve(f_try, bc1, bc2, D_matrix, h)
  global num_of_times
  num_of_times+=1
  return np.sum((u - u_tilde)**2) + Beta*np.sum(f_try**2)

def jacobian(f_try, u_tilde, bc1, bc2, D_matrix_small, h, x):
  global Beta
  grad_cost_y = Beta * 2 * (f_try)
  u = forward_solve(f_try, bc1, bc2, D_matrix_small, h)
  grad_cost_u = 2 * (u - u_tilde)

  grad_f_u = create_D_matrix_full_size(x)[0] 
  grad_f_y = f_try * 0 + 1

  lambda_T = np.linalg.solve(grad_f_u, grad_cost_u)

  return grad_cost_y - lambda_T * grad_f_y

def primal_minimization():
  bc1 = 0
  bc2 = 0
  x = np.linspace(0,1, 100)
  f = np.sin(5* np.pi * x)
  f_try =  np.random.rand(100)
  D_matrix_small, h = create_D_matrix(x)
  u_tilde = forward_solve(f, bc1, bc2, D_matrix_small, h)
  result = optim.minimize(func_to_minimize, f_try, args=(u_tilde, bc1, bc2, D_matrix_small, h, x), method='CG',
                          jac=jacobian, 
                          options={'disp':True})

  # now we test it out
  u = forward_solve(result.x, bc1, bc2, D_matrix_small, h)
  plt.plot(x, u, label='u')
  plt.plot(x, u_tilde, label='u_tilde')
  plt.legend()
  plt.show()

  plt.plot(x, result.x, label='computed y')
  plt.plot(x, f, label='real y')
  plt.legend()
  plt.show()

  print("max difference between solutions is {}".format(np.abs(np.max(u - u_tilde))))


def plot_solution():
  primal_minimization()





plot_solution()
print(num_of_times)