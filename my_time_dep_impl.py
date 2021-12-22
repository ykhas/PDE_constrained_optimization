import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as optim

def finite_difference_euler(x, t, u0, heat_source):
    '''Return the numerical solution using a forward euler scheme for a given x and t'''

    h = x[1] - x[0]
    k = t[1] - t[0]
    num_space_points = len(x) - 2
    # create finite difference matrix, assuming boundary conditions are 0
    d = np.empty(num_space_points); d.fill(-2)
    subd = np.empty(num_space_points - 1); subd.fill(1)
    supd = np.empty(num_space_points - 1); supd.fill(1)

    matrix = np.eye(num_space_points) + k / h**2 * ( np.diag(d) + np.diag(subd, -1) + np.diag(supd, 1)); matrix

    def compute_next_time_step(prev_result, matrix):
        return matrix.dot(prev_result) + heat_source

    solution = [u0[1:-1]] # populate with initial condition
    for i in range(1, len(t)):
        solution.append(compute_next_time_step(solution[i-1], matrix))
    return solution

def func_to_minimize(heat_source, u0, u_final_true, x, t):
  u = finite_difference_euler(x, t, u0, heat_source)
  return np.sum((u[-1] - u_final_true)**2) 

x = np.linspace(0,1,10)
t = np.linspace(0,1,1000)
u0 = np.sin(np.pi*x)
heat_source = 0.04*((x - 0.5)**2)[1:-1] 
u = finite_difference_euler(x, t, np.sin(np.pi * x), heat_source)
uf = u[-1] 
heat_source_guess = 0*x[1:-1] + 10
result = optim.minimize(func_to_minimize, heat_source_guess, args=(u0, uf, x, t), options={'disp':True})

print(result.x)
print(heat_source)

u_computed = finite_difference_euler(x, t, np.sin(np.pi * x), result.x)
plt.plot(x[1:-1], u_computed[-1], 'o', label='computed u')
plt.plot(x[1:-1], u[-1], '.', label = 'true u')
plt.legend()
plt.show()

plt.plot(x[1:-1], result.x, 'o', label='computed y')
plt.plot(x[1:-1], heat_source, '.', label = 'true y')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X,T = np.meshgrid(x[1:-1], t)
ax.scatter(X, T, u_computed)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('temp')
plt.show()


