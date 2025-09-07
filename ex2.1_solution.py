import casadi as ca
import numpy as np
import pylab as plt

dae = ca.DaeBuilder('racecar', 'resources/kloeser2020.fmu')
x0 = dae.start(dae.x())                  # Initial state
f = dae.create('f', ['x', 'u'], ['ode']) # System dynamics

T = 2       # Integration horizon [s]
N = 20      # Number of integration intervals
dt = T/N    # Length of one interval

nx = dae.nx() # Number of states

xvar = dae.x()

# Numeric coefficient matrices for collocation
degree = 3
method = 'radau'
tau = ca.collocation_points(degree,method)
[C,D,B] = ca.collocation_coeff(tau)

opti = ca.Opti() # Opti context

xk = ca.MX(x0)

x_traj = [xk]       # Place to store the state solution trajectory
for k in range(N): # Loop over integration intervals

    # Decision variables for helper states at each collocation point
    Xc = opti.variable(nx, degree)
    
    # Slope of polynomial at collocation points
    Z  = ca.horzcat(xk,Xc)
    Pidot = (Z @ C)/dt

    # Collocation constraints (slope matching with dynamics)
    opti.subject_to(Pidot==f(x=Xc)["ode"])
    
    # Continuity constraints
    xk_next = opti.variable(nx)
    opti.subject_to(Z @ D==xk_next)

    # Initial guesses
    opti.set_initial(Xc, ca.repmat(x0,1,degree))
    opti.set_initial(xk_next, x0)

    xk = xk_next
    x_traj.append(xk)
x_traj = ca.hcat(x_traj)

opti.minimize(0)
options = {'ipopt.hessian_approximation':'limited-memory'}
opti.solver('ipopt',options)
sol = opti.solve() # Optimize

x_opt = sol.value(x_traj)

# Post processing

yvar = dae.y()

print("distance covered:", x_opt[xvar.index('s'),-1])

H = dae.create('H', ['x'], ['y'])   # System outputs
y_opt = H(x=x_opt)["y"]

c   = y_opt[[yvar.index('c_x'),  yvar.index('c_y')]  ,:]
e_n = y_opt[[yvar.index('e_n_x'),yvar.index('e_n_y')],:]
p   = y_opt[[yvar.index('p_x'),  yvar.index('p_y')]  ,:]

n_max = 0.2

p_L = c + n_max*e_n
p_R = c - n_max*e_n

plt.plot(c[0,:].T,c[1,:].T,'k--')
plt.plot(p_L[0,:].T,p_L[1,:].T,'r')
plt.plot(p_R[0,:].T,p_R[1,:].T,'r',)

plt.plot(p[0,:].T,p[1,:].T,'b')
plt.axis('equal')

# plt.savefig('resources/car_2.1.pdf')

plt.show()
