import casadi as ca
import numpy as np
import pylab as plt

dae = ca.DaeBuilder('racecar', 'resources/kloeser2020.fmu')
x0 = dae.start(dae.x())                  # Initial state
f = dae.create('f', ['x', 'u'], ['ode']) # System dynamics
H = dae.create('H', ['x'], ['y'])        # System outputs

nx = dae.nx() # Number of states
nu = dae.nu() # Number of controls

xvar = dae.x()
uvar = dae.u()
yvar = dae.y()

# Numeric coefficient matrices for collocation
degree = 3
method = 'radau'
tau = ca.collocation_points(degree,method)
[C,D,B] = ca.collocation_coeff(tau)

opti = ca.Opti() # Opti context

T = opti.variable() # Integration horizon [s]
opti.subject_to(T>=0)
opti.set_initial(T,2)

N = 40      # Number of integration intervals
dt = T/N    # Length of one interval

xk = ca.MX(x0)

x_traj = [xk]       # Place to store the state solution trajectory
u_traj = []
for k in range(N): # Loop over integration intervals

    # Value of the states at each collocation point
    Xc = opti.variable(nx, degree)

    # Decision variables for control (constant over interval)
    uk = opti.variable(nu)
    
    # Value of the state derivatives at each collocation point
    Z  = ca.horzcat(xk,Xc)
    Pidot = (Z @ C)/dt

    # Collocation constraints
    opti.subject_to(Pidot==f(x=Xc,u=uk)["ode"])
    
    # Continuity constraints
    xk_next = opti.variable(nx)
    opti.subject_to(Z @ D==xk_next)

    # Initial guesses
    opti.set_initial(Xc, ca.repmat(x0,1,degree))
    opti.set_initial(xk_next, x0)

    xk = xk_next
    x_traj.append(xk)
    u_traj.append(uk)
x_traj = ca.hcat(x_traj)
u_traj = ca.hcat(u_traj)
y_traj = H(x=x_traj)["y"]

# Final constraint s(T)=4*pi
opti.subject_to(x_traj[xvar.index('s'),-1]==4*np.pi)

# Path constraint on n: track limits
opti.subject_to(-0.2 <= (x_traj[xvar.index('n'),:]         <= 0.2)) # Stay in lane
opti.subject_to(-1   <= (x_traj[xvar.index('D'),:]         <= 1))
opti.subject_to(-0.8 <= (x_traj[xvar.index('delta'),:]     <= 0.8))
opti.subject_to(-20  <= (u_traj[uvar.index('D_der'),:]     <= 20))
opti.subject_to(-4   <= (u_traj[uvar.index('delta_der'),:] <= 4))
opti.subject_to(-10  <= (y_traj[yvar.index('acc_long'),:]  <= 6))
opti.subject_to(-10  <= (y_traj[yvar.index('acc_lat'),:]   <= 10))

opti.minimize(T)
options = {'ipopt.hessian_approximation':'limited-memory'}
opti.solver('ipopt',options)
sol = opti.solve() # Optimize

x_opt = sol.value(x_traj)
y_opt = sol.value(y_traj)

# Post processing

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

# plt.savefig('resources/car_2.3.pdf')

plt.show()
