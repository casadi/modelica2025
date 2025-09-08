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

T = 0.5     # Integration horizon [s]
N = 10      # Number of integration intervals
dt = T/N    # Length of one interval

x0_bar = opti.parameter(nx)
opti.set_value(x0_bar, x0)

xk = opti.variable(nx)
opti.subject_to(xk==x0_bar)

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

# Path constraint on n: track limits
opti.subject_to(-0.2 <= (x_traj[xvar.index('n'),:]         <= 0.2)) # Stay in lane
opti.subject_to(-1   <= (x_traj[xvar.index('D'),:]         <= 1))
opti.subject_to(-0.8 <= (x_traj[xvar.index('delta'),:]     <= 0.8))
opti.subject_to(-20  <= (u_traj[uvar.index('D_der'),:]     <= 20))
opti.subject_to(-4   <= (u_traj[uvar.index('delta_der'),:] <= 4))
opti.subject_to(-10  <= (y_traj[yvar.index('acc_long'),:]  <= 6))
opti.subject_to(-10  <= (y_traj[yvar.index('acc_lat'),:]   <= 10))

opti.minimize(-x_traj[xvar.index('s'),-1])
options = {'ipopt.hessian_approximation':'limited-memory', "ipopt.tol": 1e-5}
opti.solver('ipopt',options)

mpc_step = opti.to_function('mpc_step', [x0_bar, x_traj, u_traj], [x_traj, u_traj])
[x_opt,u_opt] = mpc_step(x0,0,0)
print("x_opt(T): ", x_opt[:,-1])

simulator = ca.integrator('simulator', 'cvodes', dae.create(), 0, dt)

y_traj = [H(x0)]
for k in range(25):
  # Compute optimal trajectories
  [x_opt,u_opt] = mpc_step(x0, x_opt, u_opt)
  
  # What part of the trajectory to apply?
  u_apply = u_opt[:,0]
  
  # Plant model
  x0 = simulator(x0=x0,u=u_apply)["xf"]
  
  y_traj.append(H(x0))
y_opt = ca.hcat(y_traj)

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

# plt.savefig('resources/car_2.4.pdf')

plt.show()
