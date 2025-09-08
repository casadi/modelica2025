import pylab as plt
import casadi as ca
import numpy as np

dae = ca.DaeBuilder('kloeser2020', 'resources/kloeser2020.fmu')

x0 = dae.start(dae.x())

# Create a function that evaluates the ODE right hand sides and outputs
F = dae.create('F', ['x', 'u'], ['ode', 'y'])

# Solve with rockit
import rockit

ocp = rockit.Ocp(T=0.5)   # horizon length

# All model variables
v = dict()
for n in dae.x(): v[n] = ocp.state()
for n in dae.u(): v[n] = ocp.control()

# Evaluate F symbolically to get derivatives and outputs
x = ca.vertcat(*[v[n] for n in dae.x()])
u = ca.vertcat(*[v[n] for n in dae.u()])
der_x, y = F(x, u)
der_x = ca.vertsplit(der_x)
y = ca.vertsplit(y)

# Collect derivatives and outputs
for k, n in enumerate(dae.x()): ocp.set_der(v[n], der_x[k])
for k, n in enumerate(dae.y()): v[n] = y[k]


ocp.add_objective(ocp.at_tf(-v['s'])) # Maximize progress (s)

# Add bounds (track and actuator limits)
ocp.subject_to(-0.2 <= (v['n']         <= 0.2)) # Stay in lane
ocp.subject_to(-1   <= (v['D']         <= 1  ))
ocp.subject_to(-20  <= (v['D_der']     <= 20 ))
ocp.subject_to(-0.8 <= (v['delta']     <= 0.8))
ocp.subject_to(-4   <= (v['delta_der'] <= 4  ))
ocp.subject_to(-10  <= (v['acc_long']  <= 6  ))
ocp.subject_to(-10  <= (v['acc_lat']   <= 10 ))

ocp.subject_to(ocp.at_t0(ocp.x)==x0) # Start at standstill


ocp.set_initial(v['v'],1)

ocp.method(rockit.MultipleShooting(N=40))

#ocp.solver('ipopt')
ocp.solver('fatrop')

sol = ocp.solve()

print("Time spent in solver [s]: ", sol.stats['t_wall_total']-sol.stats['t_wall_nlp_f']-sol.stats['t_wall_nlp_g']-sol.stats['t_wall_nlp_grad_f']-sol.stats['t_wall_nlp_hess_l']-sol.stats['t_wall_nlp_jac_g'])

ts, v_sol = sol.sample(v['v'],grid='control')
plt.plot(ts,v_sol,'bo')

ts_fine, v_sol_fine = sol.sample(v['v'],grid='integrator',refine=10)
plt.plot(ts_fine,v_sol_fine,'b')
plt.xlabel('Time [s]')
plt.ylabel('v [m/s]')

plt.show()


