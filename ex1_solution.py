import casadi as ca
dae = ca.DaeBuilder('cstr', 'resources/cstr.fmu')
dae.disp(True)






## Task 2
f = dae.create('f', ['x', 'u'], ['ode'])
print(f)

print(f(ca.vertcat(1,2),ca.vertcat(3,4,5)))





## Task 3
opti = ca.Opti()           # Opti context

opti.minimize(0)           # Trivial objective (no degrees of freedom)
c_A  = opti.variable()     # Decision variables
q_in = opti.variable()

x = ca.vertcat(c_A, 0.3)   # State vector
u = ca.vertcat(1, 0, q_in) # Control vector
xdot = f(x, u)             # Evaluate f symbolically
opti.subject_to(xdot == 0) # Constrain state derivatives to be zero

opti.set_initial(c_A, 1)   # Set initial guesses
opti.set_initial(q_in, 1)

opti.solver('ipopt')       # Choose solver
sol = opti.solve()         # Optimize

c_A_opt = sol.value(c_A)   # Get solution
q_in_opt = sol.value(q_in)
print(f'c_A = {c_A_opt}, q_in = {q_in_opt}')






# Task 4


  
dae = ca.DaeBuilder('racecar', 'resources/kloeser2020.fmu')
f = dae.create('f', ['x', 'u'], ['ode'])

# Note that the controls of the model are rates
for u in dae.u():
  print(u,":",dae.description(u))

# Opti context
opti = ca.Opti()           # Opti context
opti.minimize(0)           # Trivial objective (no degrees of freedom)
D = opti.variable()        # Decision variable: D

xvar = dae.x()             # Form state and control vectors
x = ca.MX(dae.start(xvar))
x[xvar.index('D')] = D
u = 0
xdot = f(x, u)             # Evaluate f symbolically
opti.subject_to(xdot[xvar.index('v')] == 0) # Constraint: dot(v) == 0


opti.set_initial(D, 1)     # Set initial guesses
opti.solver('ipopt')       # Choose solver
sol = opti.solve()         # Optimize
D_opt = sol.value(D)       # Get the optimal parameters

print(f'D = {D_opt}')

print("x_dot",sol.value(xdot))
