dae = casadi.DaeBuilder('cstr', 'resources/cstr.fmu');
dae.disp(true);







%% Task 2
f = dae.create('f', {'x', 'u'}, {'ode'});
disp(f)

f([1;2],[3,4,5])



%%  1.3

opti = casadi.Opti();      % Opti context

opti.minimize(0);          % Trivial objective (no degrees of freedom)
c_A = opti.variable();     % Declare decision variables
q_in = opti.variable();

x = [c_A, 0.3];            % State vector
u = [1, 0, q_in'];         % Control vector
xdot = f(x, u);            % Evaluate f symbolically
opti.subject_to(xdot == 0);  % Constrain state derivatives to be zero

opti.set_initial(c_A, 1);  % Set initial guesses
opti.set_initial(q_in, 1);
    
opti.solver('ipopt');      % Choose solver
sol = opti.solve();        % Optimize

c_A_opt = sol.value(c_A)   % Get solution
q_in_opt = sol.value(q_in)

%% 1.4

dae = casadi.DaeBuilder('racecar', 'resources/kloeser2020.fmu');
f = dae.create('f', {'x', 'u'}, {'ode'});

% Note that the controls of the model are rates
u_names = cellstr(dae.u);
for i=1:length(u_names)
    disp([u_names{i} ': ' dae.description(u_names{i})])
end

opti = casadi.Opti();      % Opti context

opti.minimize(0);          % Trivial objective (no degrees of freedom)
D = opti.variable();       % Declare decision variable D

xvar = dae.x();            % Form state and control vectors
x = casadi.MX(dae.start(xvar));

% A helper function to get the index of a state with a particular name
x_names = cellstr(dae.x);
ind = @(name) find(strcmp(x_names, name));

x(ind('D')) = D;
u = 0;
xdot = f(x, u); % Evaluate the state derivatives symbolically
% Constrain state derivative to be zero
opti.subject_to(xdot(ind('v')) == 0); 

opti.set_initial(D, 1);    % Set initial guesses
opti.solver('ipopt');      % Choose solver
sol = opti.solve();        % Optimize
D_opt = sol.value(D)       % Get the optimal parameters