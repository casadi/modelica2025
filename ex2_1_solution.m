dae = casadi.DaeBuilder('racecar', 'resources/kloeser2020.fmu');
x0 = dae.start(dae.x())';                  % Initial state
f = dae.create('f', {'x', 'u'}, {'ode'});  % System dynamics

T = 2;       % Integration horizon [s]
N = 20;      % Number of integration intervals
dt = T/N;    % Length of one interval

nx = dae.nx; % Number of states

% A helper function to get the index of a state with a particular name
indx = @(name) find(strcmp(cellstr(dae.x), name));
indy = @(name) find(strcmp(cellstr(dae.y), name));

% Numeric coefficient matrices for collocation
degree = 3;
method = 'radau';
tau = casadi.collocation_points(degree,method);
[C,D,B] = casadi.collocation_coeff(tau);

opti = casadi.Opti(); % Opti context

xk = casadi.MX(x0);

x_traj = {xk};       % Place to store the state solution trajectory
for k=1:N % Loop over integration intervals

    % Decision variables for helper states at each collocation point
    Xc = opti.variable(nx, degree);
    
    % Slope of polynomial at collocation points
    Z  = [xk Xc];
    Pidot = (Z * C)/dt;

    % Collocation constraints (slope matching with dynamics)
    opti.subject_to(Pidot==getfield(f('x',Xc),'ode'));
    
    % Continuity constraints
    xk_next = opti.variable(nx);
    opti.subject_to(Z * D==xk_next);

    % Initial guesses
    opti.set_initial(Xc, repmat(x0,1,degree));
    opti.set_initial(xk_next, x0);

    xk = xk_next;
    x_traj{end+1} = xk;
end
x_traj = [x_traj{:}];

opti.minimize(0);
options = struct;
options.ipopt.hessian_approximation = 'limited-memory';
opti.solver('ipopt', options);
sol = opti.solve();  % Optimize

x_opt = sol.value(x_traj);

%% Post-processing




fprintf('distance covered: %.6g\n', x_opt(indx('s'), end));

% Output function
H = dae.create('H', {'x'}, {'y'});
y_opt = getfield(H('x', x_opt), 'y');

c   = full(y_opt([indy('c_x'),  indy('c_y')],   :));
e_n = full(y_opt([indy('e_n_x'),indy('e_n_y')], :));
p   = full(y_opt([indy('p_x'),  indy('p_y')],   :));

n_max = 0.2;
p_L = c + n_max * e_n;
p_R = c - n_max * e_n;

%% Plot
figure; hold on;
plot(c(1,:).',   c(2,:).',   'k--', 'LineWidth', 1.0);
plot(p_L(1,:).', p_L(2,:).', 'r',   'LineWidth', 1.2);
plot(p_R(1,:).', p_R(2,:).', 'r',   'LineWidth', 1.2);
plot(p(1,:).',   p(2,:).',   'b',   'LineWidth', 1.4);
axis equal;
