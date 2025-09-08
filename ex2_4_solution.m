dae = casadi.DaeBuilder('racecar', 'resources/kloeser2020.fmu');
x0 = dae.start(dae.x())';                  % Initial state
f = dae.create('f', {'x', 'u'}, {'ode'});  % System dynamics
H = dae.create('H', {'x'}, {'y'});         % System outputs

T = 0.5;    % Integration horizon [s]
N = 10;     % Number of integration intervals
dt = T / N; % Length of one interval

nx = dae.nx; % Number of states
nu = dae.nu; % Number of controls

% A helper function to get the index of a state with a particular name
indx = @(name) find(strcmp(cellstr(dae.x), name));
indu = @(name) find(strcmp(cellstr(dae.u), name));
indy = @(name) find(strcmp(cellstr(dae.y), name));

% Numeric coefficient matrices for collocation
degree = 3;
method = 'radau';
tau = casadi.collocation_points(degree,method);
[C,D,B] = casadi.collocation_coeff(tau);

opti = casadi.Opti(); % Opti context

x0_bar = opti.parameter(nx);
opti.set_value(x0_bar, x0);

xk = opti.variable(nx);
opti.subject_to(xk == x0_bar);

x_traj = {xk};       % Place to store the state solution trajectory
u_traj = {};
for k=1:N % Loop over integration intervals

    % Value of the states at each collocation point
    Xc = opti.variable(nx, degree);

    % Decision variables for control (constant over interval)
    uk = opti.variable(nu);
    
    % Value of the state derivatives at each collocation point
    Z  = [xk Xc];
    Pidot = (Z * C)/dt;

    % Collocation constraints
    opti.subject_to(Pidot==getfield(f('x',Xc,'u',uk),'ode'));
    
    % Continuity constraints
    xk_next = opti.variable(nx);
    opti.subject_to(Z * D==xk_next);

    % Initial guesses
    opti.set_initial(Xc, repmat(x0,1,degree));
    opti.set_initial(xk_next, x0);

    xk = xk_next;
    x_traj{end+1} = xk;
    u_traj{end+1} = uk;
end
x_traj = [x_traj{:}];
u_traj = [u_traj{:}];
y_traj = getfield(H('x',x_traj),'y');

% Path constraint on n: track limits
opti.subject_to(-0.2 <= x_traj(indx('n'), :)         <= 0.2);  % Stay in lane
opti.subject_to(-1   <= x_traj(indx('D'), :)         <= 1);
opti.subject_to(-0.8 <= x_traj(indx('delta'), :)     <= 0.8);
opti.subject_to(-20  <= u_traj(indu('D_der'), :)     <= 20);
opti.subject_to(-4   <= u_traj(indu('delta_der'), :) <= 4);
opti.subject_to(-10  <= y_traj(indy('acc_long'), :)  <= 6);
opti.subject_to(-10  <= y_traj(indy('acc_lat'), :)   <= 10);

opti.minimize(-x_traj(indx('s'), end));
options = struct;
options.ipopt.hessian_approximation = 'limited-memory';
options.ipopt.tol = 1e-5;
opti.solver('ipopt', options);

% Convert to MPC function
mpc_step = opti.to_function('mpc_step', {x0_bar}, {x_traj, u_traj});

% Simulation
simulator = casadi.integrator('simulator', 'cvodes', dae.create(), struct('tf', dt));
x_sim = x0;
y_sim = getfield(H('x', x_sim), 'y');
y_traj_sim = {y_sim};

for k = 1:25
    % Compute optimal trajectories
    [x_opt, u_opt] = mpc_step(x_sim);
    
    % Apply the first control input
    u_apply = u_opt(:, 1);
    
    % Simulate the system
    x_sim = getfield(simulator('x0', x_sim, 'u', u_apply),'xf');
    y_sim = getfield(H('x', x_sim), 'y');
    y_traj_sim{end + 1} = y_sim;
end
y_traj_sim = [y_traj_sim{:}];

%% Post-processing

c   = full(y_traj_sim([indy('c_x'),  indy('c_y')],   :));
e_n = full(y_traj_sim([indy('e_n_x'),indy('e_n_y')], :));
p   = full(y_traj_sim([indy('p_x'),  indy('p_y')],   :));

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
