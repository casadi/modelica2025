dae = casadi.DaeBuilder('racecar', 'resources/kloeser2020.fmu');
x0 = dae.start(dae.x())';                  % Initial state
f = dae.create('f', {'x', 'u'}, {'ode'});  % System dynamics
H = dae.create('H', {'x'}, {'y'});         % System outputs

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

T = opti.variable(); % Integration horizon [s]
opti.subject_to(T >= 0);
opti.set_initial(T, 2);

N = 40;                 % Number of integration intervals
dt = T / N;             % Length of one interval

xk = casadi.MX(x0);

x_traj = {xk};       % Place to store the state solution trajectory
u_traj = {};
for k=1:N % Loop over integration intervals

    % Decision variables for helper states at each collocation point
    Xc = opti.variable(nx, degree);

    % Decision variables for control (constant over interval)
    uk = opti.variable(nu);
    
    % Slope of polynomial at collocation points
    Z  = [xk Xc];
    Pidot = (Z * C)/dt;

    % Collocation constraints (slope matching with dynamics)
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

% Final constraint s(T)=4*pi
opti.subject_to(x_traj(indx('s'),end)==4*pi);

% Path constraint on n: track limits
opti.subject_to(-0.2 <= x_traj(indx('n'), :)         <= 0.2);  % Stay in lane
opti.subject_to(-1   <= x_traj(indx('D'), :)         <= 1);
opti.subject_to(-0.8 <= x_traj(indx('delta'), :)     <= 0.8);
opti.subject_to(-20  <= u_traj(indu('D_der'), :)     <= 20);
opti.subject_to(-4   <= u_traj(indu('delta_der'), :) <= 4);
opti.subject_to(-10  <= y_traj(indy('acc_long'), :)  <= 6);
opti.subject_to(-10  <= y_traj(indy('acc_lat'), :)   <= 10);

opti.minimize(T);
options = struct;
options.ipopt.hessian_approximation = 'limited-memory';
opti.solver('ipopt', options);
sol = opti.solve();  % Optimize

x_opt = sol.value(x_traj);
y_opt = sol.value(y_traj);

%% Post-processing

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
