%% ===============================================================
%               NMPC SETUP FOR THREE-TANK SYSTEM
%        (CasADi + MATLAB + Simulink Integration)
% ================================================================

disp('Inicializando NMPC con CasADi...'); %[output:04dcc8a8]

%% ============================
% 1. Importar CasADi
% =====================a=======
addpath('C:\Users\PC\Downloads\Tesis\Codigo\watertank');
import casadi.*

%% ============================
% 2. Parámetros del NMPC
% ============================
parSys.N_mpc = 20;
parSys.Q_mpc = 200;
parSys.R_mpc = 0.1;
parSys.dumax = 0.00005;

nx = 3;   
nu = 2;   

%% ============================
% 3. Variables simbólicas
% ============================
x = SX.sym('x', nx);
u = SX.sym('u', nu);
yref = SX.sym('yref');

Ad = parSys.Ad;
Bd = parSys.Bd;

%% ============================
% 4. Modelo dinámico discreto
% ============================
f_dyn = Ad*x + Bd*u;
f_dyn_fun = Function('f_dyn_fun', {x, u}, {f_dyn});

%% ============================
% 5. Formulación NMPC
% ============================

% U ahora es un vector columna grande
U = SX.sym('U', nu*parSys.N_mpc, 1);

X = SX.sym('X', nx, parSys.N_mpc+1);

X0 = SX.sym('X0', nx);

J = 0;
g = [];

X(:,1) = X0;

for k = 1:parSys.N_mpc

    % extraer uk desde el vector U
    uk = U((k-1)*nu+1 : k*nu);

    % dinámica
    X(:,k+1) = f_dyn_fun(X(:,k), uk);

    % salida
    yk = X(2,k);

    % costo
    J = J + parSys.Q_mpc*(yk - yref)^2 + parSys.R_mpc*(uk(1))^2;

    % restricción du
    if k > 1
        uk_prev = U((k-2)*nu+1 : (k-1)*nu);
        g = [g ; uk(1) - uk_prev(1)];
    end
end

%% ============================
% 6. Límites
% ============================

u_min = [0; 0];
u_max = [parSys.qi1max; parSys.qi3max];

lbw = [];
ubw = [];

for k = 1:parSys.N_mpc
    lbw = [lbw ; u_min];
    ubw = [ubw ; u_max];
end

lbg = -parSys.dumax * ones(parSys.N_mpc-1,1);
ubg =  parSys.dumax * ones(parSys.N_mpc-1,1);

%% ============================
% 7. Crear solver CasADi
% ============================
opts = struct;
opts.ipopt.print_level = 0;
opts.print_time = 0;

nlp = struct;
nlp.x = U;                     % <-- AHORA CORRECTO
nlp.f = J;
nlp.g = g;
nlp.p = [X0; yref];

solver_nmpc = nlpsol('solver_nmpc', 'ipopt', nlp, opts);

%% ============================
% 8. Guardar objeto en parSys
% ============================
parSys.solver_nmpc = solver_nmpc;
parSys.lbw = lbw;
parSys.ubw = ubw;
parSys.lbg = lbg;
parSys.ubg = ubg;

disp('NMPC listo para Simulink.'); %[output:8143ef91]


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:04dcc8a8]
%   data: {"dataType":"text","outputData":{"text":"Inicializando NMPC con CasADi...\n","truncated":false}}
%---
%[output:8143ef91]
%   data: {"dataType":"text","outputData":{"text":"NMPC listo para Simulink.\n","truncated":false}}
%---
