%% ================================================================
%    Ejercicio 2.2 — NMPC No Lineal con CasADi
%    TU Wien — Optimization-Based Control Methods SS2025
%    VERSION 2: seguimiento mejorado
%% ================================================================
clear; clc; close all;

addpath('C:\Users\PC\Downloads\Tesis\Codigo\watertank');
import casadi.*

%% ============================
% 1. Parametros fisicos (Tabla 1.1)
% ============================
p.Atank    = 1.539e-2;
p.rho      = 997.0;
p.eta      = 8.9e-4;
p.g        = 9.81;
p.alphao1  = 0.0583;   p.Do1 = 15e-3;
p.alphao2  = 0.1039;   p.A2  = 1.0429e-4;
p.alphao3  = 0.0600;   p.Do3 = 15e-3;
p.alpha120 = 0.3038;   p.D12 = 7.7e-3;  p.A12 = 0.55531e-4; p.lambdac12 = 24000;
p.alpha230 = 0.1344;   p.D23 = 15e-3;   p.A23 = 1.76715e-4; p.lambdac23 = 29600;
p.qmax     = 75e-6;
p.Ts       = 2;

%% ============================
% 2. Verificar estado estacionario para cada referencia
%    Esto nos dice si la referencia es alcanzable
% ============================
fprintf('=== Verificacion de estados estacionarios ===\n'); %[output:76ed3459]
for h2_ref = [0.10, 0.25, 0.15] %[output:group:10c36a4b]
    try
        xSol = fsolve(@(x) nl_ode([x(1); h2_ref; x(2)], [x(3); 0], p), ...
                      [h2_ref; h2_ref; 0.5], ...
                      optimoptions('fsolve','Display','off'));
        fprintf('  h2=%.2fm -> h=[%.4f %.4f %.4f] u1=%.4f\n', ... %[output:9287ce93]
            h2_ref, xSol(1), h2_ref, xSol(2), xSol(3)); %[output:9287ce93]
    catch
        fprintf('  h2=%.2fm -> NO alcanzable\n', h2_ref);
    end
end %[output:group:10c36a4b]

%% ============================
% 3. Punto de operacion en h2=0.25m (referencia mas alta)
% ============================
h2_op = 0.25;
xSol = fsolve(@(x) nl_ode([x(1); h2_op; x(2)], [x(3); 0], p), ...
              [h2_op; h2_op; 0.5], ...
              optimoptions('fsolve','Display','off'));
h1s = xSol(1); h3s = xSol(2); u1s = xSol(3);
p.xR = [h1s; h2_op; h3s];
p.uR = [u1s; 0];
fprintf('\nPunto de operacion h2=%.2fm:\n', h2_op); %[output:2a32a255]
fprintf('  h = [%.4f  %.4f  %.4f] m\n', h1s, h2_op, h3s); %[output:72208380]
fprintf('  u = [%.4f  %.4f]\n', u1s, 0); %[output:26947242]

%% ============================
% 4. Modelo CasADi (ODE no lineal) + RK4
% ============================
nx = 3; nu = 2; Ts = p.Ts;
eps_s = 1e-8;

x_s = casadi.SX.sym('x', nx);
u_s = casadi.SX.sym('u', nu);

h1 = x_s(1); h2 = x_s(2); h3 = x_s(3);
qi1 = p.qmax * u_s(1);
qi3 = p.qmax * u_s(2);

% Suavizado heights
h1p = (h1 + sqrt(h1^2 + eps_s)) / 2;
h2p = (h2 + sqrt(h2^2 + eps_s)) / 2;
h3p = (h3 + sqrt(h3^2 + eps_s)) / 2;

% Flujos outlet
qo1 = p.alphao1 * (p.Do1^2*pi/4) * sqrt(2*p.g*h1p);
qo2 = p.alphao2 * p.A2            * sqrt(2*p.g*h2p);
qo3 = p.alphao3 * (p.Do3^2*pi/4) * sqrt(2*p.g*h3p);

% Flujos coupling
dh12  = h1 - h2; abs12 = sqrt(dh12^2 + eps_s);
l12   = p.D12*p.rho/p.eta*sqrt(2*p.g*abs12);
a12   = p.alpha120*tanh(2*l12/p.lambdac12);
q12   = a12*p.A12*sqrt(2*p.g*abs12)*sign(dh12);

dh23  = h2 - h3; abs23 = sqrt(dh23^2 + eps_s);
l23   = p.D23*p.rho/p.eta*sqrt(2*p.g*abs23);
a23   = p.alpha230*tanh(2*l23/p.lambdac23);
q23   = a23*p.A23*sqrt(2*p.g*abs23)*sign(dh23);

xdot_s = [(qi1 - q12 - qo1)/p.Atank;
          (q12 - q23 - qo2)/p.Atank;
          (qi3 + q23 - qo3)/p.Atank];

f_cas  = casadi.Function('f', {x_s,u_s}, {xdot_s});
k1r    = f_cas(x_s, u_s);
k2r    = f_cas(x_s + Ts/2*k1r, u_s);
k3r    = f_cas(x_s + Ts/2*k2r, u_s);
k4r    = f_cas(x_s + Ts  *k3r, u_s);
F_rk4  = casadi.Function('F', {x_s,u_s}, {x_s + Ts/6*(k1r+2*k2r+2*k3r+k4r)});
cT     = [0 1 0];
fprintf('Modelo RK4 listo.\n'); %[output:2ebc14c3]

%% ============================
% 5. Parametros NMPC
%    CLAVE: q grande, R1 pequeno = seguimiento agresivo
% ============================
N  = 30;
q  = 500;              % peso alto en error de salida
R1 = diag([0.1; 1.0]); % u1 libre, u3 penalizado (u3=0 en SS)
R2 = diag([0.1; 0.1]); % suavizado de cambios

xmin = [0;   0;   0  ];
xmax = [0.4; 0.4; 0.4];
umin = [0;   0  ];
umax = [1;   1  ];

fprintf('q=%g  R1=[%.2f %.2f]  N=%d\n', q, R1(1,1), R1(2,2), N); %[output:32bf13e9]

%% ============================
% 6. OCP — full discretization
% ============================
fprintf('Construyendo OCP...\n'); %[output:370b46d9]

X_d  = casadi.SX.sym('X', nx, N+1);
U_d  = casadi.SX.sym('U', nu, N);
x0_p = casadi.SX.sym('x0', nx);
yr_p = casadi.SX.sym('yr');
up_p = casadi.SX.sym('up', nu);

J=0; g=[]; lbg=[]; ubg=[];

% Condicion inicial
g=[g; X_d(:,1)-x0_p]; lbg=[lbg;zeros(nx,1)]; ubg=[ubg;zeros(nx,1)];

for k = 1:N
    % Dinamica
    g=[g; X_d(:,k+1) - F_rk4(X_d(:,k), U_d(:,k))];
    lbg=[lbg; zeros(nx,1)];
    ubg=[ubg; zeros(nx,1)];

    % Salida y error
    yk = cT * X_d(:,k+1);
    ey = yk - yr_p;

    % Variacion control
    if k == 1
        du = U_d(:,1) - up_p;
    else
        du = U_d(:,k) - U_d(:,k-1);
    end

    % Costo con peso final aumentado (terminal cost)
    if k == N
        J = J + 10*q*ey^2 + U_d(:,k)'*R1*U_d(:,k) + du'*R2*du;
    else
        J = J +    q*ey^2 + U_d(:,k)'*R1*U_d(:,k) + du'*R2*du;
    end
end

w    = [reshape(X_d, nx*(N+1), 1); reshape(U_d, nu*N, 1)];
par_p = [x0_p; yr_p; up_p];
lbw  = [repmat(xmin, N+1, 1); repmat(umin, N, 1)];
ubw  = [repmat(xmax, N+1, 1); repmat(umax, N, 1)];

nlp  = struct('x',w,'f',J,'g',g,'p',par_p);
opts = struct;
opts.ipopt.print_level = 0;
opts.ipopt.max_iter    = 1000;
opts.ipopt.tol         = 1e-5;
opts.print_time        = 0;

solver_nmpc = casadi.nlpsol('solver_nmpc','ipopt',nlp,opts);
fprintf('Solver listo.\n'); %[output:5ec1982e]

%% ============================
% 7. Test de convergencia (paso 3 del ejercicio)
% ============================
fprintf('\n--- Test convergencia ---\n'); %[output:880041fd]
xk_t = [0.31; 0.15; 0.14];
yr_t = 0.10;
up_t = [0.6; 0.6];

w0_t = make_warm_start(xk_t, up_t, N, Ts, p, nx, nu, xmin, xmax);
sol_t = solver_nmpc('x0',w0_t,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                    'p',[xk_t; yr_t; up_t]);
Ut = reshape(full(sol_t.x(nx*(N+1)+1:end)), nu, N);
fprintf('  u* = [%.4f  %.4f]  (uR = %.4f)\n', Ut(1,1), Ut(2,1), p.uR(1)); %[output:2d6ce035]

%% ============================
% 8. Simulacion en bucle cerrado
% ============================
T_sim = 1200;
Nsim  = floor(T_sim/Ts);
t1    = floor(Nsim*0.33);
t2    = floor(Nsim*0.66);

yref_v = [0.10*ones(1,t1), 0.25*ones(1,t2-t1), 0.15*ones(1,Nsim-t2)];

% Condicion inicial: estado estacionario para h2=0.10
xSol0 = fsolve(@(x) nl_ode([x(1);0.10;x(2)],[x(3);0],p), ...
               [0.10;0.10;0.5], optimoptions('fsolve','Display','off'));
x_init = [xSol0(1); 0.10; xSol0(2)];
u_init = [xSol0(3); 0];

X_h = zeros(nx, Nsim+1); X_h(:,1) = x_init;
U_h = zeros(nu, Nsim);
T_h = (0:Nsim)*Ts;

uprev = u_init;
w0    = make_warm_start(x_init, u_init, N, Ts, p, nx, nu, xmin, xmax);

fprintf('\nSimulando %d pasos...\n', Nsim); %[output:1e4ebdad]
tic;
for k = 1:Nsim %[output:group:138bf67f]
    xk = X_h(:,k);
    yr = yref_v(k);

    sol  = solver_nmpc('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                       'p',[xk; yr; uprev]);
    wopt = full(sol.x);

    Xtraj = reshape(wopt(1:nx*(N+1)),    nx, N+1);
    Utraj = reshape(wopt(nx*(N+1)+1:end), nu, N);

    uk        = min(umax, max(umin, Utraj(:,1)));
    U_h(:,k)  = uk;
    uprev     = uk;

    X_h(:,k+1) = max(0, min(0.55, rk4_m(xk, uk, Ts, p)));

    % Warm start desplazado
    Xw = [Xtraj(:,2:end), Xtraj(:,end)];
    Uw = [Utraj(:,2:end), Utraj(:,end)];
    w0 = [reshape(Xw, nx*(N+1),1); reshape(Uw, nu*N,1)];

    if mod(k,50)==0
        fprintf('  t=%5.0fs | h2=%.3fm | ref=%.3fm | u=[%.3f %.3f] | %.0fs\n',... %[output:52514a39]
            k*Ts, X_h(2,k+1), yr, uk(1), uk(2), toc); %[output:52514a39]
    end
end %[output:group:138bf67f]
fprintf('Total: %.1fs\n', toc); %[output:8d15632f]

%% ============================
% 9. Graficas
% ============================
figure('Name','Ej 2.2 NMPC v2','Position',[50 50 950 700]); %[output:4ce646b7]

subplot(3,1,1); hold on; grid on; %[output:4ce646b7]
plot(T_h, X_h(1,:)*100,'b','LineWidth',1.2,'DisplayName','h_1'); %[output:4ce646b7]
plot(T_h, X_h(2,:)*100,'r','LineWidth',2.0,'DisplayName','h_2 (salida)'); %[output:4ce646b7]
plot(T_h, X_h(3,:)*100,'Color',[0 .6 0],'LineWidth',1.2,'DisplayName','h_3'); %[output:4ce646b7]
stairs(T_h(1:end-1), yref_v*100,'k--','LineWidth',1.5,'DisplayName','ref h_2'); %[output:4ce646b7]
ylabel('Altura [cm]'); xlabel('Tiempo [s]'); %[output:4ce646b7]
title('Ejercicio 2.2 — NMPC No Lineal (CasADi + IPOPT)'); %[output:4ce646b7]
legend('Location','northeast'); ylim([0 45]); %[output:4ce646b7]

subplot(3,1,2); hold on; grid on; %[output:4ce646b7]
stairs(T_h(1:end-1), U_h(1,:),'b','LineWidth',1.2,'DisplayName','u_1 (bomba 1)'); %[output:4ce646b7]
stairs(T_h(1:end-1), U_h(2,:),'r','LineWidth',1.2,'DisplayName','u_3 (bomba 3)'); %[output:4ce646b7]
yline(1,'k:'); yline(0,'k:'); %[output:4ce646b7]
ylabel('Control [0-1]'); xlabel('Tiempo [s]'); %[output:4ce646b7]
title('Entradas de control'); %[output:4ce646b7]
legend('Location','northeast'); ylim([-0.05 1.1]); %[output:4ce646b7]

subplot(3,1,3); hold on; grid on; %[output:4ce646b7]
e = X_h(2,1:Nsim) - yref_v;
plot(T_h(1:Nsim), e*100,'r','LineWidth',1.5); %[output:4ce646b7]
yline(0,'k:'); %[output:4ce646b7]
ylabel('Error h_2 [cm]'); xlabel('Tiempo [s]'); %[output:4ce646b7]
title('Error de seguimiento h_2'); %[output:4ce646b7]
ylim([-5 5]); %[output:4ce646b7]

%% ================================================================
% FUNCIONES AUXILIARES
%% ================================================================

function w0 = make_warm_start(x0, u0, N, Ts, p, nx, nu, xmin, xmax)
    Xw = zeros(nx, N+1); Xw(:,1) = x0;
    for i = 1:N
        Xw(:,i+1) = max(xmin, min(xmax, rk4_m(Xw(:,i), u0, Ts, p)));
    end
    w0 = [reshape(Xw, nx*(N+1),1); repmat(u0, N, 1)];
end

function xn = rk4_m(x, u, dt, p)
    k1 = nl_ode(x,           u, p);
    k2 = nl_ode(x+dt/2*k1,  u, p);
    k3 = nl_ode(x+dt/2*k2,  u, p);
    k4 = nl_ode(x+dt*k3,    u, p);
    xn = x + dt/6*(k1+2*k2+2*k3+k4);
end

function f = nl_ode(h, u, p)
    h1 = max(0,h(1)); h2 = max(0,h(2)); h3 = max(0,h(3));
    qi1 = p.qmax*u(1); qi3 = p.qmax*u(2);
    qo1 = p.alphao1*(p.Do1^2*pi/4)*sqrt(2*p.g*(h1+1e-10));
    qo2 = p.alphao2*p.A2          *sqrt(2*p.g*(h2+1e-10));
    qo3 = p.alphao3*(p.Do3^2*pi/4)*sqrt(2*p.g*(h3+1e-10));
    dh12=h1-h2;
    l12=p.D12*p.rho/p.eta*sqrt(2*p.g*abs(dh12)+1e-10);
    a12=p.alpha120*tanh(2*l12/p.lambdac12);
    q12=a12*p.A12*sqrt(2*p.g*abs(dh12)+1e-10)*sign(dh12+1e-15);
    dh23=h2-h3;
    l23=p.D23*p.rho/p.eta*sqrt(2*p.g*abs(dh23)+1e-10);
    a23=p.alpha230*tanh(2*l23/p.lambdac23);
    q23=a23*p.A23*sqrt(2*p.g*abs(dh23)+1e-10)*sign(dh23+1e-15);
    f=[(qi1-q12-qo1)/p.Atank;
       (q12-q23-qo2)/p.Atank;
       (qi3+q23-qo3)/p.Atank];
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
%[output:76ed3459]
%   data: {"dataType":"text","outputData":{"text":"=== Verificacion de estados estacionarios ===\n","truncated":false}}
%---
%[output:9287ce93]
%   data: {"dataType":"text","outputData":{"text":"  h2=0.10m -> h=[0.2735 0.1000 0.0709] u1=0.6894\n  h2=0.25m -> h=[0.6348 0.2500 0.1940] u1=1.0808\n  h2=0.15m -> h=[0.3982 0.1500 0.1113] u1=0.8408\n","truncated":false}}
%---
%[output:2a32a255]
%   data: {"dataType":"text","outputData":{"text":"\nPunto de operacion h2=0.25m:\n","truncated":false}}
%---
%[output:72208380]
%   data: {"dataType":"text","outputData":{"text":"  h = [0.6348  0.2500  0.1940] m\n","truncated":false}}
%---
%[output:26947242]
%   data: {"dataType":"text","outputData":{"text":"  u = [1.0808  0.0000]\n","truncated":false}}
%---
%[output:2ebc14c3]
%   data: {"dataType":"text","outputData":{"text":"Modelo RK4 listo.\n","truncated":false}}
%---
%[output:32bf13e9]
%   data: {"dataType":"text","outputData":{"text":"q=500  R1=[0.10 1.00]  N=30\n","truncated":false}}
%---
%[output:370b46d9]
%   data: {"dataType":"text","outputData":{"text":"Construyendo OCP...\n","truncated":false}}
%---
%[output:5ec1982e]
%   data: {"dataType":"text","outputData":{"text":"Solver listo.\n","truncated":false}}
%---
%[output:880041fd]
%   data: {"dataType":"text","outputData":{"text":"\n--- Test convergencia ---\n","truncated":false}}
%---
%[output:2d6ce035]
%   data: {"dataType":"text","outputData":{"text":"  u* = [0.0000  0.0000]  (uR = 1.0808)\n","truncated":false}}
%---
%[output:1e4ebdad]
%   data: {"dataType":"text","outputData":{"text":"\nSimulando 600 pasos...\n","truncated":false}}
%---
%[output:52514a39]
%   data: {"dataType":"text","outputData":{"text":"  t=  100s | h2=0.100m | ref=0.100m | u=[0.557 0.101] | 1s\n  t=  200s | h2=0.100m | ref=0.100m | u=[0.571 0.103] | 2s\n  t=  300s | h2=0.100m | ref=0.100m | u=[0.572 0.104] | 3s\n  t=  400s | h2=0.101m | ref=0.250m | u=[1.000 1.000] | 3s\n  t=  500s | h2=0.237m | ref=0.250m | u=[0.965 0.363] | 5s\n  t=  600s | h2=0.248m | ref=0.250m | u=[0.713 0.301] | 7s\n  t=  700s | h2=0.248m | ref=0.250m | u=[0.713 0.301] | 8s\n  t=  800s | h2=0.246m | ref=0.150m | u=[0.000 0.000] | 9s\n  t=  900s | h2=0.154m | ref=0.150m | u=[0.836 0.155] | 10s\n  t= 1000s | h2=0.150m | ref=0.150m | u=[0.701 0.140] | 11s\n  t= 1100s | h2=0.150m | ref=0.150m | u=[0.683 0.136] | 12s\n  t= 1200s | h2=0.150m | ref=0.150m | u=[0.682 0.136] | 13s\n","truncated":false}}
%---
%[output:8d15632f]
%   data: {"dataType":"text","outputData":{"text":"Total: 12.5s\n","truncated":false}}
%---
%[output:4ce646b7]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgEAAAF6CAYAAACA64v+AAAAAXNSR0IArs4c6QAAIABJREFUeF7tvQ2wVsWZLvrikETiz40oRnErGwnMpJIZ50qmhuKehE0cKrmpQE0ZJ\/yUd2AHPcQTIScF8msETAADkhh0xmEQN6gXSJ3EnIiVieUhsPVIyExMyamZyQQrsFGyTUKI5wYdPTk63HnW5v3o3Xutb\/1\/31rdz6qyNu69ulf302+\/79Nvv\/32sDNnzpwRPkSACBABIkAEiIB3CAwjCfBuzNlhIkAEiAARIAIBAiQBFAQiQASIABEgAp4iQBLg6cCz20SACBABIkAESAIoA0SACBABIkAEPEWAJMDTgWe3iQARIAJEgAiQBFAGiAARIAJEgAh4igBJgKcDz24TASJABIgAESAJoAwQASJABIgAEfAUAZIATwee3SYCRIAIEAEiQBJAGSACRIAIEAEi4CkCJAGeDjy7TQSIABEgAkSAJIAyQASIABEgAkTAUwRIAjwdeHabCBABIkAEiABJAGWACBABIkAEiICnCJAEeDrw7DYRIAJEgAgQAZIAygARIAJEgAgQAU8RIAnwdODZbSJABIgAESACJAGUASJABIgAESACniJAEuDpwLPbRIAIEAEiQARIAigDRIAIEAEiQAQ8RYAkwNOBZ7eJABEgAkSACJAEeCwDL774onR3d0t\/f\/8gFK677jrZvn27bNu2TQ4dOhT8e+TIkbmReuONN2T58uVBPffcc4+MGDEitM6k74UV\/spXviJbt24N\/qT9sNv+m9\/8RubPny+HDx8O3luwYIEsW7Ysd\/\/MCr7xjW\/IihUrZMOGDTJz5szgT\/jd\/fffLz09PfLjH\/84+DueXbt2yaRJk4J\/a\/tHjx4dvIcnbIzMMnZ\/tOz48eOH9Enrt7+Zdpz1m2i3jV2e8UsyCGjrnDlzBuFmlkMf8Wi7bDmPkosk3zbf0XrxO4yVibcph1pm+vTpDbnH39Nibn7bxviJJ55oyJO+Z\/fTxsGWE5VZ8ztax9NPPz2kfn0PfX\/88cdl8uTJDVlPiyXfbx8CJAHtw77tX1alsHDhQicmr2kcoLxAOKDobCNlKuBTp04FRvbee+9tGOIiBkYVqqmIo0iAkhBV7Hv37g3abZIAHSN956WXXgrImbYfBkb7qd82Db32SY1TXoPUjAQUgV+zOpqRAJVpHU991yRjwAAY24Y7bbtNo2nWr2TONPKK1zXXXNOUACdtQxgJUIIJMqI4qFyE4WDLiSmfYXWgbVH9QNk9e\/YUtmBIigPfy48ASUB+DGtbQxwJsFcrptJTI4LOw9j+9re\/DXBAnVCuME5YreFRg9bR0THEE6DKKe4989tJV3JJVltlGTOzvWrkw0jAhz\/8YXnPe94TGIYTJ04EhGTcuHHys5\/9LJQEACfTCPb29qZaUZorVDVcNk72ijGMTKTxBGzZsiVo4w033CBf\/epXG\/KgK+cwuYKXyG6HtrcZCTD7gjriPE\/NvELNcFAjrHJ\/8cUXDzLuYbKXZtzCjHacJ8AkASYReeCBB2Tjxo1BcdMDF0ck0pCZOF1SWyXpQcNJAjwY5Kguxk1cU5EdOXKk4YKdMGFC4E6HK3jRokWBon3++ecbKytzNaYrcrThrrvukrvvvruhjNToYdXW7D24OlXBhRGJsP5pGyZOnNh05RXmti9CJNTgjxo1Sk6ePNnYArC3A0AQdFWKLQKspmAs8TPME2CSgL\/5m7+RBx98MBiHpNsZugpGu\/DY2z7qWVDPg706VGzSkgBs0YBMmLKDNpuGMUyuZs+eHfTPXL0rwbTJiRo19f7Eybcauc7OzsATZq6MTXmE4VQio1tjZt2oJ8oAm1tpZpm+vr6m5K1IEnDnnXfK5z\/\/eTG9RTqO5hyHyz+uH1GeABv7IuYQ62gNAiQBrcG5kl+JignQlaupIOz4APwNsQS2YcfqK8o1GLbyCHMhmu+F1R+3J2yu7mw3rTkQWk+zPfSsA6fGc+XKlbJ+\/fpAAcPY2CQAqzRgO2vWLIFhAKZ\/\/Md\/HBjnuO2ATZs2yR133JGaBKDft956q9x+++1BzIJpkGxDEGXs05IAJTo2ibONq8qVrlhNGdVxiiIBdpuSkACMr+0NALlQEoB2m1snKg8mOcLv4MExt9XCPAFJSEDYvjzqt+NW4lbx5nbAjTfeOKR9Yf0ACW22pWBiZW9rlB0HknUeslw8AiQB8Rg5+0acklRFpu5EKETzgaIMczVGueFtRWEbAK3bfG\/p0qWBsTJXu81IgLmvHqa89RumcQkjClHKOEoY7DqiXP\/q5tfAwIceeki+853vBNVinx9kwFxZqoExgzfVGF566aUNj0waT4DuVYN8YEyjvAIIqCyKBOg3TRe9ErwwuVKCE+ZFiSIB9j54nHybsgLDj8cMODS3qvA3ewsMvwNZwWNvO2QlASpfWTwBGmiqdWh71eOWxBMQVYcG8cZ5AjRWpYhAYmcVb8U6RhJQsQFpZXPilGQzT0CYwdbVW5QLOW71ElZnWk+A7nk3ixswV35lnAxAP0wM1FjjNIIacCUBMD7Hjx8fdFIA\/68rMiUBUcGbSeIeTJkKC4oEwTCjwM3VYJkkIMzNbhtBdfmbeCb1BMStTm35jyKXtmdpzJgxjXgXe75qe6sQE9BsjuJvSeej2cc4EqDEKOrkTyv1G7+VDAGSgGQ4OflWGhIQFxNgTn67XtujoO\/qCkUNXNR7SWMCotz7YR4I7FE38xTkHXCbCKlnIYwE6Ao0zBDHkQB79asEBCu6qNMBZtS6TZrKigkI8wSABIAY6erbjAkwXdjTpk0LPB7qFYiLCTBlsdnpgK9\/\/evy5S9\/eVBsC7wScbELCMa0TxfY42CTgKResKRyl9aAKw4m6dWxDyNaYcdL0bY4EhB2Gidpn\/heexAgCWgP7pX4alRMgBoqnP01DUaz0wH2CiBp1H\/S96IiyO1VruYI0N\/bWxa6vaA5AvS9oj0CNglQpa0BlKYnQI2f7rOapCeOBJiK2exTGAHAu7ZxUqWOv9lBb\/AQRMVL2PvoiiOIle29Mbd9wiL2o8bWPMmAUxTPPvtsYKCVNCUhOXi32fl4+9QJiIaSUrscZASxFCAkUXvi6g7HVosti3mPZZqynpYExOGg5NEODLQVVRQJiFtQVELhsRGhCJAEUDAiEUjraiaURKDdCMAYgejhSFzUarbdbXTx+yDz0BdFJRZzEaOq9okkoKoj08Z2JY2ub2MT+Wki0JS84o9JgyUJZT4E1CvBjIH5cGxXaZKAdiHP7xIBIkAEiAARaDMCJAFtHgB+nggQASJABIhAuxAgCWgX8vwuESACRIAIEIE2I0ASYAzAsWPHZOzYsW0ekvZ\/njgMjAFxIA7mbKQ8UB5clAeSAGNUr732Wjl69Gj7rXCbW0BlR2XnorLLO604LzgvXJwXJAEkAUN0I5UdlZ2Lyo4kIC8CnBcuzgvnSYB9bthMTmPne6cngJPcxUmeV\/WTFHJecF4MnUWuzAunSYCdpQ053BcvXiy42Q0PbnfbvHmz6GUXJAFUdlR27io7kqG8CFA\/uKgfnCYBSAmKNKBI1QrDj5zjmtVK05fqfeUYXJIATnIXJ3le1e\/Kioc45EWA+sFF\/VBpEhCVnzxMlO1b47ANsHPnTrntttvkzjvvbJCA3bt3D7r+08xyRRLASV7VSX7gQHMF3tkp0td37h38W39n\/gyrJew9s\/yPfnRSPvShUcVYkCa12G1u9sG4NhfZ2K6ucucF9A6f6iCQNDjcFXJceRKwdu1aWb16dcNlHyYqIAvme9gGWLduncydO1fMLQB4AuJIgNa\/b9++6khli1uC2\/06Ojpa\/NXqfa7dOPzwh+fLHXeMkhMnhlcPHM9adPToMSlLHm644QaeSqqIPIGQpdH9LhwprzQJMOXCDOgzfx92b3zY7Xi4De0LX\/iCPPbYY8ElF9wOiJ51rjDcvHqlXThgRdzdLYLVP1a8PT3nepJ0dZ+371oebfnFL16RK664MtSzEOVxSOOJKLKtcR6CtH9H26ZOFYE34OGHy8kjQg9kURKQv540Y9Eu\/ZC\/l4NrqAUJ0G0BXAgyadKkVBigrAYDMjAwGXSuCHey3laHDMFwwvCDAKjxV1d03r7kKe+7PKxdK7Jmjcju3a\/IrFlX5oEytGwaw1P4xwuoEAu048ePy8yZMwuorb1VpBkLV+ZFbUhAkm2BMPExSQCuFjU9CvZ95GkEoL2iWu7XXRHuvCi1Ggckq9QV9P79A0SgCk+rcahCn+026NicOVN86+qud0gCipeJVtZYCxIAQBDp39fXV+r1oHWfjEUJDpX+AJKtwMFe\/a9eLTJvXlEjWUw9rcChmJaWVws8NLotAIJW5FN3vQMS8OSTT8qBAwekv79f7PwrRWJVdl1pxsKVeVErErBixYohMhAWE5BVUNIIQNZv1KGcK8KdF+uycQABgGGp4urfxK5sHPKOU6vK33vvySBQEySgyG2auusdkAANuEbwJE5lrVq1Ss4777zg31OnThV4YevwpBkLV+ZFLUiA7dIvS5jSCEBZbahCva4Id14sy8KhDqt\/koCh0gN5+MxnxgZxG8eOFbddY+udKt9hhi0q2xNibgdAV2\/btk0WLVokzz33XOBN6+rqIgnIq4xKLF8LEoAjf1u2bJFbb7216VHBvDiRBLTODZ53rFpRviwSgNW\/Rv7DmFT9KQuHqvfbbh9wGDZsrMBIwxNQ1LaArXcgH1V+kpIAnMD67ne\/GxAAegKqO6K1IAGAD2yzt7eXMQEtkCUq\/XLI0I4d5yL\/q7j3HyValIfB8qDxATjF8fDD+Sdk3RcfUZ4AkoD8stGKGmpBApplDmRMQPFiQqVfLAmA+3\/nzoFjZupOrUrkfxLpoTwMlQf15hQRH1B3EtBMhugJSDLD2vtOLUhAqyByeTKmwZBKvzgSYAb\/4bz5XXelGYlqvEt5CJeHoogA9U415BytSDMWrsyL2pAA+0pgHBncs2dPkP1PbwHMK0ppBCDvt6pc3hXhzotxHhzqFvzXDKs8OOQdgyqVD8Nh2LAB706e2A7qneqMcpqxcGVe1IIE6JXA5o1\/EBsQgYMHDwYXAmH\/Ke+TRgDyfqvK5V0R7rwYZ8UBBACrfsQA1NH9b+OWFYe8+FetfBgOReQPoN6pzkhjLL7\/\/aOJUmQ\/++zL8uEPX12dxmdsSS1IgH1BkPY16vcZsUjlCsr6jTqUo9LPvh2g7n\/UUKfgP3oC4mdm1LzQgE8keTLveYivceANkoCkSJX\/Hsbi2LGjiT9URgbJxB8v6MVakABd9d9\/\/\/3S09MTHDfRYEHcJYA7BYp4OBmzG78i8K9aHWnIkO3+r1La37y4psEh77eqXL4ZDhofABKQNuNj3fVO3rTB8OiOGTMm8b0wmovglltukYceemjI0XFsHT\/11FNy++23h4pTs\/ZiLO66K54EwMOHK7aXLCn\/iu2y50RtSACAsG8HLDo9Zd0nY1HCQqWfjgy55v635YjykEwePv1pkf\/yXwbyB6TJKFh3vZOHBMQZ7DCdZiYkCtsGTlJn1KmFNGPhyryoFQkoyshF1ZNGAMpuSzvrd0W482KYBAcX3f8kAeGSk0Qe1COQJqNg3fVOnrsDTC+AebmbHv0+deqUdHd3B3cS4MGlbxMmTAiyEpqeAPxt\/vz5cvjwYZkzZ45cccUVgSfgK1\/5imzdujUoO3369Eb8WBRRSDMWSeQhrw5qRflKk4Cke\/5J34sDNI0AxNVV57+7Itx5xyAOBzP5D9zAaVZ\/edvWyvJxOLSyLe38VhIcQAo17W9Sj0Dd9U7Y3QFf+MIX5NFHH5XLLrssGLJZs2bJ8OHDBw2fnQn2gQcekI997GODsgvCqMPoY8WvCeOQOdYmAfj\/KVOmBFsKetkc3nv11Vdl3Lhxgm+tW7dO5s6d29hO1vTGpjchzVgkkYd2ymvSb1eeBCi7i+tQEUmDRo+eLQcP7q7MFa5xfS7r764Id158onCoe\/KftLhQHgYQS4ODHh1MEhsyxPCgcFWfkPOQYRkD582bJ2+99ZaMHj1aHnnkEZkxY4ZccsklTUmAud2r+hwGevny5bJ3796g7IIFC4IYAJME3HzzzfLggw82DLy5yje9C2iLGVMWdj09SUBVBa9F7dK5Bzl3eWUXB2caZRdXV53\/HoaDmfwHXsb\/+B\/r3MNkbac8pCcBenQwiS5xkQTgAiEY8H\/4h3+QX\/3qV\/KJT3xChlnkptmdMFjN4zl58mTDO5DUE6Dv3XjjjY0AQXoCoud6pT0ByVRUcW9dc81H5ItffGaQYoeLF0e9XHX1hqFHpR+u9M39\/7wk8cArB6TvdJ\/09vfKpPdOkned9y7pvLhTOi\/sDD7eedHAzyo8lIf0JAAldGsgLlfEEBKASwmq\/FjnIMM8AQsXLpQf\/vCHgRv+4x\/\/eHCtcNhjxgSY+\/fqCXj66adFr5DHnj5W81j5P\/bYY5ExASh7ww03BLEB6klGOdxm+MlPfjLYMmBMwLnRIAkwJNOcjKbLV19RVo+fdcr9nlafUOkPVfpI\/qO5\/1evPtB0\/KFswp4dR3bIzr07BQSg6TNh4K8gAvhv4QcWysjzR54r8tPmxbsgnBDgkJ8H8PuYJyhvPC+\/\/LJcffW5pChxddjlzbriyuLdAL0mEyyujtzfD\/t2Z2eq7QDtcxKPQBoXdNzYVeXvP\/nJTwSnt2CQsQ3w6U9\/Wi688MIhzUsSyV9Gn8LiD\/CdNGPhip4kCYggAabgRREC6Pq5cwf0lUukwBXhzqs8gAOujv3e90Ruu+1c9r+xY5vv2Z6xMohgxT\/1yanByl8WxLeq56c9curNU\/LkS08GZYJy5hNTR9g59RG\/E\/m3YSJb\/9\/47+\/+E5Er\/j+RX14s8t7fivzqYpHLfyty8iKRUadFph5pXse3\/1jkPf8q8tsRIr95t8hlr4n8+qKBuj7xj\/Hf75ks8rvfE5nwS5GfXiHy+78Q+ZcrRf7gFZGfXCnyn3qb17F\/wtA2J207asb30U+UueK3Ih8324yJ\/sADIh\/4QOJJb3oEwjxIaQxPPHr1eyNtnoC8PYzLE3D0aHyeALTBFT1JEpCABNiEAOwet8Lhp\/lAP2DrQBcydd1CcEW48yoLpAX9\/vevDjwAZja4qTEXvu+3Llwf9rfDghX96omrZeeCf79OMOZplIeAwQVx4ID0XTpQCD\/XRhjhxjvrm3xgc9zXRWRxzDtxdTQrH1cWn67g9ztPiezfLIKfjQcTHonKPv7xWEKgHgGUtU8N+E4CEkhky15JMxau6MnakACwRd0bMiWiiFMBWl8aAdAyminu+PEBUmATAyUE6i2A50CfqnoQXBHurJpDA5g6Ov63nDgxPAgSTZsFDt+G+7\/7QHdAAPZ\/cn+yfX7D8Ee2X91OKkDXXSdywQUi73jHOQaasvMgEDBwQ36+1icvXztOrj76M5EpXSK9B6J\/jukUOd4n0uRn6Deivp3296\/1BTEVfa\/1FdJW7cuBV3qlu29NgOi8H4j07AgBN0kEoIhobJxJBLLonZTDy9cTIpBmLFzRk7UgAcgDsHjxYlm5cqU8\/vjjjfOgCCTp7OyUmTNnJhzi5q+lEYBmNSkxwDtIGvLMMwNbtAm2YwdtK5i6Xr83erTIH\/6hyJtvDt3yVcJhbwebpCMJUPYecJIyad+J2LJOW00p76u7HyTgS18a3joCgAPmtpCY7BGupTbsO7mi7PIIC7Zk7vv7++TrP\/t6UE3X61dJz7qfD\/YM6ARsEjVqni5RIlCU3snTP5YdQCDNWLgyL2pDAvRMJ6JF+\/r6gvsCmiUJMj0HprfAPDdqpx1OIwBZJ43qePvnr34l8pOfDLYB9jtZv8lyaREY2PPftOlXmXKDp\/IAYOWPaPAw45\/3CELabke874qyywKHeoUQ56E4rH1+rax5fsAzEGwTPN8lnd8K2RtsMn6aWRDbh488cq0k3YfO0geWSY5AGhvgyryoBQnQq4QnT54s119\/vSxdulQ2btwoP\/7xj2XPnj2yfft2GTnyXPQ0Ik7Xr18vmzdvDn4PjwHSTqLcnXfeGXgU8JjvpGWBycWqvDfDVtNpvmbbHa0PF2PcdNOoYGsDi8+wn00C0IcEpqdpU9p30eaoNurv07Z16tQBEgDFPFbTvyVsmBoIbAEcm30suhQaDuNv7h+hoZ\/6lAguPmnDij+qsa4ou4RDOOi1MBKAF+AZ2N+\/Xz7T+5kGGejp65Kuv01OBj760YH4gLFj600C8twdkGVMyixDElAmujnrNlf95tlR5JLGuc9mj6a1RPKI++67LyANmolq9uzZjfJpBCBndypd3Gelj4FRxZ+WBNz\/T\/fLoucWybwJ86Snqyd8jKOOmiAA8CMfqZTx1w74LA9RJMAcXHh+QP70FMdjL02T\/+tvnh4aRBjiGUDq6bvvJgmoikJMYwNcmRe18ATkFRCNHcB1lbt37w4ukcCDdJTwLmhMQRoByNumKpd3RbizYpyFBKgHoCkBQIPsff+4TDJZO1FgOZ\/lIQkJUKhBAiAHIAV4sFWAIMIu8zRHSABh3fVOnguEChTTQqpKMxauzItakIA8FwTpZRKIITAvuogiASpJ+\/btK0So6ljJiRMnpKOjo45NL6TNUAR4nnnmmUQ4PPLyI7LmX9bI7I7Zsu7960LbcOG3viWj7rij8be3Ojrk1c9\/Xl6D+7\/ij8\/yoLIAr1BSHE68cUJ+8OoPZNk\/LWuQgb99TGTaP58baIz\/yU2b5M0\/\/dMgu50ZEzAkL0SF5CMsk2WzC4SQKOjyyy8PsgbaaYMr1K1GUzDeaXR\/2u3CKva5FiQAwGU5CWCXgbDid9wOaC6KrjDcrBMujScg1gMQ5v6\/4QaRhx6qpOs\/DDOf5SGNJyAMO3gFHvznB+Xvf\/X3wZ\/XPCky96CRb6CzU64dNmwQCUBeiao+YbEuYWmDccnP7373O3n3u98t3\/zmN4NUv+985zur2q1BJCBpkKYr86IWJACegKjbBKPyBMDY69WSOsLmUUP8ru6BgWXNKFeEOys+SUmAngKI3AIwz4OhMZpNKkvSgaydKaCcz\/KQlwQo\/Fjd7zyyc9CpAt0quHbs2EEkAMSyyg+SXplPGAnABUJvv\/22IGYLRCDsKuEq9pHbAVUclQxtMo8BanFcPoFYANxPjYsl8NhBhWkEIEOzalPEZ6VvDlIzHJD\/f+reqUECoNBTAIj4Mi+CCbmCtS4C4bM8FEUCTDIA2UESqYAXnhIZtm0wCaiLXGg7m10lfNVVVwm2ZKdNmzboBFdV+5jGBrgyL5z1BGQRsjQCkKX+upRxRbjz4h2FQ1MCEOb+RyAq0svW9PFZHoomAaYI6KmCYasGbwfUVEwGNfu1116TRx99NIipef311+Wmm26S4cOHV75raWyAK\/OiFiQgSnLCXP55pCyNAOT5TtXLuiLceXEOwwFu3bG7x4Z7ABxx\/9u4UR4GECkLB+qdvDO1uPJpxqIseSiuN8lqqjUJyHNqIAyeNAKQDN56vuWKcOdF38ZBbwNEvcgD0HWlcWVwmPsfmWAqlPQnKx6Uh8EkYAfGOuaZFxP3YdZx9913M2NgHKAt+jtswF1Tpsi8884T+bd\/C\/2qjv6HTp+WD37zmy1qWXmfqTUJMKP9zYyBWeEiCSh3xZN1XNpVzjR+IADdvd1BQphBBCDM\/b9uncjZrJTtanuR3yUJGDwvkhx1s6+TtsfDrAPHzJJGpBc5rqxrKAKwAZD3M03A0bMbSAc2z7o2vI6Y1oIENDsdkCRjYNKBIQkgCTBlxTR+ehQQBACnAYLHUfe\/PV9IAgbPi24z4DNCufQgO2CTx6yjt7eXJCCpki75PdiAKceOSbPRGwjpFMGFsF0kASWPSIurJwkgCQgjARoI2DW6K7gSOHj+238TmTbt3Os1yPyXdTqRBJQ7L3zQO3r\/y\/PPPy8gSOPHjx8kjjhBgIyucSngs8pw0nIYizPrmvkBztX01ltvycv\/z8tJq67se7XwBLQKPR8mYxIsfVf6Zp6AYZcNk6lPTg1gAwHAkS7BFXDm7Utf+5rIf\/7PSaCt5Tu+y4MOWlk4+KB34M3dtm2bIH8A7m2xnyqRgCkPT5GL3nmRnP7d6abz9fRrp+Wbn2RMQEuUWlQAIAMDy4G\/LGVXTmuLr9UkAXcfvzvIBb\/\/D3uk66cy9Ow\/7oKtWfKftIj5Lg8kAc0lBrFZDz74oDz77LNB7pXjx4\/LihUrgkK4rn3GjBnBPS179+4VzddiEwFN775169agXJHbvGnkPQ0hc2VeVNoT0CwWQAc2SqjSDLy+m0YAstRflzKuCHdWvJUEbPrvm+SOf7pD5r3SGVwEM2j1j3uKse\/rQPR\/HE6+ywNJQDwJQFwD7mfBNe47d+6UVatWBSt+PcY9YcKEWE9AX19fUAdIBer7zGc+Iw8\/\/LBcccUVLbt\/II0NcGVeVJoEqOgVveKPEumPXHNNcGlM8MDda15EH6cpq\/D3gtr88ssvy9VXX11uj+y26v+X+9VEtQ\/DTX8i0vFIhww\/cUKOrTSK1TT1b6KOR7zkirLLgwHKloWDbXjiLqVBO6KeuLLaj2ZYNKujs7NT9uPoq\/GYGQPDsrXCG4CMgUm3A0AknnrqqSDVMB6QCdw\/gP9\/17velXcYm5YnCSgV3vSVJ\/EERN0dkP5rImOHxV\/cETX9BsxG8yd66g6Uy1tHXHlEvBon24c0Nq58J\/bFI7oYV1aLNcMgro48ZQPlFzM++v0+fW+ryP7NZ6+ChfFfvlxkwYK4YXbu72UZv7oBVRYOtuGJO4LY7PhhXFlgnub4oj1GIAE2CTFJgO0J0PJpYgKUBNx+++1BW7\/3ve8JbiNsRdAgSUDdZmXB7U00gSK+GU8fpOnZU1Sbt4648jDgzUhAXHmQgChDGldWYUty\/jZqWPOUDZRfjLwM6sN6ka5TIvu\/1Sly880i8+d74foPg6gs41fw9C29urJwsA0P3OLNHhjiqCeuLMo1K4+\/x9WNUrT7AAAgAElEQVRhlzdJAMpjf19jAvD\/2N9Psh2gpwOUBHz2s5+VJ554Ikg9\/Cd\/8ictuYqYJKD0aVTtD\/zF5ZfLpo0bM01ATNu+f+yVzg9OCX7qE9wN3jVF5ECvdJ79qf\/\/9pQPy+\/1Piv\/\/H\/8q1z+y3+VvzrvH2XCq++SI5f8r6HbEeouvzS6eYHqiHKz9\/Wlm\/zmVohRZ6AAQr7RF\/H7QVsqMW0IlE+TekJV39kycWXR5qi26zfNPvS9s092f2i3zLp+wCXp81OW8asbpmXhkMbw1A2zPO1FXAAyK\/7RH\/1RsD2JAMOyryNOMxZlyUMezLKUrUVMQFjHwBaRcGPUqFGyffv2Qm6oGsLIYcDPPn2v9Ulvf6\/85n\/9Rv7Hb\/5H8FsY+MDIl\/zgprpWPjj\/2uyyj1a3p5V9128hJfBfjvxLSbLH2o72tfKbrii7vJiVhUMaw5O3DyzfHIE0Y1GWPLR6jGpFAtTw9\/f3Bzgh4GTmzJmFYXbN\/3mN3PC1GwQG\/0D\/gdB6TQOIf3deOGCg\/8OV\/0HGXTxuSBn9O+rUf9svVc2ouiLceQWDOAwgSBzKxSGN4ckr0yxPEmAjUHkSYBp+PQ64ZcsWmTJlSuGBIsP+dlhwOxwerATn\/j4SQ0rDeFfNWJc1oan0y1X6ZY1bWfVSHsqVB5KAsiQ3fb1pxsKVeVFpEhB1NLDoK4RVVEZ3jZb+AwNeBp8fV4Q77xgSh3KNX97xaXX5suQhjeFpdZ99+16asShLHlqNeaVJAMCwjwliCwBBYGV4AtIIQKsHqpXfc0W482JGHEgCTBkqSx6gd\/hUB4GkNzqWJQ+tRqLyJMAGxD5+UmRcAEkAlX4rlH6rJ3ne77mi7IhDXgSoH1zUD7UjAeYggBDs2bOntNMBxUyZ+tVCpU9l56KyyzsTOS84L1ycF7UmAXkntV2engBOchcned55QuPHecF5MXQWuTIvSAKMsSUJoLKjsnNX2ZEM5UWA+sFF\/eAdCTAvuLDjCUgCOMldnOR5Vb8rKx7ikBcB6gcX9YNXJAAnDRYvXiwrVw5cC7d+\/XrZvHlzI9sgScCAiBMH4mAqO8oD5YHyMJRAuTIvvCIB8AIgxwDSDON6yuXLl8vs2bMbSYdcGdS8fJ84UOlT6bur9Kkf8iLgln7wjgTs3r1b7rnnnmAUQQImT57cSD3M87rFTA7WQgSIABHwAYGkOQWqjAVJgEECqjxQbBsRIAJEgAgQgaIR8I4ENNsOKBpc1kcEiAARIAJEoMoIeEUC4gIDqzxQbBsRIAJEgAgQgaIR8IoEADzziOCuXbsKv4mw6AFifUSACBABIkAEykLAOxIQBmSz3AFlAd+uet94440gIHLv3r1BExYsWCDLli0L\/h2Fg+v4IP30wYMHg4BRnBrxEYewK7t9xMK8m8TXuQEMcElbVr3gir4wcXBZb3pPAnzbIjAFW29onDVrlkybNi00hwLIQbPcCu0iM0V9V43fxIkTAxKAyR7WX5dxMOfA+PHjg2O0uKVzwoQJXmEBWVi6dKls3LhRLr30Upk\/f35gCH3CAWO\/devWxuIgSj9GzQdX5omNg8t603sSEJc7oChjU9V6IOydnZ0yZsyY0BwKaLerwZQw+OvWrZNx48bJCy+8EJCAw4cPe4cD5kBvb29j5aeyGjU3XJUJ0+CBBCgZPHXqlBcyAUMHPQBZwAMClFYGXJCNMBxs\/e2S3iQJOHRImuUOqKrxLqJdptKDogvDAUrBVXww2fGYfQQJ8A0HKHr0+aWXXgpI0PTp0xuEyDcs1DsGuUBSsZEjRwaG0CccYOBMEpCm7y7pCxMHU9+6pjdJAjwlAbrHpRkTfVN0cP3u3LlTVq1aFRg+VXQ+kgDzSm7NpIkkWlEK3SVFbyp3c9WL3+t2AP6dxhDOnDmzCI7etjpIAgagDyMBLupNkoCYVMJtm4klftjeA8an0rr9Jk2aVGILy6\/aDADTr2EFfOONN8p99903JLW0C27OKFRNAggSoMoPcQFhW0GuYmEHxPmKg00C0siAS7JhkwBX9ab3JMC3wED0d+3atbJ69erGxUmYuGkDgOAmdeUxjaDvgYEdHR2NdNq+BYuGEWF4RHzDwTR+afUCdIIrgcQ2Dq7qTe9JgK6C58yZE9g013MHaNSracD1SuWoHAqu51awV8I+4mD2OeponDk3XJWJJEcEXcfBXgGnnQ+uyIaJg8t6kyTAleUs+0EEiAARIAJEICUCJAEpAePrRIAIEAEiQARcQYAkwJWRZD+IABEgAkSACKREgCQgJWB8nQhUAQE7janZJpxyuPzyy+V973ufVOG4mu6za\/4BnECwH91z1fiUKmDMNhABHxAgCfBhlNlHpxEw090i7W\/VHvvoXVT7NHlTFYhL1TBke4hAWQiQBJSFLOslAi1CIIwEaFpTGFTba6DR7Xoq4qKLLgpOxVx33XVBqtglS5ZIf3+\/6KocdV1wwQWyb9++ILGSeXrAvHgI5TXLntn1sPP3yE+Px6yLJKBFAsPPEAEDAZIAigMRqDkCcSTATv4CI9\/T0yNIFY2jsUoAcLskUgfDkB85cqSRKGjbtm3BrZMoY1+sg6x6uIAKZAPfAXnQ2xgVVpME2JkpcXfD3LlzBR4MkoCaCyKbX0sESAJqOWxsNBE4h0AzEoBEN5r+FlkezbSnqEGzwSH5k+k9QJ3r16+XzZs3C0gAHr1aVo06sivqOygftS0RRQLs2ACSAEo1EWg9AiQBrcecXyQChSKQhATAjW8+cPWbdwBoumDcKIlVvU0C9PeoQ426mVYYJCAsrar5vkkiVqxYETTHDAQkCShULFgZEUiEAElAIpj4EhGoLgJxJEDTuNpBg2F3BkSRANMToNsLWTwBJor2ZSwkAdWVMbbMXQRIAtwdW\/bMEwTSxARoIN+9994boKO348V5AkAYECuAR7cXJkyYEPw7TUyAuTUAEsCYAE+ElN2sLAIkAZUdGjaMCCRDII4E2KcDzLsikpKA06dPy4EDBwadGkDr0p4OiGqLbhvgJ48IJht3vkUEikCAJKAIFFkHEXAYATNgMEs3mScgC2osQwRagwBJQGtw5leIQG0RKIIEIBCQGQNrKwJsuMMIkAQ4PLjsGhEgAkSACBCBZgiQBFA+iAARIAJEgAh4igBJgKcDz24TASJABIgAESAJoAwQASJABIgAEfAUAZIATwee3SYCRIAIEAEiQBJAGSACRIAIEAEi4CkCJAGeDjy7XTwCyJ2PDHp2nv7Ro0cHN\/DZaXvtFiDxzlNPPSW33357rsYhu5\/eDohLg4p8NNkP6rRvCyzyO0nqQlu2bNkit956q+DugjRPmRilaQffJQLtRoAkoN0jwO87g4CSABhevSwnaefylLW\/UaaBqxIJQP4CTWdMEpBU0vgeERiMAEkAJYIIFIRAnCFX4\/zFL34xyMPf398f3KI3Y8YMWb58uezduzdoCZLq4HKe7u5uWbBggWzdujX4iRWv6WnA78Ju5tMyu3btEhASZOzTW\/tMr4TtuTBv9DMhMd\/DzYF4Lr744sATgMdse1Qddrpgs+1m6mHUp+3W3+Nugn379gUeFi1n9+nuu++Wu+66S7q6uoL0xhMnTgzad+LEiQBHYG3WXSZRKkicWA0RaAkCJAEtgZkf8QGBpCRADZm5kgU+MPDqRVAjZRpLM3OfGkEYzEsvvTQwdCAPIAV4D8RB\/7Z+\/XrZvHlzMAT4xjXXXBMYSLjSdSX99NNPy\/333x+6bYH6QFCwpYEH31Ija9Zx5MiRyG0I1AFDjO\/CmOt2hbZ94cKFwZ0B6Je2w\/7WE088MaiNJn6nTp0a1C5ciKQkIqxuvF\/WlokPss4+uoMASYA7Y8metBmBqJgATZerxk9Xy6bBgzEMIwG6Ktau2d\/A348fPz7IOIatcs2Vs7YHBhxk4brrrgs8E2EudZvYmNsBS5cuDeIXlLjou3qroN1m+\/f4u4kBYibM76k3RMmN3a8wEqAGP65ueDRIAto8Yfj5SiBAElCJYWAjXEAgqSfAdNPrqjeOBJjudJTHo0asGQnQ92BIV69eLWvXrg3KhrnKw8hAMxJwyy23yG233dZwtesYmt4L\/M5ekZtjbe\/rh5EANexpSUCzukkCXJhx7EMRCJAEFIEi6yACIoNWsWGBgbYRS+MJsA2pWRdIAPb8lVyYf+vt7W24\/OEix\/69kgD8Px6TYNgGXI2ybiHgfa3D9gRECUEzcpTEE5CVBDSrmySAU5YIDCBAEkBJIAIFIVCmJ8Cse9GiRY1gvLiYANNLoPvg8AogiA7BdLZXwHSnKyxFxQRo\/IHu3+Nb119\/fbCX3ywmICsJYExAQYLNapxGgCTA6eFl51qJQFRMANpgu\/A1at8MxtOAPrjl4Wb\/7Gc\/21jdow5zXx\/vnDx5smE8dfWP98zTARMmTGicKMDJgFGjRgWQIAYAj3naIOqqX9NTUPbpAPP0QjPvB\/Az+7xp0yb52te+1sBDx908eWDWzdMBrZwZ\/FaVESAJqPLosG1EgAgQASJABEpEgCSgRHBZNREgAkSACBCBKiNAElDl0WHbiAARIAJEgAiUiABJQIngsmoiQASIABEgAlVGgCSgyqPDthEBIkAEiAARKBEBkgAD3GPHjsnYsWNLhLseVROHgXEiDsTBnLGUB8qDi\/JAEmCM6rXXXitHjx6th6UusZVUdlR2Liq7vFOG84LzwsV5UTkS0OystT2Jm+U8bzbh9dzz7Nmzg7zn+pAEcJK7OMlp\/PIiwHnBeTFUhlwhhZUkAchvjjznze4IB1lI8p49dCbJsC9nIQmgsqOyc1fZ5aUCrih94pAXAbf0ZOVIQDHDE14LPAC4Oe2mm26SJUuWBNeu0hNApR8lc1T6bim7vLqF8kB5cHGR4BUJ0AFUb0AYCfjlL38p733ve2Xfvn2CW9LwPPTQQ8FPX\/5\/zZo10tHR4U1\/7fHdsWaNXP2Rj8j0swKz9+zPPP8\/TM6IpKzx\/PM\/KZMnv9F2+TPlIWou6Pxw+e8333yzPPbYY6G6wIf+Q\/9hfE+cOCGQCVM3+tR\/s68uBJKTBDAmYMgCyfsVz9SpIn198trEiXLhRRflXUCK7NghBzrnyc6unlR17dgh\/65sRVavTlWs8Je9l4eziDbDAVuJfNxAIGlwuCvzonIkIElgYNaAwCSegKQC4Ia4h\/fCFeHOPEYgATgi+PDDxRwZ7e4OSEVgzbu6Ejdr2DCRefNEetJxh8T1J33Re3lISAKoO5JKVHXfSxMX5sq8qBwJgHho9P7kyZNl5syZhUtMs+0ATmSej5eiScCBAxLUCWsOq57wgScA\/GH\/\/lTcIWHtyV9zRdkl73F6cpzGeORtB8uXh0CacXRlXlSSBGCIYagXL14sK1eulPHjxxc66iQBzeF0RbgzCw1JwCDovJcHegJipxKuuR4zZowcP348+GkGXMcWjngBenrbtm2yaNEiGTFiRNZqgnK4phvXYOsV1Ghn2AKTJCAXzPUvnEYA6t\/b6B54r\/SLJgHYCkCd2ApI4dunJ6BasywuJsBXL+KLL74oTz31lNx+++2iZKAqJEC9ynv37hXzSPh3v\/vdYHFpLzDT2ABX9GRlPQHtmP5pBKAd7WvVN10R7sx4FU0C0BD49WHVz+CUQPLnbIyiHDuWvEzRb3ovDx54Ag4dOhSs4rE6TrsCNw0\/\/t3X1ydbt24NUFPDi9+vWLEi+N2CBQuC49n45pNPPikHsF0mIvfee2+wYj98+HBQbsKECfLVr35Vfv7zn0tvb69Mnz5d7rnnnuB0Qnd3t\/T39w\/6Rpjco653v\/vd8vjjjzc8AXjPJC5muTQ2wJV5URsSACHCU0aMgApBGgEoWtFWqT5XhDszpiQBg6DzXh48JgHnnXee7Ny5U6ZOnRq6Lau5V2699dYguZuSABh5GFqUhc6GMd+8eXPwjrrmAevu3bsDww5jjd9v375djhw5Ehh91GluCWs5bA2AIOAniATexfeaPeZ2AN6LIjppbIAr84IkwJCcNAKQ2cDUoKArwp0Z6jJIQEbfPj0BmUex8IJptgOqfA9ZZ+dAsKn5RHkCnnvuueAira6ursQkQGMClCD82Z\/9mfzgBz8Itgvw6IJOYwhAEkxjrqv0OXPmDIoJ0DbOmDFDli9fLnDxm56FtCQgLONsGhvgip4kCSAJGDJ3XBHuzFagDBKAxmQ485eRO2TuelhB7+UhgyfgrAgVOg5FVhZGAnRFrSv4VatWBavtqP1ztCfME4Dfw7An8QToFkQUCTANta7mf\/SjH8nHPvaxgJTQE5BfKipLAjDguq9kdlP3k\/J3fWgNaVhgGd+vSp3eK\/0KkQDlDu1MGuS9PGQgAVWZy0nbYeZnQR6WD3zgA5KEBOjKXlf\/WOWDNDz77LMyevRo6enpCYx1VExAEhLw\/PPPB\/v\/qvvNuhAngO\/EnSCwtwMYE3BOMipHAjSaE03EXpF5NKTZ35IKe7P3SAIG0PFe6ZdFAjRpEE4IwCeb8IEDgSQgIVglvpZmO6DEZrS86maeADQmyqC2vKEpPvjAAw80vAlmsTQ2wBU9WTkSEHc7YNzfU8jBkFfTCECe71S9rCvCnRnnskiAJg1Kmf0H+8spTxdm7npYQe\/lwQNPQF6BKfJoYN62xJU34x\/sd9PYAFfmReVIAAZFtwI2bNgw6DSAuoHK2hJIIwBxglbnv7si3JnHoGIkoN1xAd7LA0lA5qlUt4JpbIAr86KSJEBdTOZZUPzO3GMqQ7jSCEAZ369Kna4Id2Y8yyIBaBC2BOARSHHwnyQg80gWWtDX7YBCQax4ZWlsgCt6srIkoB2ykkYA2tG+Vn3TFeHOjFfZJCBl0iBwBnAHXDsQdaMg3kFiwhRXEySGx3t5oCcgsazU\/cU0NsCVeVE5EpB0zz\/pe2mEMo0ApKm3bu+6ItyZcS+TBGRc1qNJMPQw8mZM4ZgxA71cu3bgJ2IH5s7N3PPQgr\/4xStyxRVXFltpjWrTix\/pCYgetKreHYD9f+QbwGNmKuTdAefGspIkYP78+UEGqbgn75XCdv0kAQOIkAQUfJWwLWgZ8gWgCiUCUfMChw7gMeBTLAJ6+SNJQDiuVb07wMxhgFNm69atk7lz5wZHFnl3QIVJQLHTN11tJAEkAQECZXoCUD9IAJaXdsaWFOIK1z8evZtogLwN\/O5sKvYUtTV\/9de\/PimXXTaqsPrqVJF6WICtyyTA1bsDVNZACEwS0CxPwNy5R4eIqHrczD+MG\/eyfPjDV9dJnEPbWjlPQDsRJQkgCWgJCch4VDBsboAEaIpaGKoU6QcSTzWfPUNm2mbfSMAtt9wijz76qFxyySVy+eWXy8c\/\/nEZBgJrPHW4O0C3BMxTZc3uDjh2bCgJCJssmzadlCVL6k+OSQKM0SUJIAmoGwlAe3W1GhU0mNjaR7xIEjDgZUlFAnRQ8oJfRnksa60I0jBPAC7w+d3vfhfcwvfNb35Tbr75ZnnnO98ZSwKqeHcAGm33MerugO9\/PxkJOHPmmIyt8iURCWWHJIAkYIio+Kz0W0IC8JGMVwtHeQPK8ADot3yWh8yeAGvFnFAft+Y1CIt1RNXO3Y\/b\/5A2+O233w6u9QURmDVrlgwfPjyWBOCFKtwdYAePmzfR8hbBc8NIEkASQBJgI1B2TEDBJKBsy0ESkMEToEEbZQ9O1vot1hh2d8DChQvlrbfekquuuirI\/T9t2rTgKmD7MbMFVu3uAPN0AO4Z0FT0vDugwiTAFMYo+S76VIB+h9sBA0j4rPRb5gnAhzKeEsiq97OW81keMnsCsoJdoXKvvfZaEBPQ0dEhr7\/+utx0001DPAFoLu8OqNCgZWgKPQH0BNAT0A5PAL6J\/cSwy90zTOQyi5AEZPAElDkgFaybdwdUcFASNokk4CxQ8N69\/\/2H5I03JiWErtjXzCNfqNn0Jh4\/PvAt28PYzOOY1BuZ9L1ie1vt2vbL1OAE37GHHy438Cdj4qBWo0cSQBLQaplr1\/fSeINdmReVJgF6dfDevXsbMmHu69iCYm4lRL2nlxOhrHkXgR61KvrKVtO449+9vYMNOn6XxRBHBYIlCRCLe+f06dfkoosubNc8bPt35+5oEQnAUUFEkYNxlBXaXwCarii7LFD4vB2QBa+6lyEJqNAIKgGAoV62bFmjZTDi\/f39jQAPs8n4W2dnp8yYMUOWL18us2fPlkmTzq3stU7797rKhnc2KwmAIdckLTD05v+HwQpDrMbY\/KlJKey\/oY44413U8Pms9AMMWxEYqIOV4UKhosY5aT0+ywNJQFIpceM9koAKjWPU3QDNfo90wyAMMPzYo+rr6xtEIFB28eLFsnLlyiB1pP2MGHEoKIs0oWEGV1ftunJvZuxRXnOOT5ky2OC3yphnHU6flX7LSQA+WPEAQZ\/lgSQgqxapZzmSgIqNG1b22Aro6ekJjDaiUHG9MFz9pncAzbYNPEjAwYMHB3kMzOMiKGNmkML\/jx49W06e3BSg8KEPXdZA49Ch80OR6eh4S\/DfVVcN\/Ozo+N\/Bzz\/90zcrhmS65pw4cSKICPb1ufLshSP\/sHFjS3AYdccdMvznP5eTGzfKWxXE3Wd5mDPnSjlxYrg888zL0gyHG264QY4eTZZkpsrzKur8vLYZf3\/hhRfkox\/9aNNumHkHqtxfu20gAfv27UvcZCYLSgxV9hdhzFesWNGoYMOGDUEiCvtJQgLMMro1MHny5EZ9EICHHz4abNPqat90y2NFj0dd+VVf0WdF3eeVX1s8AXpXMOICyrgLOKsgnC3nszz45gmIIwFm1r0osVKdbS+ycophS4rTE9ASmMv5iAYFNtsOsL8MTwMe9SqkEYByelGNWn1W+m0hAfhohWMDfJYHH0iAGYANLyvisBYtWiRbtmyRrVu3BlMCv1+6dKncfvvtwQ2vyCLY29s76O9IxIO64ClAhkH83fbYVkPDRbcijQ1wZV5U9nRA1N5\/MyGKCww0XVQ2aUC9aQSg6sKcp32uCHdmDFoZGGg2sqKxAT7Lgw8kwEynCx25e\/fuwOC\/+eabMm7cuMCw6w18p06dkuPHjwfZA1999dUhf9dYqzpvByTd1nFlXlSWBEA3qlEPc\/+HKXjziKC6omwyYR4RtN1VJAEDqLoi3LUjAcgbgL0oRKZqVGnmThRX0Gd5yEICsGrGo0eb7f\/XlXU7\/25KB3TilClTgqBoczsAK\/45Z+Nj9Di1kgDoZDPGyjxujbpJAoqbf2XXVFkS0Cx9MNMGlysWPiv9tm0H4MMIRMG2AJ79+8sd5BS1+ywPWUhACmgr8arpCUDwNS4PgpGHOx\/u\/zBPwPXXXy9PPfXUkL\/TE1CJIU3ViMqSgFS9KOhlegLoCWgrCVAigIQV8ARUhAiQBLidMdCMCcAC6wMf+IDcdtttjf1\/rPK7urrkk5\/8ZDA94B248847A08HvAXm3zUvCz0BBRmlFlRTWRKQNk9AEViRBJAEtJ0EoAE4LYAlaNbMVUVMBqMOkgC3SUDB4lLr6tLYAFfmReVIQJJbBJulDs4jgWkEIM93ql7WFeHOjHO7AgPNBuu9AhUgAj7Lgw\/bAZnniYMF09gAV+ZF5UiAylWW0wF5ZTKNAOT9VpXLuyLcmTGuAgkwPQLYGohKY5m5k8kL+iwPJAHJ5cSFN9PYAFfmRWVJAATKzvCnQsbAwHKnmyvCnRmlqpAAdECDBbFFgERCc+e2\/OSAz\/JAEpB5FtWyIElAhYYt7Bx\/2c1LIwBlt6Wd9fus9APcq0QCVBBwdBBbA\/qAECBlJW6cslNYpr2WMub9V37xC7nyiivaKZJt+zZg39HXJceONT86S93RtiEq9MNpxtEVPVlZTwC3AwqV7VSVuSLcqTptvlxFEqDtg0dg586B4MG0xj4zIH4XXNvZI6uPzWuaPyON8agymnFpg+PuDjBjuuzcAVXut7YtzTi6oicrSwIwKGE3AZYpSGkEoMx2tLtuV4Q7M45VJgFmp\/RaS1xnqQ9+l\/RSC723Ogaok7\/+tYy67NyFWplxrWNBuALwHDtGEnB2ixYZA6MSuJmJhzTnwKpVq2TEiBG1GP00NsAVPVlZEsBkQe2bM64Id2YE60ICMncwXUGv5cEICmiGQxrjkQ798t8u8u4A09jHeRXK71n6L6QZR1fmRWVJQPrhy18ijQDk\/1p1a3BFuDMjTBIwCDqv5SEDCahb2uAy7g6wb3XNPBdbXDCNDXBlXpAEGEKWRgBaLJst\/Zwrwp0ZNJIAkgBFIAMJyCx3bSpY9N0B2AZYv369bN68WUaOHNmmXmX7bBob4IqerDQJMN1UOqRlJQpC\/WkEIJuI1aOUK8KdGW2SAJIAj0hAkXcH4IIh3EKIa4XrEgdgCnsaG+CKnqwsCVACgAhT805qsNb+\/v5ShCyNAGQ2MDUo6IpwZ4aaJIAkwCMSUNTdAbhe+PHHH2\/cnggIy1y0ZZ7fTQqmsQGu6MnKkgDeHVCGiCer0xXhTtbbkLdIAkgCPCIBmeeJgwVJAio2qFj146aqnp4ewRWV2Gvq7u4O2KXpHSiq2WkEoKhvVrEekoB\/v7wHp8IefljG4kY\/zx+v5cGDmADPxXtQ99PYAFfmRWU9AToy2K9asWJFY6A2bNgQeUY1rzCnEYC836pyeVeEOzPG9ATQE0BPQObpU+eCgQ2YMiVRF1752MfkylmzEr1b5ZcqTwJaCR5JwADaJAH0BJjzzmt5SOEJaKWu4rfKQ+AockQneE5u2iSjlixJ8Ga1X6ksCcBWAB51+2vyoFmzZpXmCfjINdfIM888c27E7MxrzdK0hv0t6v3jx8OlImsa2KzlImTzzTfflPPPP7\/akltm685m3eN2AElhcI8E5CEmY2BecRw2bOB+KFwWGTz45lmPVPBL3CRZkcdrUmiMgSs4VJIE2ARAcdco1smTJxdPBPr65ND73y+T3nyzIlOtSTOSpoXN2JO33npLhg8fnrG0IyXs9TgAACAASURBVMVKVvp1QskVZZcJ83aRAP3u\/v3J00Bn6mD6Ql7LA0lAeoFJWyLu4qAyE1GceMc7pOPOO9M2eeAmN\/sJM9RRxrtko562Q5zkXAGbMuO1PLSDBARXF+4IvA9VfLyWB5KA8kUyjgTE\/T2uhYcOHZI5c+YEr9lBhowJoPGj8Rs6g7xW+i0gAbgQEp\/Bor+rb4dId\/fZ\/6nOFgDnhbvzorLbAZ2dnaEuf5wWOHjwYKZkQWY+awypndqSJIAkgMrOXWUXt0AI\/XsLSABsPhb+Z86IyJDggEytLrWQ16SQnoBSZatRueYDwC\/sHAHm79K2Bl4AxBts3749SGm5fPlymT17tkyaNCmoiiSAJIAkgCRgEAKtJAFr\/v3a4jVrzrKBtNqtde+TBLilJyvpCVBxLjpHAEiA5rXGN0ACzCBDkgC3hDuvWqSyozwEfnr467u6pKxTMzgIgP+6Ov\/9HzgisHp1XtEttTznhVvzotIkoGhJTkICfvnLX8p73\/te2bdvn9xyyy1BEx566KHgpy\/\/j0nuc\/91vD\/4wQ8GXiLfxt\/urykPUVjo\/HDt7zvWrJErz8YQ\/d8\/\/7n83VVXBbrgz3\/96+Dnf73ssoaawu\/0\/7P8ffrbb8tfP\/dcqK6pEr433HBDI5OmjneV2hemq8toH+bF0aNHizZTLa\/POxLA7YB4GaNHZAAj4kAczNlCeaA8uCgPXpGAJIGB8SaSbxABIkAEiAAREHoC6igE5hHBXbt2NYIC69gXtpkIEAEiQASIQB4EvPIE5AGKZYkAESACRIAIuIYASYBrI8r+EAEiQASIABFIiABJQEKg+BoRIAJEgAgQAdcQIAkQkWaphF0bcL2Eae\/evUHXFixY0LipMQoH1\/Gxs1D6iIMm6Orv75fp06c3MnL6hoWZm8TXuQEM+vr6MusFV\/SFiYPLetN7EhB3YsA1EmAKtnk987Rp02Tx4sWycuXKoMuaUhn\/Dvv9yJEjnYBGjd\/EiRMDw4fJ7hsO5hwYP358kFVzypQpMmHCBK+wgCwsXbpUNm7cKJdeeqnMnz8\/MIQ+4YCx37p1a2NxEKUfo\/SCK\/rCxsFlvek9CYhLJeyEpWvSCQg77mkYM2ZMaEplFG2WW6HO+MDgr1u3TsaNGycvvPBCQAIOHz7sHQ6YA729vY2Vn45p1NxwVSZMgwcSoGTw1KlTXsgEDB30AGQBDwhQWhlwQTbCcLD1nEt6kyQgJpVwnY1cXNtNpQdFF5ZSGUqhWarluG9U+e+Y7HjMPoIE+IaDZtJ86aWXAhKk2wE+YqHeMcgF7hiBxysq06ircwMGziQBaeaDS5iYOJh6zDW9SRLgKQnQPS69QMk3RQfX786dO2XVqlWB4VNF56PhAxnas2fPoIu1cKdGlEJ3SdGbyt1c9eL3uh2Af6cxhDNnzqwy941tG0nAAERhJMBFvUkSEHOzYOyMqeEL9h4wupDW7ac3L9aw+0GT7cup8DusgG+88Ua57777htw06YKbM2qsTAKI2zVV+SEuIGwryFUs7IA4X3GwSUAaGXBJNmwS4Kre9J4E+BYYiP6uXbtWVq9eHbg69UkbAORKYKASIF3p+R4Y2NHR0bhd07dg0TAiDI+IbziYxi+tXsB8ciWQ2MbBVb3pPQlQIzDn7E1hrqcS1qhXc1W4YcMGgQszKqWy66mW7ZWwjziYfY46GmfODVdlIskRQddxsFfAaeeDK7Jh4uCy3iQJqKs\/m+0mAkSACBABIpATAZKAnACyOBEgAkSACBCBuiJAElDXkWO7iQARIAJEgAjkRIAkICeALE4E2oGAncbUbANOOVx++eXyvve9L4j1aPej++xmOmK7TbrnqvEp7W4zv08EfEGAJMCXkWY\/nUXATHeLtL9Ve+yjd1Ht0+RNVSAuVcOQ7SECZSFAElAWsqyXCLQIgTASoGlNYVBtr4FGt+upiIsuukjwu+uuuy5IFbtkyRLBRUK6KkddF1xwgezbty9IrGSeHjAvHkJ5zbJndj3s\/D3y0+Mx6yIJaJHA8DNEwECAJIDiQARqjkAcCbCTv8DI9\/T0CFJF42isEoDly5cLUgfDkB85cqSRKGjbtm2CWydRxr5YB1n1Zs2aFWw74DsgD7iDAUmH9DFJgJ2ZEnc3zJ07V+DBIAmouSCy+bVEgCSglsPGRhOBcwg0IwFIdKPpb5Hl0Ux7iho0GxySP5neA9SpN0mCBOCBlwCPGnVkV9R3UD5qWyKKBJhEQevFT24HULqJQOsQIAloHdb8EhEoBYEkJABufPOBq9+8A0DTBeNGSRhhmwTo700SYKYVBgkIS6tqvm+SiBUrVgTNMQMB6QkoRTxYKRFoigBJAAWECNQcgTgSoGlc7aDBsDsDokiA6QnQ7YUsngATavsyFpKAmgsim19LBEgCajlsbDQROIdAmpgADeS79957gwr0zoQ4TwAIA2IF8Oj2woQJE4J\/p4kJMLcGQAIYE0BJJgLtRYAkoL348+tEIDcCcSTAPh1g3hWRlAScPn1aDhw4MOjUABqe9nRAVFt02wA\/GROQWyRYARFIjABJQGKo+CIR8BMBM2AwCwLME5AFNZYhAq1BgCSgNTjzK0SgtggUQQIQCMiMgbUVATbcYQRIAhweXHaNCBABIkAEiEAzBEgCKB9EgAgQASJABDxFgCTA04Fnt4kAESACRIAIkARQBogAESACRIAIeIoASYCnA89uD0YA2e5w5t3OrDd69OggZ347b+fTtiHtr2bdq\/L46TFAtNG+R6BZu7OWC6sTeQ2OHz+e+bihHn1cuHBh5jqqPEZsGxFQBEgCKAtEQCRIeQsSUEVDW+W2VVF4ijDgRdRRRWzYJiJgI0ASQJkgAglIgBqFrq6uIGnOxIkTBWlzu7u7g+twcTWuXouLI3V6Va55LA6rU9zaZ78fNgBmEh7Ugdv9TIKCs\/eaf7\/Z0Tut26wPv9PrhPFvbRf+bV8HrH2BR0T7Ds8IHvRdV8pah3kjoXoCnnjiCbn\/\/vsDkvWlL30pKPvFL35RXnjhheB2Qv0mshbiJkPTgxDWT\/xd3zOvQTYzGqpHR8ekWf+b4a99xu2I5rXHnDREwBUESAJcGUn2IxcCcattNSIw\/uriNo26uulhNO0Uu2q8w94Pa7S6xc1rfZU84DumwdXUvc08GLabfcuWLY024jphNeYzZswYZIRhSPFdZBjUvz3\/\/PPB9khaEgDCYtYD4w\/CgKuJ8X0QmUWLFoV+H++Z\/dT3otpir+Lt\/wexAClpts1jj7cSmXZvDeUSchYmAiEIkARQLIiA4QmwYwJ0lX3ixIlBK19zBa2r6jAiYRocGFwYVXMV3mwVqqtsu16TaOgVwFipRu2\/m6l6ba+BbRDx\/3v27AnuCcAVwjDWavjMd9OSANPoRhElmwSYZMXs51133SV333134B0xV\/9KhOKMfhzhQ9+0DuBlEy98hw8RcAUBkgBXRpL9yIVAnGEI2yM2V+QwDGHvFE0C1FDCOJuP7ca3wTBd\/vibkgEYWt260DIaDPn44483PAYwwq0kAWrow\/r5wAMPyMaNG4PmgviA5JjxHPY42KQpbqxNEhC23YGx1m2SONxzCSULE4EWIEAS0AKQ+YnqIxBnGJKQgKI9AboKjfMEpEHXPAUB9zyeKNc4DF2VPAHaT3t7I44E2N6OuLGOIwHYwli\/fr18+ctfDsjI7Nmzg3gNPkSgjgiQBNRx1NjmwhGIMwxJSAAalSQmIG47oOiYALtvpgdD9+Sx4m0WEzBt2rRgtX3y5MlgewDlzNW3roybBQbqtkLS7QCNSYiKCUjqCYjbHggTJruM7fVBGeC6ePFiWblyZVuPkBY+GVihVwiQBHg13OxsFAJReQLwvhnAZp4bDzMMSgTUxW5GlEe9H9Ymsz1lnA4IaxfakfR0APImmJH7euKhSBKA0wJxpwPCtgPCYiA0pgOxE0lyP8SRAP0G6qpD7gbOfCIQhQBJAGWDCBCBUARs0mLvrfsKGwmAryPvZr9JAtwcV\/aKCORGwFxRa2VxWxm5P1qDCuwgS2JSg0FjEyMRIAmgcBABIkAEiAAR8BQB70iAmc0tyd6gp3LBbhMBIkAEiIAHCHhFAtS9ySM9Hkg2u0gEiAARIAKxCHhFAnikJ1Ye+AIRIAJEgAh4hEClSUCzY1v2GCXJ3GUH9PBCEI8knV0lAkSACBCBIQhUngSsXbtWVq9eLUhbGvWALCR5zyyvWwOTJ09u3Bd+7bXXNl7Zt2+ft+KCM9UdHR3e9l87ThwGkPAVh\/N\/+EO5cvZseXPSJHll167cOIw9q19efuYZeavG88tXeQhTiGPHjq29nqw0CTDRtVfx+rckHoCoUUKQIB5N9gEScPTo0doPat4OHDt2TFwQbuKQF4GB8t7Kw4EDIlOninR1iezfnx+H7m6RHTtE5s0TOXsTYzEj1NpavJUHC2ZXcKgFCdBtARjrPDm6QSR6e3sDox9WJ0mA50rf0Ume10S4ouxS4wCDDcN91mjnxqGvT0RXjmfOpG5OVQrkxqEqHcnZDldwqA0JSOvub7b6D0vpivdJAkgCTLlxZZLn1HX5V8B5G9Cu8kWTAPQDngV4GI4dE+nsbFfPcn2X88ItPVkLEgDIkUO8r6+v1DzdJAFuCXcuTeezG5wekQEEyiABusUAAgAiUMOHJMAtPVkrErBixYohUyZPTIBdGUmAW8KdV79S2XkuD2WQAECKLQFsDezfPxBvULOH88KteVELEtCq8\/0kAW4Jd17dSmXnuTyURQKsevPKaavLc164NS9qQQJwnG\/Lli1y6623Nj0qmHcykAS4Jdx55YHKznN5KIsEANZhwxTc2sUGcF64NS9qQQIAuRnZn1e5R5UnCXBLuPPKCZWd5\/JQJglYu1ZkzRqRu+4Swb9r9HBeuDUvakECmmUOZExA8dqDk9ytSZ5XQryVhzJJgAYIYnBqdlzQW3mwJpIrONSCBORVYknL0xNA42fKiiuTPKn8R73nLQ5lkgCArccFkTgIuQhq8ngrDyQB7ZXQF198UZYuXSobN26U8ePHB0cG9+zZI9u3by8sToAkgCSAJGDoPPdW6ZdNArT+sxkJ26thk3\/dW3kgCUguJEW\/GXUFMIjAwYMH5Z577pERI0bk\/ixJAEkASQBJQAOBskkAPlTD44IkAW7pyVpsB0RdEJTl4qBmTIEkwC3hzssKqew8l4dWkIAaHhfkvHBrXtSCBAByrPrvv\/9+6enpCbYDNFgQdwnoBUB5lT5JgFvCnVceqOw8lweN4EcU\/+rV5aVPrtlxQc4Lt+ZFbUgAYEdcQHd3t\/T39wejsGHDhsY1wHkVPsqTBLgl3HllgsrOc3loFQmo2e2CnBduzYtakYC8Sj2uPEmAW8IdN95xf6ey81weWkUCanZckPPCrXlRaRKQdM8\/6XtxSp8kwC3hjhvvuL9T2XkuD60iAYBZjwvW4D4Bzgu35kXlScD8+fPl8OHDcfpaikgaRBLglnDHCk3MC1R2nstDK0mAegNqcFyQ88KteVFpEpBXiactTxLglnCnHX\/7fSo7z+WhlSQAUNfkuCDnhVvzgiTA0PwkAW4JN0lAXgQ8l4dWk4CaJA8iCXBrXpAEkAQMsRSc5G5N8rxUwFt5aDUJwEDpccEKxwZ4Kw\/WRHIFB5IAkgCSgAgr6cokJwnIiEA7SEANvAGcF24tEkgCSAJIAkgCmlpJb5V+O0hADbwB3soDPQEZ2XSNijEmwC2Gm1f0qOw8l4d2kQD9bkVPCnBeuDUv6AmgJ4CeAHoC6AkIQ6BdJABtqfBJAZIAkoC8C6zKlqcnwC3hzitoVHaey0M7SUBf3wARwFOxIEHOC7fmRS09Abg++Gtf+5pccsklsnz58iDn\/86dO2X69OkycuTIzLqfJMAt4c4sCGcLUtl5Lg+a07+nR2TevPIuEIoSVM0iWLFtAc4Lt+ZFLUnAt7\/9bZk2bVowEtu3b5fZs2fL3\/3d35EE5LV6NH6DEKSyc0vZpZ4e7SYBaLASgRkzRL7zndRdKKMA54Vb86KWJGDXrl3yoQ99KLhS+OTJk\/JXf\/VXMnz4cPnc5z5HT0ABs56T3K1JnlckvJWHKpAADJ7mDqiIR8BbebAmkis41JIEvP7663LgwAGZMmWKXHjhhfLqq6\/K3r17A08AtgiaPYcOHZI5c+YEr9hXEXM7gMbPlB1XJjlJQEYEqkICzFsGQQSwPdHZmbFT+YtxXrilJ2tJArKKMW4bXLx4saxcuTKoYv369bJ58+aG94AkwC3hzionWo7KznN5qAoJwDAgUBBbA\/iJZ80akdWr84p4pvKcF27Ni9qQAHMFb0pumtsDUcdXvvKVII5gxIgRQVAh4gkmTZoUVFk6CcAEBoMv+2emqX2uECe5W5M8pzi0PiAub4OLKl8lEqBE4JFHBhv\/T3xC5C\/+QgQeghZ5B3Lph7y6L8\/Y5v22VT4XDnn6UXDZWpAArOBxpfCyZcsaBjsLDiABu3fvlnvuuScoDhIwefJkmTlz5kB12HuzJ5Iy7ywfZJn6InDmjL\/Gzxo1V5RdamGsGgkwO6Btszul+svUY\/rvOF0W9fe4cqmBdaPAyU2bZNSSJbXvTG1IwNq1a2X16tW5Av+SkIDpIrL37LDi33js\/\/92R0fw+z\/\/9a+Dn\/\/1sssq8\/\/D3nwztv1x\/ePfRTDB\/\/zb35bzzz9fHnrooWB8b7nlluBnVf\/\/1ptvlmFvvCG75syRd\/zsZzL7bDT5t8eMCdr9qX\/5l+Dnt\/7gD1L9\/67OTnnX+ecH5aPKav2F\/n3YMPnW7\/9+Q8mW\/v2f\/rTxPXwLc+nJN98MZOG1T31Kbr75ZnnsscdCZUHlI0o2yvg72vfowoVy\/o9+JCOee07+4tChyLmP7zfTba36e5TuVH0apUur+PeeL32JJKCVFAgGvLe3N\/AGZH0SbQd8\/\/uDXfZZP1bjct6u\/DBmxurv2JQpMlYTtlRxPLFCg3sYyWQQPManHATOnAnqrc28sFfu+v9JtgvCtistVGuDQznS0KjVFRwq7QnQbYDDhw9HDmeamAAGBiabFa4Id7LeWm\/VgQTgprmdOwcbfihv\/W\/KlMHbWkmUfxRYfX3y8u\/9nlz99tvFxLNkGpSEhQre8w1id7DXfvbxel4YQ0AcBsBwBYdKk4CEUz\/Va2aAIfINaFAgKik9MDBVS9v3sivCnQnBqpMATR6DzsG4I0K85KAwr+WBxm\/INKI8kARk0q11KEQS4JZwZ5K5qpIA84iYGv958zJ1MW0hKn3OC1NmKA9uyYN3noBmCpAkwC3hTmvsgverSALMy2RAABADkMfFnxIYKn3OC5KAoZPGlXlBEmCMLUkAlV3lSIBJABYtEvn611Oa8Pyvu6Ls8iJBHKgfXCRDJAEkAdzzMxGokifA3AKA6x\/pYtvw0PjR+Llo\/PJOJVfmBUkASQBJQFVJwOc+J\/LXfz3g+j92LK\/OylzeFWWXGYCzBYkDyZCLZIgkgCSAJKCKJADHAOGVaEMMgC0QNH40fi4aP5LCAQRqQQJwvr+IjIFxg86YACq7SsQEmNsA2AJo0SmAqPlBEsB5QRIwdHa4Mi9qQQIAPy7+6ezsPJfnP86iZ\/g7SQCVXSVIAPKRb948cP4fJwHa\/Lii7PLCSByoH1wkQ7UgAc0yB6bJGBinBEgCOMnbTgL0NAC2AeAFMDLWxclvWX+n8eO8cNH45Z0vrsyLWpCAvIOVtDxJAJVd20mAZgRcvlxkw4akolvqe64ou7wgEQfqBxfJUG1IwBtvvBFc\/bt3r95xJzJ9+vTgWuARI0bknd9BeZIATnJZu1ZkzZrgv2N\/+ZetvUAIlwCBBLT5NIA9mWj8OC9cNH55jYYr86IWJEAJwOjRowfdIog4gf7+\/sKIAEkAlV1bSYB6AUBCcCdARR5XlF1eOIkD9YOLZKgWJCDqdEDRpwZIAjjJ20YCzFiANuYECDOUNH6cFy4aP5LCAQRqQQLQUKz6sRXQ09Mj48ePlxdffFG6u7uDLYFly5blHU9uBxgIeq3027UdUFEvAMTCa3ngvBiiWykPbpHC2pAAwP6Nb3xDVqxY0RDKDRs2FHpkkJ4At4Q7EzNsBwkwvQAtvhwoCUZU+pwX9AQMnSmuzItakYAkCivPOyQBVHZt2Q54+GGR+fMHkgK16X6AZvPGFWWXRzfQI3IOPcqDW3qyFiSg6L3\/KGVAEuCWcGdS+q32BJi3BMILUIG8ADZuVPqcF\/QE0BOQSZ8WWYgZA4tEs3ldXiv9VpOAih4LpNJ3V+nn1SRe6wcDPFdwqI0nYP78+XL48OEh8suMgXmnNJXdIARaTQI0ILACdwRESZIryi7vTCEO9Ii4SI5rQQLyTt6k5bkdwEne0piACh8LdFHZJdUDJEP0FCaRFVdIYS1IAGMCkohkce+4ItyZEGmlJ0C3AioaEKj4eS0PDrp\/M80L4jAENlfmRS1IANBnTEDeqZu8vCvCnbzHxputJAG6FVDRgECSgMES5PW8IAkgCcikUAsqxFsECwIyYTVeK7tWkYAaBASSBJAEhKkMr\/WDg2SoNp6AhPYr12uMCRiAz+tJ3ioS0N0tsmNHZXMDmBPJa3lwUOnnUpK+6wcH5YEkwBhUkgCSgJYEBtYgNwBJwFBTSTJE\/eDivKg0CbADAnFfwM6dO2XVqlXB9cFFBwySBHCSt4QE1GgrwHvPkIMrP3oC8iLglp70jgQgwHDr1q3BKOJqYr2QCP9PEuCWcGea6q3YDtBvVDwgkDEBjAlgTEC0FnHFM+QVCXjjjTdk+fLlMnv2bJk0adKQ0SUJGIDEaxwMEnDtzp1y9OjRTFwislDNtgK8lwduF1JPRkxmV\/SkVyQA2weLFy+WlStXBtcR248rg5rXanmNQ9kkoGZbASQB52aT1\/OCZMhZe+EVCTh06JDMmTOnMZgLFiyQZcuWNf4fk5yP3wh86rXXZNPJk\/KtCy+UO0aNKhyMz7\/6qnz+f\/5P+fp73iNfv+SSwutnhUSACLQOgcI9ha1reuNLlScBUXcGaA+y3h2gWwOTJ0+WmTNntgF6fpIIEAEiQASIQHsRqDQJyAsNThN0d3dLf3+\/TJ8+Xe65557gVIE+CBLEY3oD8n6T5YkAESACRIAI1AUBp0mAPQjYDujt7Q2MvmYhxL\/DggTrMoBsJxEgAkSACBCBrAh4RQIAknlE0I4JyAoiyxEBIkAEiAARqCMC3pGAsEEyAwY3bNjgdIyAxkLs3bs3gMIkQlE4uI7PN77xDTl48GBju8hHHKK2znzDArKwYsUKr+cGMOjr62tsk6aVAVf0hYmDy3rTexJgHhvEzF+\/fr1s3rxZRo4cWUdSF9tmU7B1S2TWrFkybdq0xvFJEwf8W49VuoiPGr+JEycGJACTPay\/LuNgH52Ft2zKlCkyYcIEr7CALCxdulQ2btwol156qSAoGduFPuGgnlJdHETpx6j54Mo8sXFwWW96TwLAWjHg27dvD4IGmyUTirWwNXxBr2geM2ZMKA66heIiPjD469atk3HjxskLL7wQkIDDhw97h4MZK2OKcNTccFUmTIMHEqBk8NSpU17IBAwd9ADipvCAAKWVARdkIwwHW7W7pDdJAg4dkt27dwcGAA9IgC\/HBk2lB0UXhgOUgqv4YLLjMfsIEuAbDlD06PNLL70UkCA9SeMjFuodg1yA+MIjqPjYOsLVuWGemkrbd5cwiTo95preJAnwlATYKZTTTva651YwL6MyjZ2Phg9kaM+ePYO8YSDCUQrdJUUf5fnA73U7AP\/2iRiSBAxIRRgJcFFvkgR4uB0Qlj45rduv7scqzQAwNQRYAd94441y3333DdkecsHNGbVbZRJAbImp8kNcQNhWmatY2AFxvuJgk4A0MuCSbNgkwFW96T0J8C0wMOr65bQBQC4FTppG0PfAwI6OjsaWmG\/BomFEGB4R33AwjV9aveBKYKDtCXBZb3pPAjDY5pGWXbt2OZ08yMyToCtDPRYZhYPr+NgrYR9xMPscdWzUnBuuykSSI4Ku42CvgNPOB1dkw8TBZb1JElDDiH42mQgQASJABIhAEQiQBBSBIusgAkSACBABIlBDBEgCajhobDIRIAJEgAgQgSIQIAkoAkXWQQRajICdxtT8PE45XH755fK+972vEimwdZ897CZPbbfuubqetrvFYsLPEYFYBEgCYiHiC0Sg2giY6W7Hjx9fucbaR++iGqjJm+qeg6JyA8AGEYEmCJAEUDyIQM0RCCMBmtYUBtX2Gmh0u56KuOiiiwS\/u+6664JUsUuWLJH+\/n7RVTnquuCCC2Tfvn1BRkHz9IB58RDKa5Y9E9Kw8\/dbt24NXjHrIgmouSCy+bVEgCSglsPGRhOBcwjEkQA7+QuMfE9PjyBV9Jw5cxoEACmzkToYhvzIkSONREHbtm0T3DqJMvbFOsiqhwuoQDbwHZAHpNdF0iF9TBJgZ6bE3Q1z584VeDBIAijVRKD1CJAEtB5zfpEIFIpAMxKARDea\/hZZHs20p2iEZoND8ifTe4A69UZNkAA88BLgUaOO7IrmrZtR2xJRJMAkClovfnI7oFDxYGVEoCkCJAEUECJQcwSSkAC48c0Hrn7zDgBNF9zZ2RkYYZsE6O9NEmCmFQaJCEurar5vkogVK1YEzTEDAekJqLkgsvm1RIAkoJbDxkYTgXMIxJEAvRLXDhoMuzMgigSYngDdXsjiCTDHzb6MhSSAUk0EWo8ASUDrMecXiUChCKSJCdBAvnvvvTdog96OF+cJAGFArAAe3V6YMGFC8O80MQHm1gBIAGMCChUFVkYEUiNAEpAaMhYgAtVCII4E2KcDzLsikpKA06dPy4EDBwadGgAKaU8HRLVFtw3wkzEB1ZIvtsZtBEgC3B5f9o4I5EbADBjMUhnzBGRBjWWIQGsQIAloDc78ChGoLQJFkAAEAjJjYG1FgA13GAGSAIcHl10jAkSACBABItAM0y\/8PgAAATlJREFUAZIAygcRIAJEgAgQAU8RIAnwdODZbSJABIgAESACJAGUASJABIgAESACniJAEuDpwLPbRIAIEAEiQARIAigDRIAIEAEiQAQ8RYAkwNOBZ7eJABEgAkSACJAEUAaIABEgAkSACHiKAEmApwPPbhMBIkAEiAARIAmgDBABIkAEiAAR8BQBkgBPB57dJgJEgAgQASJAEkAZIAJEgAgQASLgKQIkAZ4OPLtNBIgAESACRIAkgDJABIgAESACRMBTBEgCPB14dpsIEAEiQASIAEkAZYAIEAEiQASIgKcIkAR4OvDsNhEgAkSACBABkgDKABEgAkSACBABTxEgCfB04NltIkAEiAARIAIkAZQBIkAEiAARIAKeIkAS4OnAs9tEgAgQASJABEgCKANEgAgQASJABDxF4P8H5JsGynV6OwUAAAAASUVORK5CYII=","height":378,"width":513}}
%---
