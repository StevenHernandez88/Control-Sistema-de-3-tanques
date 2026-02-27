%% 
% Parameterfile Three-tank 
% Exercise 1 of Optimization-based Control Methods VU
%
% --------------------------------------
% [0] https://wiki.anton-paar.com/at-de/wasser/

%% == clear workspace and initialize system ==

close all;
clear;
clc;

%% General parameters

% Trajektorienfilter
parSys.Tank3_on = 1;                % Connection to thrid tank: activated
parSys.noise_on = 1;                % Mesurement noise: activated            
parSys.big_step = 0;                % Large step in desired trajectory: activated


% Poles of the filter for trajectory generation
parInit.TrajGen.pol_fast = -1;            % Zeitkonstante Trajektorienfilter y1d: Überschreitet Systemdynamik
parInit.TrajGen.pol_slow = -0.023;        % Zeitkonstante Trajektorienfilter y1d: Angepasst an die Systemdynamik
parInit.TrajGen.pol_rate_limit = -0.06;   % Zeitkonstante Trajektorienfilter y1d: Angepasst an die Systemdynamik

% Sampling time
parSys.Ts = 2;  

%% Parameter of the system

parSys.Atank     = 1.539e-2;      % Base area of the tanks
parSys.rho       = 997.0;         % Density of water at 25°C, see [0] 
parSys.eta       = 8.9e-4;        % Dynamic viscosity of water at 25°C, see [0] 
parSys.g         = 9.81;          % Gravitational acceleration
parSys.alphao1   = 0.0583;        % Contraction coefficient outlet valve 1
parSys.Do1       = 15e-3;         % Effective diameter outlet valve 1
parSys.alphao2   = 0.1039;        % Contraction coefficient outlet valve 2
parSys.A2        = 1.0429e-4;     % Effective cross sectional area outlet valve 2
parSys.alphao3   = 0.0600;        % Contraction coefficient outlet valve 3
parSys.Do3       = 15e-3;         % Effective diameter outlet valve 3
parSys.alpha120  = 0.3038;        % Turbulent contracton coefficient coupling valve 12
parSys.D12       = 7.7e-3;        % Equivalent hydraulic diameter coupling valve 12
parSys.A12       = 0.55531e-4;    % Effective cross sectional area coupling valve 12
parSys.lambdac12 = 24000;         % Critical flow number coupling valve 12
parSys.alpha230  = 0.1344;        % Turbulent contracton coefficient coupling valve 23
parSys.D23       = 15e-3;         % Equivalent hydraulic diameter coupling valve 23
parSys.A23       = 1.76715e-4;    % Effective cross sectional area coupling valve 23
parSys.lambdac23 = 29600;         % Critical flow number coupling valve 23

% Constraints of pump flows
parSys.qi1max = 1.2e-4;           % Maximum flow pump 1 / Caudal de salida del tanque 1-2 (estaba en parSys.qi1max = 4.5e-3 / 60;)
parSys.qi3max = 4.5e-3/60;        % Maximum flow pump 2
parSys.qi1min = 0;                % Minimum flow pump 1
parSys.qi3min = 0;                % Minimum flow pump 2

% Constraints of water height
parSys.hmax = 0.6;                % Maximum water height
parSys.hmin = 0;                  % Minimum water height

% Filter for desired trajectory
parInit.TrajGen.N = 3;
parInit.TrajGen.charpol = poly(parInit.TrajGen.pol_fast*ones(1,parInit.TrajGen.N));
% parInit.TrajGen.charpol = poly(parInit.TrajGen.pol_slow*ones(1,parInit.TrajGen.N));
trajSys = c2d(ss(tf(parInit.TrajGen.charpol(end),parInit.TrajGen.charpol)), parSys.Ts, 'zoh');
parInit.TrajGen.Phi = trajSys.A;
parInit.TrajGen.Gamma = trajSys.B;
parInit.TrajGen.C = trajSys.C;
parInit.TrajGen.D = trajSys.D;

clear trajSys

% Simulation length
parInit.T_end = 1200;

% Measurement noise
parInit.meas.noiseAmplitude = 5e-5;
parInit.meas.noisepow = parInit.meas.noiseAmplitude^2*parSys.Ts;


%% Steady state and initial conditions

% Parametrize desired steady state in second tank
h2d = 0.10;

% Calculate desired steady state for first and third tank for h1=h3
xSol = fsolve(@(x) wrapper(x, h2d, parSys), [h2d; h2d; 0]);
h1d = xSol(1);
h3d = xSol(2);
qi1d = xSol(3) / 60e3;

% Calculate volumetric flows from desired heights
[qo1d, qo2d, qo3d, ~, ~, ~] = calc_qo(h1d, h2d, h3d, parSys);
[q12d, q23d, ~, ~, ~, ~] = calc_qc(h1d, h2d, h3d, parSys);

% Steady-state condition for first and third tank
qi3d = qo3d - q23d;

% Write steady state to struct
parSys.xR = [h1d; h2d; h3d];
parSys.uR = [qi1d; qi3d];

% Initial conditions
parSys.h1_0 = h1d;
parSys.h2_0 = h2d;
parSys.h3_0 = h3d;


%% Linearization, discretization and control design

% Calculate matrices for linearized continuous dynamics
[AA, BB] = calcLinearization(parSys.xR, parSys);
CC = [0,1,0];

% Define linearized continuous-time system dynamics
sysC = ss(AA, BB, CC, 0);

% Calculate ZOH discretization for given sampling time Ts#
sysD = c2d(sysC, parSys.Ts, 'zoh');
% parSys.Phi = sysD.A;
% parSys.Gamma = sysD.B;
% parSys.C = sysD.C;

%Parametros para el MPC
parSys.Ad = sysD.A;
parSys.Bd = sysD.B;
parSys.Cd = sysD.C;
parSys.N_mpc = 20; 


% Calculate state feedback gain
Qlqr = diag([0 10 0]);
Rlqr = diag([1, 1])*(60e3);
[K,S,e] = dlqr(sysD.A,sysD.B,Qlqr,Rlqr,0);
parSys.K = -K;

% Calculate feedforward gain
HH = (parSys.K / (eye(3) - sysD.A - sysD.B*parSys.K)) * sysD.B + eye(2);
sol = [HH.'*HH, ((sysD.C/(eye(3) - sysD.A - sysD.B*parSys.K))*sysD.B).'; ... 
    (sysD.C/(eye(3) - sysD.A - sysD.B*parSys.K))*sysD.B, 0] \ [zeros(2,1); 1];

parSys.G = sol(1:2);


%% Initialization of the linear MPC

% Number of samples for the prediction horizion
parMPC.N = 0;

% Cost function weights
parMPC.qq = 0;
parMPC.RR1 = zeros(2);
parMPC.RR2 = zeros(2);

% Set up linear model predicitve controller here ...




%% Definition of wrapper function

function res = wrapper(x, h2d, parSys)
% Wrapper function for calulating the steady-state h1=h3

    h1d = x(1);
    h3d = x(2);
    qi1d = x(3) / 60e3;

    [qo1d, qo2d, qo3d, ~, ~, ~] = calc_qo(h1d, h2d, h3d, parSys);
    [q12d, q23d, ~, ~, ~, ~] = calc_qc(h1d, h2d, h3d, parSys);
    
    res = [qi1d - q12d - qo1d; ...
            q12d - q23d - qo2d; ...
            q23d - qo3d] * 60e3; % Residuum in l/min
end



