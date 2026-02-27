function [AA, BB] = calcLinearization(xR, parSys)
% Calculates a linearized model of the three-tank system for a given steady state

% Read states from given steady state
h1 = xR(1);
h2 = xR(2);
h3 = xR(3);

% Read system parameters from struct
Atank = parSys.Atank;

% Calculate derivatives of volumetric flow for outlet valves
[~, ~, ~, dqo1dh1, dqo2dh2, dqo3dh3] = calc_qo(h1, h2, h3, parSys);

% Calculate derivatives of volumetric flow for coupling valves
[~, ~, dq12dh1, dq12dh2, dq23dh2, dq23dh3] = calc_qc(h1, h2, h3, parSys);

% Calculate dynamic matrix of linearized system
AA = [-dq12dh1 - dqo1dh1, -dq12dh2                      ,  0; ...
       dq12dh1         ,   dq12dh2 - dq23dh2 - dqo2dh2  , -dq23dh3; ...
       0               ,            dq23dh2             ,  dq23dh3 - dqo3dh3] / Atank;
    
% Calculate input matrix of linearized system
BB = [1, 0; 0, 0; 0, 1] / Atank;
    
end

