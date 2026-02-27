function [qo1, qo2, qo3, dqo1dh1, dqo2dh2, dqo3dh3] = calc_qo(h1, h2, h3, parSys)
% Calculates the volumetric flows and derivatives in the outlet valves based on the tank heights

% Read system parameters from struct
g         = parSys.g;
alphao1   = parSys.alphao1;
Do1       = parSys.Do1;
alphao2   = parSys.alphao2;
A2        = parSys.A2;
alphao3   = parSys.alphao3;
Do3       = parSys.Do3;
hmin      = parSys.hmin;

% Consider the physical state constraints h > hmin
if h1 <= hmin                                         
    h1 = 0;
end

if h2 < hmin
    h2 = 0;
end

if h3 < hmin
    h3 = 0;
end

% Volumetric flow for outlet valves   
qo1 = alphao1 * Do1^2*pi/4 * sqrt(2*g*h1);
qo2 = alphao2 * A2 * sqrt(2*g*h2);
qo3 = alphao3 * Do3^2*pi/4 * sqrt(2*g*h3);

% Calculate derivatives of volumetric floww for outlet valves
dqo1dh1 = alphao1 * Do1^2*pi/4 * sqrt(0.2e1) * (2 * g * h1) ^ (-0.1e1 / 0.2e1) * g / 0.2e1;
dqo2dh2 = alphao2 * A2 * sqrt(0.2e1) * (2 * g * h2) ^ (-0.1e1 / 0.2e1) * g / 0.2e1;
dqo3dh3 = alphao3 * Do3^2*pi/4 * sqrt(0.2e1) * (2 * g * h3) ^ (-0.1e1 / 0.2e1) * g / 0.2e1;

end

