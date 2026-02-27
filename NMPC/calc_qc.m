function [q12, q23, dq12dh1, dq12dh2, dq23dh2, dq23dh3] = calc_qc(h1, h2, h3, parSys)
% Calculates the volumetric flows and derivatives in the coupling valves based on the tank heights

% Read system parameters from struct
rho       = parSys.rho;
eta       = parSys.eta;
g         = parSys.g;
alpha120  = parSys.alpha120;
D12       = parSys.D12;
A12       = parSys.A12;
lambdac12 = parSys.lambdac12;
alpha230  = parSys.alpha230;
D23       = parSys.D23;
A23       = parSys.A23;
lambdac23 = parSys.lambdac23;
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

% Volumetric flow for coupling valve 12
lambda12 = D12 * rho/eta * sqrt(2*g*abs(h1-h2));        % Flow number
alpha12 = alpha120 * tanh(2 * lambda12/lambdac12);      % Contraction coefficient

q12 = alpha12 * A12 * sqrt(2*g*abs(h1-h2)) * sign(h1-h2);

% Volumetric flow for coupling valve 23
lambda23 = D23 * rho/eta * sqrt(2*g*abs(h2-h3));        % Flow number
alpha23 = alpha230 * tanh(2 * lambda23/lambdac23);      % Contraction coefficient

q23 = alpha23 * A23 * sqrt(2*g*abs(h2-h3)) * sign(h2-h3);


% Calculate derivatives of volumetric flows for coupling valves
dlambda12dh1 = D12 * rho / eta * sqrt(0.2e1) * ((g * abs(h1 - h2)) ^ (-0.1e1 / 0.2e1)) * g * sign(h1 - h2) / 0.2e1;
dlambda12dh2 = -D12 * rho / eta * sqrt(0.2e1) * ((g * abs(h1 - h2)) ^ (-0.1e1 / 0.2e1)) * g * sign(h1 - h2) / 0.2e1;

dalpha12dh1 = 0.2e1 * alpha120 * dlambda12dh1 / lambdac12 * (0.1e1 - tanh((2 * lambda12 / lambdac12)) ^ 2);
dalpha12dh2 = 0.2e1 * alpha120 * dlambda12dh2 / lambdac12 * (0.1e1 - tanh((2 * lambda12 / lambdac12)) ^ 2);

dq12dh1 = dalpha12dh1 * A12 * sqrt(0.2e1) * sqrt((g * abs(h1 - h2))) * sign(h1 - h2) + alpha12 * A12 * sqrt(0.2e1) * ((g * abs(h1 - h2)) ^ (-0.1e1 / 0.2e1)) * sign(h1 - h2) ^ 2 * g / 0.2e1;
dq12dh2 = dalpha12dh2 * A12 * sqrt(0.2e1) * sqrt((g * abs(h1 - h2))) * sign(h1 - h2) - alpha12 * A12 * sqrt(0.2e1) * ((g * abs(h1 - h2)) ^ (-0.1e1 / 0.2e1)) * sign(h1 - h2) ^ 2 * g / 0.2e1;


dlambda23dh2 = D23 * rho / eta * sqrt(0.2e1) * ((g * abs(h2 - h3)) ^ (-0.1e1 / 0.2e1)) * g * sign(h2 - h3) / 0.2e1;
dlambda23dh3 = -D23 * rho / eta * sqrt(0.2e1) * ((g * abs(h2 - h3)) ^ (-0.1e1 / 0.2e1)) * g * sign(h2 - h3) / 0.2e1;

dalpha23dh2 = 0.2e1 * alpha230 * dlambda23dh2 / lambdac23 * (0.1e1 - tanh((2 * lambda23 / lambdac23)) ^ 2);
dalpha23dh3 = 0.2e1 * alpha230 * dlambda23dh3 / lambdac23 * (0.1e1 - tanh((2 * lambda23 / lambdac23)) ^ 2);

dq23dh2 = dalpha23dh2 * A23 * sqrt(0.2e1) * sqrt((g * abs(h2 - h3))) * sign(h2 - h3) + alpha23 * A23 * sqrt(0.2e1) * ((g * abs(h2 - h3)) ^ (-0.1e1 / 0.2e1)) * sign(h2 - h3) ^ 2 * g / 0.2e1;
dq23dh3 = dalpha23dh3 * A23 * sqrt(0.2e1) * sqrt((g * abs(h2 - h3))) * sign(h2 - h3) - alpha23 * A23 * sqrt(0.2e1) * ((g * abs(h2 - h3)) ^ (-0.1e1 / 0.2e1)) * sign(h2 - h3) ^ 2 * g / 0.2e1;

end

