%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation model for the three-tank system
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs:   u1(1)...  qi1       Volumetric flow pump 1
%           u1(2)...  qi3       Volumetric flow pump 2
%           u2(1)...  is_o1_open   Indicates that outlet valve 1 open
%           u2(2)...  is_o2_open   Indicates that outlet valve 2 open
%           u2(3)...  is_23_open   Indicates that coupling valve 23 open
%
% states:   x(1)...   h1        Water height tank 1
%           x(2)...   h2        Water height tank 2
%           x(3)...   h3        Water height tank 3
%
% outputs:  y1   ...            State vector
%           y2   ...            Volumetric flows
%
% parameters:
%           Atank     Tank-Grundflaeche
%           rho       Dichte Wasser
%           eta       Viskositaet
%           g         Erdbeschleunigung
%           alphaA1   Kontraktionskoeffizient AV1
%           DA1       Durchmesser AV1       
%           alphaA2   Kontraktionskoeffizient AV2
%           A2        Querschnittsflaeche AV2       
%           alphaA3   Kontraktionskoeffizient AV3
%           DA3       Durchmesser AV3
%           alpha12_0 Kontraktionskoeffizient ZV12
%           A12       Querschnittsflaeche ZV12
%           Dh12      hydraulischer Durchmesser ZV12        
%           lambdac12 kritische Fliesszahl ZV12
%           alpha23_0 Kontraktionskoeffizient ZV23
%           D23       Durchmesser ZV23
%           lambdac23 kritische Fliesszahl ZV23
%           hmin      Minimale Fuellhoehe
%           h10       Anfangszustand h1
%           h20       Anfangszustand h2
%           h30       Anfangszustand h3
%           qZ1max    Maximaler Zufluss qZ1
%           qZ3max    Maximaler Zufluss qZ3
%           flaglimqZ Stellgroessenbeschraenkung ein/aus
%           h_max     Maximale Füllhöhe
%           h_min     Minimale Füllhöhe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample Time: Continuous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

function [dx, q_pump, q_valve] = watertank_mfun(x, qi, is_open, parSys)

    % allows compilation of p-code file
    coder.allowpcode('plain');
    
    % Read system parameters from struct
    Atank     = parSys.Atank;
    rho       = parSys.rho;
    eta       = parSys.eta;
    g         = parSys.g;
    alphao1   = parSys.alphao1;
    Do1       = parSys.Do1;
    alphao2   = parSys.alphao2;
    A2        = parSys.A2;
    alphao3   = parSys.alphao3;
    Do3       = parSys.Do3;
    alpha120  = parSys.alpha120;
    D12       = parSys.D12;
    A12       = parSys.A12;
    lambdac12 = parSys.lambdac12;
    alpha230  = parSys.alpha230;
    D23       = parSys.D23;
    A23       = parSys.A23;
    lambdac23 = parSys.lambdac23;
    hmin      = parSys.hmin;
    hmax      = parSys.hmax;
    qi1max    = parSys.qi1max;
    qi3max    = parSys.qi3max;
    qi1min    = parSys.qi1min;
    qi3min    = parSys.qi1min;

    % Read states
    h1     = x(1);
    h2     = x(2);
    h3     = x(3);

    % Read pump flows and consider constraints  
    if qi(1) > qi1max
        qi1 = qi1max;
    elseif qi(1) < qi1min
        qi1 = qi1min;
    else
        qi1 = qi(1);
    end

    if qi(2) > qi3max
        qi3 = qi3max;
    elseif qi(2) < qi3min
        qi3 = qi3min;
    else
        qi3 = qi(2);
    end

    % Read opening of valves
    is_o1_open = is_open(1);
    is_o3_open = is_open(2);
    is_23_open = is_open(3);
  
%% Volumetric flows
    
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

    % Volumetric flow for outlet valve 1
    if is_o1_open
        qo1 = alphao1 * Do1^2*pi/4 * sqrt(2*g*h1);
    else
        qo1 = 0;
    end

    % Volumetric flow for outlet valve 2
    qo2 = alphao2 * A2 * sqrt(2*g*h2);

    % Volumetric flow for outlet valve 3
    if is_o3_open
        qo3 = alphao3 * Do3^2*pi/4 * sqrt(2*g*h3);
    else
        qo3 = 0;
    end

    % Volumetric flow for coupling valve 12
    lambda12 = D12 * rho/eta * sqrt(2*g*abs(h1-h2));        % Flow number
    alpha12 = alpha120 * tanh(2 * lambda12/lambdac12);      % Contraction coefficient

    q12 = alpha12 * A12 * sqrt(2*g*abs(h1-h2)) * sign(h1-h2);

    % Volumetric flow for coupling valve 23
    lambda23 = D23 * rho/eta * sqrt(2*g*abs(h2-h3));        % Flow number
    alpha23 = alpha230 * tanh(2 * lambda23/lambdac23);      % Contraction coefficient

    if is_23_open
        q23 = alpha23 * A23 * sqrt(2*g*abs(h2-h3)) * sign(h2-h3);
    else
        q23 = 0;
    end

%% Differential equation
    dx = zeros(3,1);
    dx(1) = (qi1 - q12 - qo1) / Atank;
    dx(2) = (q12 - q23 - qo2) / Atank;
    dx(3) = (qi3 + q23 - qo3) / Atank;

    % Consider state constraints (overflow)
    idx2large = ([h1; h2; h3] >= hmax) & (dx > 0);
    dx(idx2large) = 0;
    
    idx2low = ([h1; h2; h3] <= hmin) & (dx < 0);
    dx(idx2low) = 0;
  
%% Model outputs
  
    % Constraint pump flows
    q_pump = [qi1; qi3];
    
    % Valve flows
    q_valve = [qo1; qo2; qo3; q12; q23];

%% Function end
end






