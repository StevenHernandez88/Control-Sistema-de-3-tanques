function val = nmpc_evalin(varname)
% Wrapper para evalin — declarado extrinsic desde NMPCController
    val = evalin('base', varname);
end

%[appendix]{"version":"1.0"}
%---
