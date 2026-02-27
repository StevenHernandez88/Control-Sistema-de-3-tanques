classdef NMPCController < matlab.System

    methods (Access = protected)

        function u = stepImpl(~, input)
            % Declarar funciones externas (no compilables)
            coder.extrinsic('evalin');

            x    = input(1:3);
            yref = input(4);

            % Leer del workspace base
            solver = evalin('base', 'parSys.solver_nmpc');
            lbw    = evalin('base', 'parSys.lbw');
            ubw    = evalin('base', 'parSys.ubw');
            lbg    = evalin('base', 'parSys.lbg');
            ubg    = evalin('base', 'parSys.ubg');
            N      = evalin('base', 'parSys.N_mpc');
            qi1max = evalin('base', 'parSys.qi1max');
            qi3max = evalin('base', 'parSys.qi3max');

            % Resolver NMPC
            p   = [x(:); yref];
            u0  = zeros(N*2, 1);
            sol = solver('x0', u0, 'lbx', lbw, 'ubx', ubw, ...
                         'lbg', lbg, 'ubg', ubg, 'p', p);
            U_opt = full(sol.x);
            u     = U_opt(1:2);
            u(1)  = min(qi1max, max(0, u(1)));
            u(2)  = min(qi3max, max(0, u(2)));
        end

        function num = getNumInputsImpl(~)
            num = 1;
        end

        function num = getNumOutputsImpl(~)
            num = 1;
        end

        function out = getOutputSizeImpl(~)
            out = [2, 1];
        end

        function out = getOutputDataTypeImpl(~)
            out = 'double';
        end

        function out = isOutputComplexImpl(~)
            out = false;
        end

        function out = isOutputFixedSizeImpl(~)
            out = true;
        end

    end
end