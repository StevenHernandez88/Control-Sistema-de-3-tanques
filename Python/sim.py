"""
COMPARACION FINAL CODIT 2026
Sistema de 3 Tanques: NMPC vs Koopman-B (sqrt(h)) vs Lineal GS

CASOS DE PRUEBA:
- Caso 1: Regulacion h2=0.25m (60s)
- Caso 2: Multi-escalon MONOTÓNICO ascendente (240s)
- Caso 3: Rechazo de perturbacion Delta_h1=+8cm (120s)

CONTROLADORES:
1. NMPC Optimizado (w_h2=20k)
2. Koopman-B con observables sqrt(h)  [CORREGIDO]
3. Lineal Gain-Scheduling LQR

CORRECCIONES APLICADAS (respecto a version anterior):
------------------------------------------------------
[C1] observable_function: Psi(h) en R^7, definida SOLO sobre el estado x,
     SIN incluir u dentro del mapa de lifting.
     Psi(h) = [sqrt(h1), sqrt(h2), sqrt(h3), h1*h2, h2*h3, h1*h3, 1]^T

[C2] La entrada u entra UNICAMENTE por B en la dinamica lineal de Koopman:
     z_{k+1} = A * z_k + B * u_k   (A in R^7x7, B in R^7x2)
     Esto es consistente con la formulacion EDMD estandar
     (Proctor et al. 2018, Korda & Mezic 2018).

[C3] Regresion EDMD corregida:
     Phi = [Psi(x_k) | u_k]  shape (M, 9)   <- u separado, no dentro de Psi
     Y   = Psi(x_{k+1})       shape (M, 7)

[C4] Condicion inicial corregida: z_0 = Psi(x_k)  SIN u
[C5] Referencia corregida:        z_ref = Psi(h_ref)  SIN u_S
[C6] h2 se recupera del espacio lifteado como h2 = z[1]^2
     (porque z[1] = sqrt(h2))
[C7] Dimension del espacio lifteado: n_z = 7 (antes era 9 con un diccionario
     distinto al reportado en el paper).
[C8] Se registra el tiempo por paso de control (ms/step) ademas del tiempo
     total, para reporte correcto en el paper.
[C9] Construccion del KoopmanMPC recibe x_ss (no z_ss) como argumento,
     y calcula z_ref = Psi(x_ss) internamente.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds, least_squares
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
import pickle

warnings.filterwarnings('ignore')

output_dir = Path('/mnt/user-data/outputs')
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SISTEMA DE 3 TANQUES
# ============================================================================

class ThreeTankSystem:
    """Modelo fisico no-lineal del sistema de 3 tanques."""

    Ts    = 0.3
    Atank = 153.9e-4
    g     = 9.81
    rho   = 997
    eta   = 8.9e-4
    qmax  = 75e-6

    alpha_o1, A_o1 = 0.0583, np.pi * (15e-3 / 2)**2
    alpha_o2, A_o2 = 0.1039, 1.0429e-4
    alpha_o3, A_o3 = 0.06,   np.pi * (15e-3 / 2)**2

    alpha_12_0, A_12 = 0.3038, 0.55531e-4
    lambda_c12,  D_12 = 24000, 7.7e-3

    alpha_23_0, A_23 = 0.1344, 1.76715e-4
    lambda_c23,  D_23 = 29600, 15e-3

    @classmethod
    def compute_steady_state(cls, h2_ref):
        """Busqueda robusta de estado estacionario."""

        def steady_state_eqs(vars, u3_val):
            h1, h3, u1 = vars
            h2 = h2_ref

            qi1 = cls.qmax * u1
            qi3 = cls.qmax * u3_val

            q_o1 = cls.alpha_o1 * cls.A_o1 * np.sqrt(2*cls.g*h1) if h1 > 0 else 0
            q_o2 = cls.alpha_o2 * cls.A_o2 * np.sqrt(2*cls.g*h2) if h2 > 0 else 0
            q_o3 = cls.alpha_o3 * cls.A_o3 * np.sqrt(2*cls.g*h3) if h3 > 0 else 0

            def q_coupling(h_up, h_down, A, D, lambda_c):
                dh = h_up - h_down
                if abs(dh) < 1e-8:
                    return 0
                lam   = D * (cls.rho / cls.eta) * np.sqrt(2*cls.g*abs(dh))
                alpha = (1.0/np.e) * np.tanh(2 * lam / lambda_c)
                return alpha * A * np.sqrt(2*cls.g*abs(dh)) * np.sign(dh)

            q_12 = q_coupling(h1, h2, cls.A_12, cls.D_12, cls.lambda_c12)
            q_23 = q_coupling(h2, h3, cls.A_23, cls.D_23, cls.lambda_c23)

            return [qi1 - q_12 - q_o1,
                    q_12 - q_23 - q_o2,
                    qi3 + q_23 - q_o3]

        u3_grid       = np.linspace(0.0, 0.3, 25)
        best_solution = None
        best_residual = float('inf')

        for u3_try in u3_grid:
            for h1_f, h3_f, u1_f in [
                (h2_ref*1.2, h2_ref*0.9, 0.5),
                (h2_ref*1.4, h2_ref*0.7, 0.4),
                (h2_ref*1.0, h2_ref*1.0, 0.6),
            ]:
                try:
                    sol = least_squares(
                        lambda vars: steady_state_eqs(vars, u3_try),
                        [h1_f, h3_f, u1_f],
                        bounds=([0.01, 0.01, 0.0], [0.55, 0.55, 1.05]),
                        max_nfev=500
                    )
                    if sol.success:
                        h1_ss, h3_ss, u1_ss = sol.x
                        residual = np.linalg.norm(sol.fun)
                        if (0.01 < h1_ss < 0.55 and 0.01 < h3_ss < 0.55 and
                                0 <= u1_ss <= 1.05 and residual < best_residual):
                            best_residual = residual
                            best_solution = (h1_ss, h3_ss, u1_ss, u3_try)
                except:
                    pass

        if best_solution is not None and best_residual < 1e-3:
            h1_ss, h3_ss, u1_ss, u3_ss = best_solution
            return (np.array([h1_ss, h2_ref, h3_ss]),
                    np.array([np.clip(u1_ss, 0, 1), np.clip(u3_ss, 0, 1)]))
        return None

    @classmethod
    def dynamics(cls, x, u):
        """Dinamica no-lineal del sistema."""
        h1, h2, h3 = np.clip(x, 0, 0.55)
        u1, u3     = np.clip(u, 0, 1)

        qi1 = cls.qmax * u1
        qi3 = cls.qmax * u3

        def q_out(h, a, A):
            return a * A * np.sqrt(2*cls.g*h) if h > 0.01 else 0.0

        q_o1 = q_out(h1, cls.alpha_o1, cls.A_o1)
        q_o2 = q_out(h2, cls.alpha_o2, cls.A_o2)
        q_o3 = q_out(h3, cls.alpha_o3, cls.A_o3)

        def q_coup(h_u, h_d, A, D, lc):
            dh = h_u - h_d
            if abs(dh) < 1e-8:
                return 0.0
            lam   = D * (cls.rho / cls.eta) * np.sqrt(2*cls.g*abs(dh))
            alpha = (1.0/np.e) * np.tanh(2 * lam / lc)
            return alpha * A * np.sqrt(2*cls.g*abs(dh)) * np.sign(dh)

        q_12 = q_coup(h1, h2, cls.A_12, cls.D_12, cls.lambda_c12)
        q_23 = q_coup(h2, h3, cls.A_23, cls.D_23, cls.lambda_c23)

        dh1 = (qi1 - q_12 - q_o1) / cls.Atank
        dh2 = (q_12 - q_23 - q_o2) / cls.Atank
        dh3 = (qi3 + q_23 - q_o3) / cls.Atank

        return np.array([dh1, dh2, dh3])

    @classmethod
    def integrate_step(cls, x, u, Ts):
        """Integracion RK45."""
        sol = solve_ivp(
            lambda t, y: cls.dynamics(y, u),
            [0, Ts], x,
            method='RK45', max_step=Ts/3
        )
        return np.clip(sol.y[:, -1], 0, 0.55)

    @classmethod
    def compute_jacobian(cls, x_ss, u_ss, h=1e-5):
        """Calcula matrices Jacobianas en tiempo continuo."""
        f0 = cls.dynamics(x_ss, u_ss)

        A = np.zeros((3, 3))
        for i in range(3):
            xp = x_ss.copy(); xp[i] += h
            A[:, i] = (cls.dynamics(xp, u_ss) - f0) / h

        B = np.zeros((3, 2))
        for i in range(2):
            up = u_ss.copy(); up[i] += h
            B[:, i] = (cls.dynamics(x_ss, up) - f0) / h

        return A, B


# ============================================================================
# CONTROLADOR 1: NMPC OPTIMIZADO
# ============================================================================

class NMPC_Optimized:
    """MPC No-Lineal con parametros optimizados."""

    def __init__(self, h2_ref, x_ss, u_ss,
                 N=10, w_h2=20000, w_u=0.1, w_du=1.0):
        self.N               = N
        self.h2_ref          = h2_ref
        self.Ts              = ThreeTankSystem.Ts
        self.x_ss            = x_ss
        self.u_ss            = u_ss
        self.u_last          = u_ss.copy()
        self.w_h2            = w_h2
        self.w_h1h3          = 5.0
        self.w_u             = w_u
        self.w_du            = w_du
        self.terminal_factor = 50

    def cost_function(self, u_seq, x0):
        """Funcion de costo del NMPC."""
        u_mat = u_seq.reshape(self.N, 2)
        x     = x0.copy()
        cost  = 0.0

        for k in range(self.N):
            u        = np.clip(u_mat[k], 0, 1)
            cost    += (x[1] - self.h2_ref)**2 * self.w_h2
            cost    += ((x[0]-self.x_ss[0])**2 + (x[2]-self.x_ss[2])**2) * self.w_h1h3
            cost    += (u[0]**2 + u[1]**2) * self.w_u
            du       = u - (self.u_last if k == 0 else u_mat[k-1])
            cost    += (du[0]**2 + du[1]**2) * self.w_du
            x        = ThreeTankSystem.integrate_step(x, u, self.Ts)

        cost += (x[1] - self.h2_ref)**2 * self.w_h2 * self.terminal_factor
        return cost

    def compute_control(self, x):
        """Resuelve el problema de optimizacion."""
        result    = minimize(
            lambda u: self.cost_function(u, x),
            np.tile(self.u_ss, self.N),
            method='L-BFGS-B',
            bounds=Bounds(lb=np.zeros(2*self.N), ub=np.ones(2*self.N)),
            options={'ftol': 1e-3, 'gtol': 1e-3, 'maxiter': 50, 'maxfun': 150}
        )
        u_current   = np.clip(result.x[:2], 0, 1.0)
        self.u_last = u_current.copy()
        return u_current

    def update_reference(self, h2_ref_new):
        """Actualiza referencia y punto de operacion."""
        if abs(h2_ref_new - self.h2_ref) > 1e-6:
            ss = ThreeTankSystem.compute_steady_state(h2_ref_new)
            if ss is not None:
                self.x_ss, self.u_ss = ss
                self.u_last = self.u_ss.copy()
            self.h2_ref = h2_ref_new


# ============================================================================
# CONTROLADOR 2: KOOPMAN-B — CORREGIDO [C1..C9]
# ============================================================================

class KoopmanMPC_SqrtObservables:
    """
    Koopman-MPC con observables fisicamente motivados.

    Mapa de lifting [C1] — solo sobre el estado h, SIN u:
        Psi(h) = [sqrt(h1), sqrt(h2), sqrt(h3),
                  h1*h2,    h2*h3,    h1*h3,    1]^T   in R^7

    Motivacion fisica:
      - sqrt(hi): captura la no-linealidad de Torricelli  (q_oi ~ sqrt(hi))
                  y hace la evolucion aproximadamente lineal: d(sqrt(hi))/dt ~ cte
      - hi*hj:    captura los acoplamientos entre tanques
      - 1:        bias para offset en estado estacionario

    Dinamica lineal en espacio Koopman [C2]:
        z_{k+1} = A * z_k + B * u_k     A in R^7x7, B in R^7x2
    u entra UNICAMENTE por B.

    Recuperacion de h2 desde el espacio lifteado [C6]:
        z[1] = sqrt(h2)   =>   h2 = z[1]^2

    Referencias:
      Proctor, S.L., Brunton, S.L., Kutz, J.N. (2018). Generalizing Koopman
      theory to allow inputs and control. SIAM J. Appl. Dyn. Syst., 17(1).
      Korda, M., Mezic, I. (2018). Linear predictors for nonlinear dynamical
      systems: Koopman operator meets model predictive control. Automatica, 93.
    """

    N_Z = 7   # dimension del espacio lifteado

    @staticmethod
    def observable_function(x):
        """
        Psi: R^3 -> R^7

        Entrada: x = [h1, h2, h3]
        Salida:  z = [sqrt(h1), sqrt(h2), sqrt(h3),
                      h1*h2,    h2*h3,    h1*h3,    1]

        NOTA: u NO forma parte de este mapa [C1].
        """
        h1, h2, h3 = x
        eps = 1e-8   # evitar sqrt de cero
        return np.array([
            np.sqrt(max(h1, eps)),   # z[0] = sqrt(h1)
            np.sqrt(max(h2, eps)),   # z[1] = sqrt(h2)  -> h2 = z[1]^2
            np.sqrt(max(h3, eps)),   # z[2] = sqrt(h3)
            h1 * h2,                 # z[3] = acoplamiento 1-2
            h2 * h3,                 # z[4] = acoplamiento 2-3
            h1 * h3,                 # z[5] = acoplamiento 1-3
            1.0                      # z[6] = bias
        ])

    @staticmethod
    def h2_from_z(z):
        """
        Recupera h2 desde el espacio lifteado [C6].
        z[1] = sqrt(h2)   =>   h2 = z[1]^2
        """
        return z[1] ** 2

    def __init__(self, A, B, h2_ref, x_ss, u_ss,
                 N=10, w_h2=50000, w_u=0.05, w_du=0.5):
        """
        Parametros:
          A, B    : matrices del modelo Koopman identificado
          h2_ref  : nivel de referencia para h2
          x_ss    : estado estacionario [h1_ss, h2_ref, h3_ss]   [C9]
          u_ss    : entrada estacionaria [u1_ss, u3_ss]
        """
        self.A               = A          # R^(7x7)
        self.B               = B          # R^(7x2)
        self.N               = N
        self.h2_ref          = h2_ref
        self.x_ss            = x_ss.copy()
        self.u_ss            = u_ss.copy()
        self.u_last          = u_ss.copy()
        self.w_h2            = w_h2
        self.w_u             = w_u
        self.w_du            = w_du
        self.terminal_factor = 50

        # z_ref = Psi(h_ref)  SIN u [C5]
        self.z_ref = self.observable_function(x_ss)

    def cost_function(self, u_seq, z0):
        """Funcion de costo en el espacio Koopman lifteado."""
        u_mat = u_seq.reshape(self.N, 2)
        z     = z0.copy()
        cost  = 0.0

        for k in range(self.N):
            u = np.clip(u_mat[k], 0, 1)

            # h2 se recupera como (sqrt(h2))^2 = z[1]^2  [C6]
            h2_pred = self.h2_from_z(z)
            cost   += (h2_pred - self.h2_ref)**2 * self.w_h2
            cost   += (u[0]**2 + u[1]**2) * self.w_u
            du      = u - (self.u_last if k == 0 else u_mat[k-1])
            cost   += (du[0]**2 + du[1]**2) * self.w_du

            # Dinamica lineal Koopman: z_{k+1} = A*z_k + B*u_k  [C2]
            z = self.A @ z + self.B @ u

        # Costo terminal
        cost += (self.h2_from_z(z) - self.h2_ref)**2 * self.w_h2 * self.terminal_factor
        return cost

    def compute_control(self, x):
        """Resuelve el MPC en espacio Koopman."""
        # z0 = Psi(x_k)  SIN u  [C4]
        z0 = self.observable_function(x)

        result = minimize(
            lambda u: self.cost_function(u, z0),
            np.tile(self.u_ss, self.N),
            method='L-BFGS-B',
            bounds=Bounds(lb=np.zeros(2*self.N), ub=np.ones(2*self.N)),
            options={'ftol': 1e-4, 'gtol': 1e-4, 'maxiter': 80, 'maxfun': 200}
        )

        u_current   = np.clip(result.x[:2], 0, 1.0)
        self.u_last = u_current.copy()
        return u_current

    def update_reference(self, h2_ref_new):
        """Actualiza referencia. z_ref = Psi(h_ref) sin u [C5]."""
        if abs(h2_ref_new - self.h2_ref) > 1e-6:
            ss = ThreeTankSystem.compute_steady_state(h2_ref_new)
            if ss is not None:
                self.x_ss, self.u_ss = ss
                self.u_last = self.u_ss.copy()
                # z_ref = Psi(x_ss_new)  SIN u  [C5]
                self.z_ref = self.observable_function(self.x_ss)
            self.h2_ref = h2_ref_new


# ============================================================================
# CONTROLADOR 3: LINEAL GAIN-SCHEDULING
# ============================================================================

class LinearGainScheduling:
    """Controlador lineal con multiples modelos LQR interpolados."""

    def __init__(self, h2_refs=None):
        self.h2_refs = h2_refs or [0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36]
        self.models  = {}
        self.Ts      = ThreeTankSystem.Ts
        self.u_prev  = None

        print("\n  Calculando modelos Lineal GS...")

        for h2_ref in self.h2_refs:
            ss = ThreeTankSystem.compute_steady_state(h2_ref)
            if ss is None:
                continue
            x_ss, u_ss = ss
            A_c, B_c   = ThreeTankSystem.compute_jacobian(x_ss, u_ss)

            Q = np.diag([5, 2000, 5])
            R = np.diag([0.05, 0.05])

            try:
                P = solve_continuous_are(A_c, B_c, Q, R)
                K = np.linalg.inv(R) @ B_c.T @ P
                self.models[h2_ref] = {'x_ss': x_ss, 'u_ss': u_ss, 'K': K}
            except:
                pass

        print(f"  Modelos exitosos: {len(self.models)}/{len(self.h2_refs)}")

    def select_model(self, h2_current):
        """Interpolacion lineal entre los dos modelos mas cercanos."""
        distances   = {h2: abs(h2 - h2_current) for h2 in self.models}
        sorted_refs = sorted(distances, key=lambda k: distances[k])

        h2_1 = sorted_refs[0]
        h2_2 = sorted_refs[1] if len(sorted_refs) > 1 else h2_1

        d1, d2 = distances[h2_1], distances[h2_2]
        if d1 < 1e-6:
            return self.models[h2_1]

        w1 = d2 / (d1 + d2)
        m1, m2 = self.models[h2_1], self.models[h2_2]
        return {
            'x_ss': w1*m1['x_ss'] + (1-w1)*m2['x_ss'],
            'u_ss': w1*m1['u_ss'] + (1-w1)*m2['u_ss'],
            'K':    w1*m1['K']    + (1-w1)*m2['K'],
        }

    def compute_control(self, x, h2_ref_current):
        """Calcula la accion de control lineal."""
        m  = self.select_model(h2_ref_current)
        u  = m['u_ss'] - m['K'] @ (x - m['x_ss'])
        if self.u_prev is not None:
            u = 0.7*self.u_prev + 0.3*u
        u           = np.clip(u, 0, 1.0)
        self.u_prev = u.copy()
        return u


# ============================================================================
# IDENTIFICACION EDMD — CORREGIDA [C3]
# ============================================================================

def identify_koopman_sqrt():
    """
    Identificacion del operador Koopman con observables sqrt(h).

    Regresion EDMD [C3]:
        Psi(x_{k+1}) ~= A * Psi(x_k) + B * u_k
        A in R^(7x7),  B in R^(7x2)

    Formulacion de la regresion:
        Phi = [Psi(x_k) | u_k]   shape (M, 9)   <- u separado, NO dentro de Psi
        Y   = Psi(x_{k+1})        shape (M, 7)
        Theta = (Phi^T Phi + lambda I)^{-1} Phi^T Y   shape (9, 7)
        A = Theta[:7, :].T        B = Theta[7:, :].T

    Dataset:
      - 50 trayectorias
      - 10 referencias (0.12 a 0.35 m)
      - ~8300 muestras
    """

    print("\n" + "="*80)
    print("IDENTIFICACION KOOPMAN CORREGIDA")
    print("Psi(h) = [sqrt(h1), sqrt(h2), sqrt(h3), h1*h2, h2*h3, h1*h3, 1]  in R^7")
    print("u entra SEPARADO en la regresion EDMD (no dentro de Psi)")
    print("="*80)

    obs_fn = KoopmanMPC_SqrtObservables.observable_function

    z_data      = []
    u_data      = []
    z_next_data = []

    Ts      = ThreeTankSystem.Ts
    t_arr   = np.arange(0, 50, Ts)
    h2_refs = [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]

    print(f"  Referencias: {h2_refs}")
    print(f"  Muestras esperadas: ~{len(h2_refs) * 5 * len(t_arr)}\n")

    traj_count = 0

    for h2_ref in h2_refs:
        for pert_factor in [0.95, 0.90, 0.98, 0.85, 0.92]:

            ss = ThreeTankSystem.compute_steady_state(h2_ref)
            if ss is None:
                continue
            x_ss, u_ss = ss

            x    = x_ss.copy()
            x[1] = h2_ref * pert_factor

            for _ in range(len(t_arr) - 1):
                error = x[1] - h2_ref
                u     = u_ss.copy()
                u[0] += -0.3 * error + np.random.randn() * 0.03
                u[1] += -0.2 * error + np.random.randn() * 0.02
                u     = np.clip(u, 0, 1)

                # z = Psi(x_k)  SIN u [C3]
                z      = obs_fn(x)
                x_next = ThreeTankSystem.integrate_step(x, u, Ts)
                z_next = obs_fn(x_next)

                z_data.append(z)
                u_data.append(u)
                z_next_data.append(z_next)

                x = x_next

            traj_count += 1
            if traj_count % 10 == 0:
                print(f"  {traj_count} trayectorias completadas...")

    z_data      = np.array(z_data)        # (M, 7)
    u_data      = np.array(u_data)        # (M, 2)
    z_next_data = np.array(z_next_data)   # (M, 7)

    n_z        = z_data.shape[1]    # 7
    n_u        = u_data.shape[1]    # 2
    lambda_reg = 1e-4

    # Phi = [Psi(x_k) | u_k]  shape (M, 9) — u NO esta dentro de Psi [C3]
    Phi   = np.hstack([z_data, u_data])   # (M, 9)
    Y     = z_next_data                   # (M, 7)

    Theta = np.linalg.solve(
        Phi.T @ Phi + lambda_reg * np.eye(n_z + n_u),
        Phi.T @ Y
    )                                     # (9, 7)

    A = Theta[:n_z, :].T   # R^(7x7)
    B = Theta[n_z:,  :].T  # R^(7x2)

    # Validacion
    z_pred = z_data @ A.T + u_data @ B.T
    rmse   = np.sqrt(np.mean((z_pred - z_next_data)**2))

    print(f"\n  Dataset final: {traj_count} trayectorias, {len(z_data)} muestras")
    print(f"  Operador identificado: A={A.shape}, B={B.shape}, RMSE={rmse:.6f} m")
    print("="*80 + "\n")

    return A, B


# ============================================================================
# PERFILES DE REFERENCIA
# ============================================================================

def get_reference_profile(case):
    """Define perfiles de referencia para cada caso de prueba."""

    if case == 1:
        return lambda t: 0.25, 60, "Regulation h2=0.25m"

    elif case == 2:
        def h2_ref_func(t):
            if   t < 60:  return 0.15
            elif t < 120: return 0.20
            elif t < 180: return 0.25
            else:         return 0.30
        return h2_ref_func, 240, "Multi-step Monotonic"

    elif case == 3:
        return lambda t: 0.25, 120, "Disturbance Rejection"

    else:
        raise ValueError(f"Caso {case} no definido")


# ============================================================================
# SIMULACION GENERICA
# ============================================================================

def simulate_controller(controller, controller_name, h2_ref_func,
                        sim_time, case_num, apply_disturbance=False):
    """Simula un controlador y registra metricas incluyendo ms/paso [C8]."""

    Ts    = ThreeTankSystem.Ts
    t_arr = np.arange(0, sim_time, Ts)

    h2_ref_init = h2_ref_func(0) if callable(h2_ref_func) else h2_ref_func
    ss = ThreeTankSystem.compute_steady_state(h2_ref_init)
    if ss is None:
        print(f"  ERROR: No se encontro SS inicial para {controller_name}")
        return None

    x_ss, u_ss = ss
    x_init     = x_ss.copy()
    x_init[1]  = h2_ref_init * 0.95

    x_traj      = np.zeros((len(t_arr), 3))
    u_traj      = np.zeros((len(t_arr), 2))
    h2_ref_traj = np.zeros(len(t_arr))
    step_times  = []   # tiempo por paso de control [C8]

    x         = x_init.copy()
    x_traj[0] = x

    print(f"\n  {'='*76}")
    print(f"  {controller_name:^76}")
    print(f"  {'='*76}")

    total_start = time.time()

    for k in range(len(t_arr) - 1):

        h2_ref_current = h2_ref_func(t_arr[k]) if callable(h2_ref_func) else h2_ref_func
        h2_ref_traj[k] = h2_ref_current

        if hasattr(controller, 'update_reference'):
            controller.update_reference(h2_ref_current)

        # Medir tiempo por paso [C8]
        t0 = time.time()
        if isinstance(controller, LinearGainScheduling):
            u = controller.compute_control(x, h2_ref_current)
        else:
            u = controller.compute_control(x)
        step_times.append(time.time() - t0)

        u_traj[k] = u

        # Perturbacion (Caso 3)
        if apply_disturbance and case_num == 3 and 40 <= t_arr[k] < 40.3:
            x[0] += 0.08
            print(f"  t={t_arr[k]:.1f}s | PERTURBACION: Delta_h1=+8cm aplicada")

        x          = ThreeTankSystem.integrate_step(x, u, Ts)
        x_traj[k+1] = x

        if k % 100 == 0 or (apply_disturbance and 39 < t_arr[k] < 41):
            err = abs(x[1] - h2_ref_current) * 100
            print(f"  t={t_arr[k]:6.1f}s | h2={x[1]:.4f}m | "
                  f"ref={h2_ref_current:.4f}m | err={err:5.2f}cm")

    total_elapsed = time.time() - total_start
    avg_step_ms   = np.mean(step_times) * 1000   # ms por paso [C8]

    errors    = np.abs(x_traj[:, 1] - h2_ref_traj) * 100
    error_ss  = np.mean(errors[-50:])

    # Metricas especificas Caso 3
    settling_time     = None
    overshoot_percent = None

    if apply_disturbance and case_num == 3:
        idx_d     = int(40 / Ts)
        threshold = h2_ref_traj[idx_d] * 0.02
        settled   = np.where(errors[idx_d:] < threshold)[0]
        if len(settled) > 0:
            settling_time = t_arr[idx_d + settled[0]] - 40
        overshoot         = np.max(errors[idx_d:idx_d+100]) - errors[max(idx_d-1, 0)]
        overshoot_percent = overshoot / h2_ref_traj[idx_d] * 100

    print(f"  Tiempo total: {total_elapsed:.1f}s | "
          f"Promedio por paso: {avg_step_ms:.1f}ms | "
          f"Error medio: {np.mean(errors):.2f}cm | "
          f"Error SS: {error_ss:.2f}cm")
    if settling_time is not None:
        print(f"  Asentamiento: {settling_time:.1f}s | "
              f"Overshoot: {overshoot_percent:.1f}%")

    return {
        't':                t_arr,
        'x':                x_traj,
        'u':                u_traj,
        'h2_ref':           h2_ref_traj,
        'errors':           errors,
        'error_mean':       np.mean(errors),
        'error_max':        np.max(errors),
        'error_ss':         error_ss,
        'settling_time':    settling_time,
        'overshoot_percent':overshoot_percent,
        'computation_time': total_elapsed,
        'avg_step_ms':      avg_step_ms,       # [C8]
        'controller_name':  controller_name,
        'case_num':         case_num,
    }


# ============================================================================
# VISUALIZACION
# ============================================================================

def plot_case_comparison(case_results, case_num, case_name):
    """Genera graficas de comparacion para un caso especifico."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    colors = {'NMPC': '#2E86AB', 'Koopman-B': '#A23B72', 'Linear': '#F18F01'}
    styles = {'NMPC': '-',       'Koopman-B': '--',       'Linear': '-.'}
    LW, LW_REF, LW_THR = 4.0, 3.5, 2.5

    # Panel 1: Tracking
    for name, data in case_results.items():
        axes[0].plot(data['t'], data['x'][:, 1],
                     color=colors[name], linestyle=styles[name],
                     linewidth=LW, label=name, alpha=0.9)
    ref = list(case_results.values())[0]
    axes[0].plot(ref['t'], ref['h2_ref'],
                 'k:', linewidth=LW_REF, label='Reference', alpha=0.7)
    axes[0].set_ylabel('Level h2 [m]', fontsize=13, fontweight='bold')
    axes[0].set_title(f'Case {case_num}: {case_name} - Tracking',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Error de seguimiento
    for name, data in case_results.items():
        axes[1].plot(data['t'], data['errors'],
                     color=colors[name], linestyle=styles[name],
                     linewidth=LW, label=name, alpha=0.85)
    axes[1].axhline(y=2, color='green', linestyle='--', alpha=0.6,
                    linewidth=LW_THR, label='2 cm threshold')
    axes[1].set_ylabel('Absolute error [cm]', fontsize=13, fontweight='bold')
    axes[1].set_title('Tracking Error', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')

    # Panel 3: Señal de control u1
    for name, data in case_results.items():
        n = min(len(data['t']), len(data['u']) + 1)
        axes[2].plot(data['t'][:n-1], data['u'][:n-1, 0],
                     color=colors[name], linestyle=styles[name],
                     linewidth=LW, label=name, alpha=0.85)
    axes[2].set_xlabel('Time [s]', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Control action u1 (Pump 1)', fontsize=13, fontweight='bold')
    axes[2].set_title('Control Signal u1', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_ylim([-0.05, 1.05])

    plt.tight_layout()

    safe_name = case_name.lower().replace(' ','_').replace('=','').replace('2','2')
    fname     = output_dir / f'case{case_num}_{safe_name}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\n  Grafica guardada: {fname.name}")
    plt.close()
    return fname


def plot_global_comparison(all_results):
    """Genera grafica comparativa global con tiempo en ms/paso [C8]."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    controllers = ['NMPC', 'Koopman-B', 'Linear']
    colors      = {'NMPC': '#2E86AB', 'Koopman-B': '#A23B72', 'Linear': '#F18F01'}
    metrics     = {c: {'error_mean': [], 'error_ss': [], 'ms_step': []}
                   for c in controllers}

    for cn in [1, 2, 3]:
        ck = f'Caso_{cn}'
        if ck in all_results:
            for c in controllers:
                if c in all_results[ck]:
                    d = all_results[ck][c]
                    metrics[c]['error_mean'].append(d['error_mean'])
                    metrics[c]['error_ss'].append(d['error_ss'])
                    metrics[c]['ms_step'].append(d['avg_step_ms'])   # [C8]

    x, w = np.arange(3), 0.25
    cfg  = [
        ('error_mean', 'Average Error [cm]',    'Average Error per Case'),
        ('error_ss',   'SS Error [cm]',          'Steady-State Error'),
        ('ms_step',    'Time per step [ms]',     'Avg Compute Time per Step (ms, log)'),
    ]

    for ax, (key, ylabel, title) in zip(axes, cfg):
        for i, c in enumerate(controllers):
            ax.bar(x + (i-1)*w, metrics[c][key], w,
                   label=c, color=colors[c], alpha=0.8)
        ax.set_xlabel('Case', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Case 1', 'Case 2', 'Case 3'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3, which='both', axis='y')

    plt.tight_layout()
    fname = output_dir / 'global_comparison.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\n  Grafica global guardada: {fname.name}")
    plt.close()
    return fname


# ============================================================================
# TABLA DE RESULTADOS
# ============================================================================

def generate_results_table(all_results):
    """Genera tabla de resultados con speedup por paso [C8]."""

    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(f"{'Case':<6} {'Controller':<13} {'Avg Err':>9} {'SS Err':>8} "
          f"{'Total(s)':>10} {'ms/step':>9} {'Speedup':>10}")
    print("-"*80)

    latex = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Quantitative Comparison of Control Strategies Across All Scenarios.}",
        "\\label{tab:results}",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        ("\\multirow{2}{*}{Case} & \\multirow{2}{*}{Method} & "
         "\\multirow{2}{*}{\\shortstack{Mean Error\\\\(cm)}} & "
         "\\multirow{2}{*}{\\shortstack{Steady-State\\\\Error (cm)}} & "
         "\\multirow{2}{*}{\\shortstack{Total Time\\\\(s)}} & "
         "\\multirow{2}{*}{\\shortstack{ms/step}} & "
         "\\multirow{2}{*}{\\shortstack{Speedup\\\\($\\times$)}} \\\\"),
        "\\\\",
        "\\midrule",
    ]

    speedups_step = {}

    for cn in [1, 2, 3]:
        ck = f'Caso_{cn}'
        if ck not in all_results:
            continue
        cd = all_results[ck]

        # Speedup por paso entre NMPC y Koopman-B [C8]
        if 'NMPC' in cd and 'Koopman-B' in cd:
            speedups_step[cn] = cd['NMPC']['avg_step_ms'] / cd['Koopman-B']['avg_step_ms']

        for ctrl, data in cd.items():
            if ctrl == 'NMPC':
                sp_str = "1.0 (base)"
            elif ctrl == 'Koopman-B' and cn in speedups_step:
                sp_str = f"{speedups_step[cn]:.1f}x"
            else:
                sp_str = "---"

            print(f"{cn:<6} {ctrl:<13} "
                  f"{data['error_mean']:>7.2f} cm "
                  f"{data['error_ss']:>6.2f} cm "
                  f"{data['computation_time']:>10.1f}s "
                  f"{data['avg_step_ms']:>8.1f}ms "
                  f"{sp_str:>10}")

            latex.append(
                f"{cn} & {ctrl} & {data['error_mean']:.2f} & "
                f"{data['error_ss']:.2f} & "
                f"{data['computation_time']:.1f} & "
                f"{data['avg_step_ms']:.1f} & {sp_str} \\\\"
            )

        if cn < 3:
            latex.append("\\hline")

    print("="*80)

    if speedups_step:
        avg_sp = np.mean(list(speedups_step.values()))
        print(f"\n  Speedup promedio Koopman-B vs NMPC (por paso): {avg_sp:.1f}x")

    latex += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    ltx_path = output_dir / 'results_table.tex'
    ltx_path.write_text('\n'.join(latex))
    print(f"  Tabla LaTeX guardada: {ltx_path.name}")

    return ltx_path


# ============================================================================
# EJECUCION PRINCIPAL
# ============================================================================

def main():
    """Ejecuta la comparacion completa de los 3 casos."""

    print("\n" + "="*80)
    print("  CODIT 2026 — NMPC vs Koopman-B sqrt(h) vs Linear GS")
    print("="*80)

    # ------------------------------------------------------------------
    # PASO 1: Identificar operador Koopman (CORREGIDO)
    # ------------------------------------------------------------------
    print("\n--- PASO 1: IDENTIFICACION KOOPMAN ---")
    A_koopman, B_koopman = identify_koopman_sqrt()

    # ------------------------------------------------------------------
    # PASO 2: Inicializar controlador lineal
    # ------------------------------------------------------------------
    print("\n--- PASO 2: INICIALIZACION LINEAL GS ---")
    controller_linear = LinearGainScheduling()

    all_results = {}

    # ------------------------------------------------------------------
    # PASO 3: Simulaciones por caso
    # ------------------------------------------------------------------
    for case_num in [1, 2, 3]:

        print(f"\n{'='*80}")
        print(f"  CASO {case_num}")
        print(f"{'='*80}")

        h2_ref_func, sim_time, case_name = get_reference_profile(case_num)
        apply_dist = (case_num == 3)

        h2_ref_init = h2_ref_func(0) if callable(h2_ref_func) else h2_ref_func
        ss = ThreeTankSystem.compute_steady_state(h2_ref_init)
        if ss is None:
            print(f"  ERROR: No se encontro SS para Caso {case_num}")
            continue
        x_ss, u_ss = ss

        case_results = {}

        # NMPC
        ctrl_nmpc = NMPC_Optimized(h2_ref_init, x_ss, u_ss)
        r = simulate_controller(ctrl_nmpc, "NMPC",
                                h2_ref_func, sim_time, case_num, apply_dist)
        if r: case_results['NMPC'] = r

        # Koopman-B — [C9] recibe x_ss, calcula z_ref internamente
        ctrl_koop = KoopmanMPC_SqrtObservables(
            A_koopman, B_koopman,
            h2_ref_init, x_ss, u_ss,   # x_ss no z_ss [C9]
            w_h2=50000
        )
        r = simulate_controller(ctrl_koop, "Koopman-B",
                                h2_ref_func, sim_time, case_num, apply_dist)
        if r: case_results['Koopman-B'] = r

        # Lineal GS
        r = simulate_controller(controller_linear, "Linear",
                                h2_ref_func, sim_time, case_num, apply_dist)
        if r: case_results['Linear'] = r

        all_results[f'Caso_{case_num}'] = case_results
        plot_case_comparison(case_results, case_num, case_name)

    # ------------------------------------------------------------------
    # PASO 4: Analisis global
    # ------------------------------------------------------------------
    print("\n--- PASO 4: ANALISIS GLOBAL ---")

    with open(output_dir / 'final_results_complete.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("  Resultados guardados: final_results_complete.pkl")

    plot_global_comparison(all_results)
    generate_results_table(all_results)

    # ------------------------------------------------------------------
    # PASO 5: Resumen final
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("  RESUMEN FINAL")
    print("="*80)

    for ctrl in ['NMPC', 'Koopman-B', 'Linear']:
        errs = [all_results[f'Caso_{c}'][ctrl]['error_mean']
                for c in [1,2,3]
                if ctrl in all_results.get(f'Caso_{c}', {})]
        ms   = [all_results[f'Caso_{c}'][ctrl]['avg_step_ms']
                for c in [1,2,3]
                if ctrl in all_results.get(f'Caso_{c}', {})]
        if errs:
            print(f"  {ctrl:<13}: error_medio={np.mean(errs):.2f}cm  "
                  f"tiempo_paso={np.mean(ms):.1f}ms")

    if ('Caso_1' in all_results and 'NMPC' in all_results['Caso_1']
            and 'Koopman-B' in all_results['Caso_1']):
        nmpc_ms  = np.mean([all_results[f'Caso_{c}']['NMPC']['avg_step_ms']
                            for c in [1,2,3]
                            if 'NMPC' in all_results.get(f'Caso_{c}', {})])
        koop_ms  = np.mean([all_results[f'Caso_{c}']['Koopman-B']['avg_step_ms']
                            for c in [1,2,3]
                            if 'Koopman-B' in all_results.get(f'Caso_{c}', {})])
        print(f"\n  Speedup promedio (por paso): {nmpc_ms/koop_ms:.1f}x")

    print("\n  COMPARACION COMPLETADA")
    return all_results


if __name__ == "__main__":
    results = main()