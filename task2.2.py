#!/usr/bin/env python3
"""
Parafoil 6-DOF dynamics + autonomous homing guidance
=====================================================

Implements the rigid 6-DOF parafoil model of:
  Zhao, Tao, Sun & Sun, "Dynamic modelling of parafoil system based on
  aerodynamic coefficients identification", Automatika 64:2, 291-303 (2023).

Equations used (paper eq. numbers in comments):
  - Coordinate transform Bd-s                         (1)
  - Translational kinematics  x. = Bd-s^T * Vc         (2)
  - Rotational kinematics     Euler rates from p,q,r   (3)
  - Force equation            m(Vc. + W x Vc)=FW+FA    (4),(11)
  - Moment equation           IT*W. + W x IT*W = MA    (5),(12)
  - FW, FA, MA, SW, IT                                 (6)-(10)

The aerodynamic-coefficient VALUES used below are taken directly from the
paper's Table 1 (roll/yaw, identified by RWLS) and Table 3 (lift/drag/pitch).
The paper does not publish numeric moments of inertia (IXX, IYY, IZZ, IXZ),
so representative values for a small ram-air canopy + payload of this size
are assumed (clearly marked ASSUMED below) -- everything else is exactly the
published model.

On top of the dynamics, a simple "energy management + proportional homing"
guidance law steers the asymmetric control deflection delta_a:
  1. SPIRAL phase: if the vehicle is too high for the remaining distance to
     the target (i.e. it can't glide there at the nominal L/D), hold a
     constant-deflection turn to bleed off altitude (like a real parafoil's
     autonomous "energy management" holding pattern).
  2. HOMING phase: once inside the reachable glide cone, point the nose at
     the target using proportional heading control on delta_a.
This is deliberately simple (a real system would add wind estimation, a
final into-wind flare leg, etc.) but is a complete, working closed loop.

Bonus: `lla_to_local_xyz` converts the lat/lon/altitude stream produced by
the Question-1 ground station into the local x (north), y (east), z (down)
frame this guidance algorithm runs in.

Run:  python3 parafoil_guidance.py
Produces parafoil_path.png (3D path + target) in the working directory.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)

# --------------------------------------------------------------------------- #
# 0. Bonus: lat/lon/alt (Q1 telemetry) -> local x,y,z
# --------------------------------------------------------------------------- #

EARTH_R = 6_371_000.0  # m


def lla_to_local_xyz(lat_deg, lon_deg, alt_m, lat0_deg, lon0_deg, alt0_m=0.0):
    """Equirectangular projection, accurate to a few metres over a few km
    (plenty for a CanSat/parafoil drop radius). Returns (x=north, y=east,
    z=down) relative to the (lat0, lon0, alt0) origin -- i.e. the exact
    frame TelemetryPacket() -> guidance model below expects.

    lat_deg, lon_deg, alt_m may be scalars or numpy arrays (vectorised),
    which lets you feed it straight from the Q1 GCS log/CSV.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    x = EARTH_R * (lat - lat0)                     # north
    y = EARTH_R * (lon - lon0) * math.cos(lat0)    # east
    z = -(np.asarray(alt_m) - alt0_m)              # down (NED)
    return x, y, z


# --------------------------------------------------------------------------- #
# 1. Model parameters (paper's Tables 1-3, plus geometry Table 2)
# --------------------------------------------------------------------------- #

# --- geometry (Table 2) ---
m = 0.5 + 4.0          # kg, canopy + payload mass
b = 3.0                 # m, canopy span
c = 1.0                 # m, canopy chord
d = 0.1                 # m, control line length that can be pulled down
S = b * c               # m^2, forced (reference) area

# --- environment ---
rho = 1.225             # kg/m^3, air density
g = 9.81                # m/s^2

# --- aerodynamic coefficients: lift/drag/pitch (Table 3) ---
CL0, CLa, CLda = 0.5, 1.7190, 0.0001
CD0, CDa2, CDda = 0.2, 0.7, 0.0001
Cm0, Cma, Cmq = 0.1397, -1.4308, -0.2251

# --- aerodynamic coefficients: roll/yaw, identified by RWLS (Table 1) ---
Clphi, Clp, Clda = -0.04, -0.08, -0.00001
Cnr, Cnda = -0.012, -0.00008
delta_bias = -0.0003

# --- moments of inertia: NOT published in the paper -> ASSUMED, based on
#     typical small-parafoil/CanSat-payload scale (order-of-magnitude only).
IXX, IYY, IZZ, IXZ = 1.6, 0.5, 1.7, 0.05     # kg*m^2  (ASSUMED)
IT = np.array([[IXX, 0.0, IXZ],
               [0.0, IYY, 0.0],
               [IXZ, 0.0, IZZ]])
IT_inv = np.linalg.inv(IT)


def Bd_s(phi, theta, psi):
    """Eq. (1): inertial -> body transform, rotation order Z-Y-X."""
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    return np.array([
        [cth * cpsi,                 cth * spsi,                 -sth],
        [sphi * sth * cpsi - cphi * spsi, sphi * sth * spsi + cphi * cpsi, sphi * cth],
        [cphi * sth * cpsi + sphi * spsi, cphi * sth * spsi - sphi * cpsi, cphi * cth],
    ])


def euler_rate_matrix(phi, theta):
    """Eq. (3): [p,q,r] -> [phi.,theta.,psi.]."""
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, tth = math.cos(theta), math.tan(theta)
    return np.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi,       -sphi],
        [0.0, sphi / cth, cphi / cth],
    ])


def skew(w):
    """Eq. (9): SW, the skew-symmetric cross-product operator of W=[p,q,r]."""
    p, q, r = w
    return np.array([[0.0, -r, q],
                      [r, 0.0, -p],
                      [-q, p, 0.0]])


# --------------------------------------------------------------------------- #
# 2. Full nonlinear 6-DOF derivative (eqs. 4-12)
# --------------------------------------------------------------------------- #

def dynamics(state, delta_a):
    """state = [xn, yn, zn, u, v, w, phi, theta, psi, p, q, r]
    xn,yn,zn : inertial NED position (m), zn negative-up (so altitude=-zn)
    u,v,w    : body-frame velocity (m/s)
    phi,theta,psi : Euler angles (rad)
    p,q,r    : body-frame angular rates (rad/s)
    delta_a  : commanded asymmetric brake deflection, in [-1, 1]
    """
    xn, yn, zn, u, v, w, phi, theta, psi, p, q, r = state
    Vc = np.array([u, v, w])
    Wv = np.array([p, q, r])
    Vmag = max(np.linalg.norm(Vc), 1e-3)      # guard against div-by-0
    alpha = math.atan2(w, u) if abs(u) > 1e-6 else 0.0

    Bds = Bd_s(phi, theta, psi)

    # --- forces, eq. (6)-(7) ---
    FW = Bds @ np.array([0.0, 0.0, m * g])
    CL = CL0 + CLa * alpha + CLda * delta_a
    CD = CD0 + CDa2 * alpha ** 2 + CDda * delta_a
    FA = 0.5 * rho * S * Vmag * CL * np.array([w, 0.0, -u]) \
        - 0.5 * rho * S * Vmag * CD * np.array([u, v, w])

    Vc_dot = (FW + FA) / m - skew(Wv) @ Vc

    # --- moments, eq. (8) ---
    Mx = Clphi * b * phi + Clp * b ** 2 * p / (2 * Vmag) + Clda * delta_a * b / d
    My = Cm0 * c + Cma * c * alpha + Cmq * c ** 2 * q / (2 * Vmag)
    Mz = Cnr * b ** 2 * r / (2 * Vmag) + Cnda * delta_a * b / d + delta_bias
    MA = 0.5 * rho * S * Vmag ** 2 * np.array([Mx, My, Mz])

    W_dot = IT_inv @ (MA - skew(Wv) @ (IT @ Wv))

    # --- kinematics, eq. (2)-(3) ---
    pos_dot = Bds.T @ Vc
    euler_dot = euler_rate_matrix(phi, theta) @ Wv

    return np.concatenate([pos_dot, Vc_dot, euler_dot, W_dot])


def rk4_step(state, delta_a, dt):
    k1 = dynamics(state, delta_a)
    k2 = dynamics(state + 0.5 * dt * k1, delta_a)
    k3 = dynamics(state + 0.5 * dt * k2, delta_a)
    k4 = dynamics(state + dt * k3, delta_a)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------- #
# 3. Guidance: energy management + proportional homing on heading
# --------------------------------------------------------------------------- #

class Guidance:
    """
    glide_ratio  : nominal forward/descent ratio in a straight glide
                   (~3.25 in the paper's own no-wind simulation, Sec 4.1.1)
    safety_margin: shrink the usable glide cone a bit so the vehicle doesn't
                   arrive exactly at zero altitude with zero margin
    kp           : proportional gain on heading error -> delta_a
    da_max       : saturation on the control deflection
    spiral_da    : constant deflection commanded while bleeding altitude
    """

    def __init__(self, target_xy=(0.0, 0.0), glide_ratio=3.2,
                 safety_margin=0.85, kp=0.6, da_max=1.0, spiral_da=0.8,
                 capture_radius=8.0):
        self.tx, self.ty = target_xy
        self.glide_ratio = glide_ratio
        self.safety_margin = safety_margin
        self.kp = kp
        self.da_max = da_max
        self.spiral_da = spiral_da
        self.capture_radius = capture_radius
        self.mode = "SPIRAL"

    @staticmethod
    def _wrap_pi(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def command(self, xn, yn, zn, psi):
        altitude = -zn
        dist = math.hypot(self.tx - xn, self.ty - yn)

        # usable straight-line glide distance from current altitude
        reachable = self.glide_ratio * self.safety_margin * altitude

        if dist < self.capture_radius:
            self.mode = "CAPTURE"
            return 0.0

        if dist > reachable and altitude > 15.0:
            # too high / too close: spend altitude in a holding turn
            self.mode = "SPIRAL"
            return self.spiral_da

        # inside the reachable cone: fly straight at the target
        self.mode = "HOMING"
        psi_des = math.atan2(self.ty - yn, self.tx - xn)
        err = self._wrap_pi(psi_des - psi)
        # NOTE ON SIGN: with Cnda < 0 in Table 1, a *positive* delta_a
        # produces a *negative* yaw-rate contribution (see eq. 8), so the
        # feedback gain is applied with a negative sign to get negative
        # feedback (i.e. delta_a that actually reduces the heading error).
        da = -self.kp * err
        return float(np.clip(da, -self.da_max, self.da_max))


# --------------------------------------------------------------------------- #
# 4. Simulation
# --------------------------------------------------------------------------- #

def simulate(seed=None, x0=None, y0=None, z0=700.0, t_max=400.0, dt=0.02):
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = rng.uniform(-400.0, 400.0)
    if y0 is None:
        y0 = rng.uniform(-400.0, 400.0)

    # start in a trimmed, steady glide (matches paper Sec. 4.1 initial cond.)
    state = np.array([x0, y0, -z0,      # xn, yn, zn (zn negative-up)
                       6.0, 0.0, 3.0,   # u, v, w
                       0.0, 0.0,        # phi, theta
                       math.atan2(-y0, -x0),  # psi: start roughly facing target
                       0.0, 0.0, 0.0])  # p, q, r

    guide = Guidance(target_xy=(0.0, 0.0))
    log = {"t": [], "x": [], "y": [], "z": [], "mode": [], "delta_a": []}

    t = 0.0
    while t < t_max:
        altitude = -state[2]
        if altitude <= 0.0:
            break
        da = guide.command(state[0], state[1], state[2], state[8])
        log["t"].append(t)
        log["x"].append(state[0])
        log["y"].append(state[1])
        log["z"].append(altitude)
        log["mode"].append(guide.mode)
        log["delta_a"].append(da)

        state = rk4_step(state, da, dt)
        t += dt

    for k in ("t", "x", "y", "z", "delta_a"):
        log[k] = np.array(log[k])
    return log


# --------------------------------------------------------------------------- #
# 5. Plot
# --------------------------------------------------------------------------- #

def plot_result(log, save_path="parafoil_path.png"):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"SPIRAL": "#ff9f1c", "HOMING": "#2ec4b6", "CAPTURE": "#e71d36"}
    modes = np.array(log["mode"])
    for phase in ("SPIRAL", "HOMING", "CAPTURE"):
        sel = modes == phase
        if sel.any():
            ax.plot(log["x"][sel], log["y"][sel], log["z"][sel],
                    ".", ms=2.5, color=colors[phase], label=phase)

    ax.plot(log["x"], log["y"], log["z"], color="grey", lw=0.6, alpha=0.6)
    ax.scatter([log["x"][0]], [log["y"][0]], [log["z"][0]],
               c="black", marker="^", s=70, label="spawn")
    ax.scatter([0], [0], [0], c="lime", marker="*", s=200,
               edgecolor="black", label="target (0,0,0)")

    ax.set_xlabel("x - north (m)")
    ax.set_ylabel("y - east (m)")
    ax.set_zlabel("altitude (m)")
    ax.set_title("Autonomous parafoil homing: energy-management + proportional guidance")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"saved {save_path}")

    miss = math.hypot(log["x"][-1], log["y"][-1])
    print(f"landed at x={log['x'][-1]:.1f} m, y={log['y'][-1]:.1f} m, "
          f"altitude={log['z'][-1]:.1f} m  -> miss distance {miss:.1f} m, "
          f"flight time {log['t'][-1]:.1f} s")


if __name__ == "__main__":
    result = simulate(seed=1)
    plot_result(result)