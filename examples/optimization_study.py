# This file is part of the Centrifugal Compressor Model
# Copyright (c) 2024 G.F. Frate
# Licensed under the MIT License

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from CoolProp.CoolProp import PropsSI
from compressor_model import CentrifugalCompressor


def run_optimized_simulations():
    # -----------------------------
    # Define fluid and inlet state
    # -----------------------------
    fluid = 'Air'
    p_01 = 1e5  # Inlet total pressure in Pa
    T_01 = 100 + 273.15  # Inlet total temperature in Kelvin
    rho_01 = PropsSI("DMASS", "T", T_01, "P", p_01, fluid)  # Inlet density
    a_01 = PropsSI("A", "T", T_01, "P", p_01, fluid)  # Speed of sound at inlet

    # -----------------------------
    # Compressor base geometry
    # -----------------------------
    r2 = 200e-3  # Radius at impeller exit (m)
    base_geom = {
        'N_blades': 16,  # Number of blades
        'blade_thickness_inlet': 2e-3,  # Blade thickness at inlet (m)
        'blade_thickness_outlet': 2e-3,  # Blade thickness at outlet (m)
        'roughness': 3.6e-6,  # Surface roughness (m)
        'beta2_blade': -40,  # Blade angle at outlet (deg)
        'beta1_blade_rms': -50,  # Blade angle at inlet RMS (deg)
        'alpha1': 0,  # Absolute flow angle at inlet (deg)
        'r2': r2,  # Radius at impeller exit
        'r3': 1.6 * r2,  # Radius at vaneless diffuser exit
        'r1_hub': 0.3 * r2  # Hub radius at inlet
        # b2 and r1_shroud will be optimized
    }

    # -----------------------------
    # Impeller tip speed
    # -----------------------------
    Ma_u2 = 0.8  # Tip Mach number
    u_2 = Ma_u2 * a_01  # Tip speed
    omega_rads = u_2 / r2  # Angular velocity (rad/s)
    omega_rpm = omega_rads * 60 / (2 * math.pi)  # Convert to RPM

    # -----------------------------
    # Prepare parametric study
    # -----------------------------
    phi_values = np.arange(0.05, 0.105, 0.01)  # Flow coefficients (non-dimensional)
    eta_ts_values = []  # To store optimized total-static isentropic efficiencies

    # -----------------------------
    # Run simulations with optimization
    # -----------------------------
    for i, phi in enumerate(phi_values, 1):
        # Compute mass flow rate for this phi
        mdot = phi * rho_01 * u_2 * (2 * r2) ** 2

        # Objective function: negative efficiency (to maximize eta_ts)
        def objective(x):
            # Extract optimization variables: ratios relative to r2
            b2_ratio, r1s_ratio = x

            # Create geometry dictionary with optimized b2 and r1_shroud
            geom = base_geom.copy()
            geom['b2'] = b2_ratio * r2  # Set blade height at impeller outlet
            geom['r1_shroud'] = r1s_ratio * r2  # Set shroud radius at inlet

            # Initialize and configure compressor model
            cmp = CentrifugalCompressor()
            cmp.set_fluid(fluid)
            cmp.set_inlet_conditions(p_01, T_01)
            cmp.set_geometry(geom)
            cmp.set_operating_conditions(mdot, omega_rpm)

            # Run simulation and return negative efficiency (to maximize it)
            cmp.simulate(verbose=False)

            return -cmp.eta_is_ts

        # -----------------------------
        # Optimization settings
        # -----------------------------
        bounds = [(0.1, 0.2), (0.5, 0.7)]  # Bounds for b2/r2 and r1_shroud/r2
        x0 = [0.17, 0.6]  # Initial guess

        result = minimize(objective, x0, bounds=bounds, method='SLSQP',tol=1e-4)

        # Store result
        if result.success:
            eta_ts_values.append(-result.fun)
        else:
            eta_ts_values.append(np.nan)

        print(f"[{i}/{len(phi_values)}] Optimization exit status: {result.status}, "
            f"objective: {-result.fun:.6f}")

    # -----------------------------
    # Plot results
    # -----------------------------
    plt.plot(phi_values, eta_ts_values, marker='o')
    plt.xlabel("Phi (Flow Coefficient) [-]")
    plt.ylabel("Optimized Isentropic Efficiency (Total-Static) [-]")
    plt.title("Efficiency vs Phi with Geometry Optimization")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Execute the study
# -----------------------------
if __name__ == "__main__":
    run_optimized_simulations()
