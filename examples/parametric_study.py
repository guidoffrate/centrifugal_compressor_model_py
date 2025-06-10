# This file is part of the Centrifugal Compressor Model
# Copyright (c) 2024 G.F. Frate
# Licensed under the MIT License

import math
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from compressor_model import CentrifugalCompressor


def run_simulations():
    # -----------------------------
    # Define fluid and inlet state
    # -----------------------------
    fluid = 'Air'
    p_01 = 1e5  # Inlet total pressure in Pa
    T_01 = 100 + 273.15  # Inlet total temperature in Kelvin
    rho_01 = PropsSI("DMASS", "T", T_01, "P", p_01, fluid)  # Inlet density
    a_01 = PropsSI("A", "T", T_01, "P", p_01, fluid)  # Speed of sound at inlet

    # -----------------------------
    # Compressor geometry
    # -----------------------------
    r2 = 200e-3  # Radius at impeller exit (m)
    geom = {
        'N_blades': 16,  # Number of blades
        'blade_thickness_inlet': 2e-3,  # Blade thickness at inlet (m)
        'blade_thickness_outlet': 2e-3,  # Blade thickness at outlet (m)
        'roughness': 3.6e-6,  # Surface roughness (m)
        'beta2_blade': -40,  # Blade angle at outlet (deg)
        'beta1_blade_rms': -50,  # Blade angle at inlet RMS (deg)
        'alpha1': 0,  # Absolute flow angle at inlet (deg)
        'r2': r2,  # Radius at impeller exit
        'r3': 1.6 * r2,  # Radius at vaneless diffuser exit
        'r1_hub': 0.3 * r2,  # Hub radius at inlet
        'b2': 0.17 * r2,  # blade height at impeller outlet
        'r1_shroud': 0.7 * r2  # shroud radius at inlet
    }

    # -----------------------------
    # Impeller tip speed
    # -----------------------------
    Ma_u2 = 0.8  # Tip Mach number
    u_2 = Ma_u2 * a_01  # Tip speed
    omega_rads = u_2 / geom['r2']  # Angular velocity (rad/s)
    omega_rpm = omega_rads * 60 / (2 * math.pi)  # Convert to RPM

    # -----------------------------
    # Prepare parametric study
    # -----------------------------
    phi_values = np.arange(0.05, 0.105, 0.01)  # Flow coefficients (non-dimensional)
    eta_ts_values = []  # To store total-static isentropic efficiencies

    # -----------------------------
    # Run simulations
    # -----------------------------
    for phi in phi_values:
        # Create a new compressor instance for each case
        cmp = CentrifugalCompressor()
        cmp.set_fluid(fluid)
        cmp.set_inlet_conditions(p_01, T_01)
        cmp.set_geometry(geom)

        # Mass flow rate based on phi
        mdot = phi * rho_01 * u_2 * (2 * geom['r2']) ** 2
        cmp.set_operating_conditions(mdot, omega_rpm)

        # Run simulation
        cmp.simulate(verbose=False)

        # Store total-static isentropic efficiency
        eta_ts_values.append(cmp.eta_is_ts)

    # -----------------------------
    # Plot results
    # -----------------------------
    plt.plot(phi_values, eta_ts_values, marker='o')
    plt.xlabel("Phi (Flow Coefficient) [-]")
    plt.ylabel("Isentropic Efficiency (Total-Static) [-]")
    plt.title("Isentropic Efficiency vs Flow Coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Execute the study
# -----------------------------
if __name__ == "__main__":
    run_simulations()
