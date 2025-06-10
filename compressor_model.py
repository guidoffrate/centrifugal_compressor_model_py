# This file is part of the Centrifugal Compressor Model
# Copyright (c) 2024 G.F. Frate
# Licensed under the MIT License

import math
import CoolProp.CoolProp as CP
from CoolProp import AbstractState, PT_INPUTS, DmassT_INPUTS, DmassP_INPUTS
from scipy.optimize import fsolve
from scipy.optimize import brentq
import numpy as np


class CentrifugalCompressor:
    """
    A Python class for modeling a centrifugal compressor based on the MATLAB code.
    This is the base structure with initial placeholders for properties and methods.
    """

    def __init__(self):
        # Geometry and blade design
        self.N_blades = None
        self.blade_thickness_inlet = None
        self.blade_thickness_outlet = None
        self.roughness = None
        self.disk_clearance = None
        self.radial_clearance = None
        self.axial_length = None
        self.r1_shroud = None
        self.r1_hub = None
        self.r1_rms = None
        self.r2 = None
        self.r3 = None
        self.r4 = None
        self.b2 = None
        self.b3 = None
        self.cone_diameter_outlet = None
        self.cone_diameter_inlet = None
        self.cone_length = None

        # Flow areas
        self.A1 = None
        self.A1_throat = None
        self.A2 = None
        self.A3 = None
        self.A4 = None
        self.A5 = None

        # Mass flow and speed
        self.mdot = None
        self.omega_rpm = None
        self.omega_rads = None

        # Flow angles
        self.alpha1 = None
        self.beta1_blade_rms = None
        self.beta1_blade_shroud = None
        self.beta1_blade_hub = None
        self.beta1_rms = None
        self.beta2_blade = None
        self.alpha2 = None

        # Velocities
        self.u1_rms = None
        self.u1_hub = None
        self.u1_shroud = None
        self.u2 = None

        self.c1 = None
        self.c1_meridional = None
        self.c1_tangential = None
        self.c1_throat = None
        self.c2 = None
        self.c2_meridional = None
        self.c2_tangential = None
        self.c3 = None
        self.c3_meridional = None
        self.c3_tangential = None
        self.c3_is = None
        self.c4 = None
        self.c5 = None
        self.w1_rms = None
        self.w1_shroud = None
        self.w1_hub = None
        self.w1_throat = None
        self.w2 = None

        # Thermodynamic properties
        self.T_01 = None
        self.p_01 = None
        self.h_01 = None
        self.s_01 = None
        self.rho_01 = None
        self.T_1 = None
        self.p_1 = None
        self.h_1 = None
        self.s_1 = None
        self.rho_1 = None
        self.rho_1throat = None
        self.T_1throat = None
        self.p_1throat = None
        self.h1_throat = None
        self.Ma_1 = None
        self.T_2 = None
        self.p_2 = None
        self.h_2 = None
        self.s_2 = None
        self.rho_2 = None
        self.T_02 = None
        self.p_02 = None
        self.h_02 = None
        self.h_02is = None
        self.T_02is = None
        self.T_2is = None
        self.h_2is = None
        self.p_02is = None
        self.p_2 = None
        self.Ma_2 = None
        self.T_3 = None
        self.T_03 = None
        self.h_3 = None
        self.h_03 = None
        self.s_3 = None
        self.rho_3 = None
        self.Re_3 = None
        self.T_3is = None
        self.h_3is = None
        self.p_03 = None
        self.p_03is = None
        self.rho_3is = None
        self.T_4 = None
        self.T_04 = None
        self.h_4 = None
        self.h_04 = None
        self.s_4 = None
        self.rho_4 = None
        self.p_4 = None
        self.p_04 = None
        self.Ma_4 = None
        self.T_5 = None
        self.T_05 = None
        self.h_5 = None
        self.h_05 = None
        self.s_5 = None
        self.rho_5 = None
        self.p_5 = None
        self.p_05 = None
        self.Ma_5 = None

        # Losses and performance
        self.L_eulero = None
        self.Dh_internal_loss = None
        self.Dh_external_loss = None
        self.eta_is_tt = None
        self.eta_is_ts = None
        self.eta_pol_tt = None
        self.eta_pol_ts = None
        self.phi = None
        self.psi = None
        self.PR_tt = None
        self.PR_ts = None
        self.PR_ss = None
        self.RR = None

        # Fluid properties
        self.fluid = None
        self.cp_fluid = None
        self.k_fluid = None
        self.R_fluid = None
        self.mu_ref = None
        self.fluid_as = None

        # Simulation settings
        self.convergence = {}
        self.fsolve_options = {}
        self.verbose = False

    def set_geometry(self, geometry_specs):
        """
        set_geometry Sets the compressor geometrical specifications
        """

        self.N_blades = geometry_specs['N_blades']
        # self.blade_thickness = geometry_specs['blade_thickness']
        self.blade_thickness_inlet = geometry_specs['blade_thickness_inlet']
        self.blade_thickness_outlet = geometry_specs['blade_thickness_outlet']

        self.disk_clearance = 1e-3  # (m) - imposed here. It can be moved outside.
        self.radial_clearance = 0.15e-3  # (m) - imposed here. It can be moved outside.

        self.roughness = geometry_specs['roughness']
        self.cone_divergence_angle = 5  # (deg) - imposed here. It can be moved outside, but it is probably always constant

        self.r2 = geometry_specs['r2']

        self.r1_shroud = geometry_specs['r1_shroud']
        self.r1_hub = geometry_specs['r1_hub']
        self.r1_rms = ((self.r1_shroud ** 2 + self.r1_hub ** 2) / 2) ** 0.5
        self.A1 = math.pi * (self.r1_shroud ** 2 - self.r1_hub ** 2)
        self.alpha1 = geometry_specs['alpha1']
        self.beta1_blade_rms = geometry_specs['beta1_blade_rms']

        self.b2 = geometry_specs['b2']
        self.A2 = (2 * math.pi * self.r2 - self.N_blades * self.blade_thickness_outlet) * self.b2
        self.beta2_blade = geometry_specs['beta2_blade']

        self.r3 = geometry_specs['r3']
        self.b3 = self.b2
        self.A3 = 2 * math.pi * self.r3 * self.b3

        self.r4 = 1.5 * self.r3  # (m) - imposed here. It can be moved outside.
        self.A4 = math.pi * (self.r4 - self.r3) ** 2

        self.cone_length = self.r3 * 1.2  # (m) - imposed here. It can be moved outside.
        self.cone_diameter_inlet = 2 * (self.r4 - self.r3)
        self.cone_diameter_outlet = self.cone_diameter_inlet + 2 * self.cone_length * math.sin(
            math.radians(self.cone_divergence_angle / 2))
        self.A5 = math.pi * self.cone_diameter_outlet ** 2 / 4

        self.axial_length = 0.4 * (2 * self.r2 - (2 * self.r1_shroud + 2 * self.r1_hub) / 2)

    def set_fluid(self, fluid):
        """
        Set the fluid and its thermodynamic reference properties.
        """
        self.fluid = fluid
        self.fluid_as = AbstractState("BICUBIC&HEOS", fluid)
        self.mu_ref = CP.PropsSI('VISCOSITY', 'T', 273.15, 'P', 101325, fluid)
        self.R_fluid = CP.PropsSI('GAS_CONSTANT', fluid) / CP.PropsSI('MOLAR_MASS', fluid)

    def set_inlet_conditions(self, p_01, T_01):
        """
        Set the thermodynamic properties at the compressor suction.
        """

        if self.fluid is None:
            raise ValueError("The 'fluid' attribute must be set before calling 'set_inlet_conditions'. "
                         "Use 'set_fluid(fluid_name)' first.")

        self.T_01 = T_01
        self.p_01 = p_01

        self.cp_fluid = CP.PropsSI('CPMASS', 'T', T_01, 'P', p_01, self.fluid)
        self.k_fluid = self.cp_fluid / CP.PropsSI('CVMASS', 'T', T_01, 'P', p_01, self.fluid)

        self.h_01 = self.T_01 * self.cp_fluid
        self.s_01 = CP.PropsSI('S', 'T', T_01, 'P', p_01, self.fluid)
        self.rho_01 = CP.PropsSI('DMASS', 'T', T_01, 'P', p_01, self.fluid)

    def set_operating_conditions(self, mdot, omega_rpm):
        """
        Set the compressor operating conditions.
        """
        self.mdot = mdot
        self.omega_rpm = omega_rpm
        self.omega_rads = 2 * math.pi * omega_rpm / 60

    def _section_1(self, rho_guess):
        self.rho_1 = rho_guess
        self.c1 = self.mdot / (self.A1 * self.rho_1)
        self.T_1 = self.T_01 - self.c1 ** 2 / (2 * self.cp_fluid)
        self.p_1 = self.p_01 / ((1 + ((self.k_fluid - 1) / 2) * (
                    self.c1 / math.sqrt(self.k_fluid * self.R_fluid * self.T_1)) ** 2) ** (
                                            self.k_fluid / (self.k_fluid - 1)))

    def _section_1throat(self, rho_guess):
        self.rho_1throat = rho_guess
        self.w1_throat = self.mdot / (self.rho_1throat * self.A1_throat)
        w1_tan = self.w1_throat * math.sin(math.radians(abs(self.beta1_blade_rms)))
        w1_mer = self.w1_throat * math.cos(math.radians(self.beta1_blade_rms))
        c1_tan = self.u1_rms - w1_tan
        self.c1_throat = math.sqrt(c1_tan ** 2 + w1_mer ** 2)
        self.h1_throat = self.h_01 - self.c1_throat ** 2 / 2
        self.T_1throat = self.h1_throat / self.cp_fluid
        a_1throat = math.sqrt(self.k_fluid * self.R_fluid * self.T_1throat)
        self.Ma_1throat = self.c1_throat / a_1throat
        self.p_1throat = self.p_01 / (
                    (1 + (self.k_fluid - 1) / 2 * self.Ma_1throat ** 2) ** (self.k_fluid / (self.k_fluid - 1)))

    def _section_2(self, rho_guess):
        self.rho_2 = rho_guess

        # Meridional and tangential velocity
        self.c2_meridional = self.mdot / (self.rho_2 * self.A2)
        self.c2_tangential = self.slip_factor() * self.u2 - self.c2_meridional * math.tan(
            math.radians(abs(self.beta2_blade)))
        self.c2 = math.sqrt(self.c2_meridional ** 2 + self.c2_tangential ** 2)

        # Relative velocities
        self.w2_tangential = self.u2 - self.c2_tangential
        self.w2 = math.sqrt(self.w2_tangential ** 2 + self.c2_meridional ** 2)
        self.alpha2 = math.degrees(math.acos(self.c2_meridional / self.c2))
        self.beta2 = math.degrees(math.acos(self.c2_meridional / self.w2))

        # Euler work
        self.L_eulero = self.u2 * self.c2_tangential - self.u1_rms * self.c1_tangential
        self.h_02 = self.h_01 + self.L_eulero
        self.T_02 = self.h_02 / self.cp_fluid

        self.h_2 = self.h_02 - self.c2 ** 2 / 2
        self.T_2 = self.h_2 / self.cp_fluid

        # Internal losses
        Dh_inducer = self.loss_inducer_incidence()
        Dh_skin = self.loss_impeller_skin_friction()
        Dh_loading = self.loss_impeller_blade_loading()
        Dh_clearance = self.loss_impeller_clearance()
        self.Dh_internal_loss = Dh_inducer + Dh_skin + Dh_loading + Dh_clearance

        # External losses
        Dh_mixing = self.loss_impeller_mixing()
        Dh_disk = self.loss_impeller_disk_friction()
        Dh_recirculation = self.loss_impeller_recirculation()
        self.Dh_external_loss = Dh_mixing + Dh_disk + Dh_recirculation

        # Isentropic outlet enthalpy and pressure
        self.h_02is = self.h_02 - self.Dh_internal_loss
        self.h_2is = self.h_02is - self.c2 ** 2 / 2
        self.T_02is = self.h_02is / self.cp_fluid
        self.T_2is = self.h_2is / self.cp_fluid
        self.p_02is = self.p_01 * (self.T_02is / self.T_01) ** (self.k_fluid / (self.k_fluid - 1))
        self.p_2 = self.p_02is * (self.T_02is / self.T_2is) ** (self.k_fluid / (1 - self.k_fluid))

    def _section_3(self, rho_guess):
        self.rho_3 = rho_guess
        self.c3_meridional = self.mdot / (self.rho_3 * self.A3)

        # Solve for Re_3
        def _residual_Re3(Re_guess):
            self._calculate_Re3(Re_guess)
            self.fluid_as.update(DmassT_INPUTS, self.rho_3, self.T_3)
            mu_3 = self.fluid_as.viscosity()
            D_hydr = 2 * self.b3
            return Re_guess - self.rho_3 * self.c3 * D_hydr / mu_3

        # Generate a good starting point for Re
        self.fluid_as.update(DmassP_INPUTS, self.rho_3, self.p_2)
        mu_3_guess = self.fluid_as.viscosity()
        Re_3_guess = self.rho_3 * self.c2 * 2 * self.b3 / mu_3_guess
        Re_3_sol, info_re, ier_re, msg_re = fsolve(_residual_Re3, Re_3_guess, xtol=1e-6, full_output=True)

        if ier_re != 1:
            # Use free vortex without losses
            self.c3_tangential = self.c2_tangential * (self.r2 / self.r3)
            self.c3 = math.sqrt(self.c3_meridional ** 2 + self.c3_tangential ** 2)
            self.T_3 = self.T_03 - self.c3 ** 2 / (2 * self.cp_fluid)
            self.h_3 = self.h_03 - self.c3 ** 2 / 2
            if self.verbose:
                print(f"Warning: Re_3 fsolve did not converge: {msg_re}")
        else:
            self._calculate_Re3(Re_3_sol[0])
            if self.verbose:
                print(f"Warning: fsolve converged for Re_3: {msg_re}")

        # From the actual conditions we calculate the isentropic ones
        self.h_03is = self.h_03 - self.loss_vaneless_diffuser()
        self.T_03is = self.h_03is / self.cp_fluid
        self.p_03is = self.p_02 * (self.T_02 / self.T_03is) ** (self.k_fluid / (1 - self.k_fluid))

        def _residual_rho_3is(rho_3is_guess):
            self._calculate_rho_3is(rho_3is_guess)
            return rho_3is_guess - self.p_3 / (self.R_fluid * self.T_3is)

        rho_3is_sol, info_is, ier_is, msg_is = fsolve(_residual_rho_3is, self.rho_3, xtol=1e-6, full_output=True)
        if ier_is != 1:
            self.rho_3is = self.rho_3
            if self.verbose:
                print(f"Warning: rho_3is fsolve did not converge: {msg_is}")
        else:
            self._calculate_rho_3is(rho_3is_sol[0])
            if self.verbose:
                print(f"Warning: fsolve converged for rho_3is: {msg_is}")

    def _calculate_Re3(self, Re_guess):
        self.Re_3 = Re_guess
        cf = 0.0058 * (1.8e5 / self.Re_3) ** 0.2  # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        self.c3_tangential = self.c2_tangential / (
                self.r3 / self.r2 + (2 * math.pi * cf * self.rho_2 * self.c2_tangential * (
                self.r3 ** 2 - self.r2 * self.r3)) / self.mdot)
        self.c3 = math.sqrt(self.c3_meridional ** 2 + self.c3_tangential ** 2)
        self.h_3 = self.h_03 - self.c3 ** 2 / 2
        self.T_3 = self.T_03 - self.c3 ** 2 / (2 * self.cp_fluid)

    def _calculate_rho_3is(self, rho_3is_guess):
        self.rho_3is = rho_3is_guess
        c3_mer_is = self.mdot / (self.rho_3is * self.A3)
        c3_tan_is = self.c2_tangential * (self.r2 / self.r3)
        self.c3_is = math.sqrt(c3_mer_is ** 2 + c3_tan_is ** 2)
        self.T_3is = self.T_03is - self.c3_is ** 2 / (2 * self.cp_fluid)
        self.p_3 = self.p_03is * (self.T_03is / self.T_3is) ** (self.k_fluid / (1 - self.k_fluid))

    def _section_4(self, rho_guess):
        self.rho_4 = rho_guess
        self.c4 = self.mdot / (self.rho_4 * self.A4)
        self.T_4 = self.T_04 - self.c4 ** 2 / (2 * self.cp_fluid)
        self.p_04 = self.p_03 - (self.p_03 - self.p_3) * self.loss_factor_volute()
        self.Ma_4 = self.c4 / math.sqrt(self.k_fluid * self.R_fluid * self.T_4)
        self.p_4 = self.p_04 / ((1 + ((self.k_fluid - 1) / 2) * self.Ma_4 ** 2) ** (self.k_fluid / (self.k_fluid - 1)))

    def slip_factor(self):
        # Reference:
        # (Wiesner, 1967) - doi:10.1115/1.3616734
        # (Meroni, 2018) - doi:10.1016/j.apenergy.2018.09.210
        # (Aungier, 2000) - doi:10.1115/1.800938
        sigma = 1 - math.sqrt(math.cos(math.radians(self.beta2_blade))) / self.N_blades ** 0.7
        sigma_star = math.sin(math.radians(19 + 0.2 * (90 - abs(self.beta2_blade))))
        A = (sigma - sigma_star) / (1 - sigma_star)
        B = (self.r1_rms / self.r2 - A) / (1 - A)
        if self.r1_rms / self.r2 < A:
            return sigma
        return sigma * (1 - B ** math.sqrt((90 - abs(self.beta2_blade)) / 10))

    def loss_inducer_incidence(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        beta1_opt = math.degrees(
            math.atan((self.A1 / self.A1_throat) * math.tan(math.radians(abs(self.beta1_blade_rms)))))
        E = math.sin(math.radians(abs(beta1_opt - self.beta1_rms)))
        return self.w1_rms ** 2 * E ** 2 / 2

    def loss_impeller_skin_friction(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        self.fluid_as.update(DmassT_INPUTS, self.rho_2, self.T_2)
        mu_2 = self.fluid_as.viscosity()
        E = (math.cos(math.radians(self.beta1_blade_shroud)) + math.cos(math.radians(self.beta1_blade_hub))) / 2
        Lb = (math.pi / 8) * (2 * self.r2 - (self.r1_shroud + self.r1_hub) - self.b2 + 2 * self.axial_length) * (
                    2 / (E + math.cos(math.radians(self.beta2_blade))))
        F = 2 * self.r2 / ((self.N_blades / math.pi * math.cos(math.radians(self.beta2_blade))) + 2 * self.r2 / self.b2)
        G = math.tan(math.radians(self.beta1_blade_shroud))
        lamda = self.r1_hub / self.r1_shroud
        H = 2 * self.r1_shroud / (2 / (1 - lamda) + (2 * self.N_blades / (math.pi * (1 + lamda))) * math.sqrt(
            1 + G ** 2 * (1 + lamda ** 2 / 2)))
        D_hydr = F + H
        Re = (self.rho_2 * self.c2 * D_hydr) / mu_2
        Re_e = (Re - 2000) * (self.roughness / D_hydr)
        Cf = self.friction_factor(Re, Re_e, D_hydr)
        K1 = math.sqrt((self.w1_rms ** 2 + self.w2 ** 2) / 2)
        K2 = math.sqrt((self.w1_throat ** 2 + self.w2 ** 2) / 2)
        W = max(K1, K2)
        return 4 * Cf * (Lb / D_hydr) * W ** 2

    def loss_impeller_blade_loading(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        E = (self.N_blades / math.pi * (1 - self.r1_shroud / self.r2) + 2 * (self.r1_shroud / self.r2))
        F = 0.75 * self.u2 * self.c2_tangential / self.u2 ** 2 * self.w2 / self.w1_shroud
        Df = 1 - (self.w2 / self.w1_shroud) + F / E
        return 0.05 * Df ** 2 * self.u2 ** 2

    def loss_impeller_clearance(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        E = 4 * math.pi / (self.b2 * self.N_blades)
        F = (self.r1_shroud ** 2 - self.r1_hub ** 2) / ((self.r2 - self.r1_shroud) * (1 + self.rho_2 / self.rho_1))
        return 0.6 * (self.radial_clearance / self.b2) * abs(self.c2_tangential) * math.sqrt(
            E * F * abs(self.c2_tangential) * self.c1)

    def loss_impeller_mixing(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        b_star = 1
        csi = 0.15
        epsilon_w = (-0.07 + math.sqrt(0.07 ** 2 + 4 * 0.93 * csi)) / (2 * 0.93)
        E = math.tan(math.radians(self.alpha2))
        return (1 / (1 + E ** 2)) * ((1 - epsilon_w - b_star) / (1 - epsilon_w)) ** 2 * self.c2 ** 2 / 2

    def loss_impeller_disk_friction(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        self.fluid_as.update(DmassT_INPUTS, self.rho_2, self.T_2)
        mu_2 = self.fluid_as.viscosity()
        rho_ave = (self.rho_1 + self.rho_2) / 2
        Re = self.rho_2 * self.u2 * self.r2 / mu_2
        if Re > 3e5:
            Kf = 0.102 * (self.disk_clearance / self.b2) ** 0.1 / Re ** 0.2
        else:
            Kf = 3.7 * (self.disk_clearance / self.b2) ** 0.1 / Re ** 0.5
        return 0.25 * rho_ave * self.u2 ** 3 * self.r2 ** 2 * Kf / self.mdot

    def loss_impeller_recirculation(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        E = (self.N_blades / math.pi * (1 - self.r1_shroud / self.r2) + 2 * (self.r1_shroud / self.r2))
        F = 0.75 * self.u2 * self.c2_tangential / self.u2 ** 2 * self.w2 / self.w1_shroud
        Df = 1 - (self.w2 / self.w1_shroud) + F / E
        return 8e-5 * math.sinh(3.5 * math.radians(self.alpha2) ** 3) * Df ** 2 * self.u2 ** 2

    def loss_vaneless_diffuser(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        cf = 0.005 * (1.8e5 / self.Re_3) ** 0.2
        return (cf * self.r2 * (1 - (self.r2 / self.r3) ** 1.5) * self.c2 ** 2) / (
                1.5 * self.b2 * math.cos(math.radians(self.alpha2)))

    def loss_factor_volute(self):
        # (Aungier, 2000) - doi:10.1115/1.800938
        # Meridional velocity loss
        loss_1 = (self.c3_meridional / self.c3) ** 2

        # Tangential velocity loss
        sp = self.r3 * self.c3_tangential / (self.r4 * self.c4)
        if sp >= 1:
            loss_2 = 0.5 * self.r3 / self.r4 * (self.c3_tangential / self.c3) ** 2 * (1 - 1 / sp ** 2)
        else:
            loss_2 = self.r3 / self.r4 * (self.c3_tangential / self.c3) ** 2 * (1 - 1 / sp) ** 2

        # Skin friction losses
        L_ave = math.pi * (self.r3 + self.r4) / 2  # (m) - Average length travelled by the fluid in the volute
        D_hydr = math.sqrt(4 * self.A4 / math.pi)
        self.fluid_as.update(DmassT_INPUTS, self.rho_4, self.T_4)
        mu_4 = self.fluid_as.viscosity()
        Re = (self.rho_4 * self.c4 * D_hydr) / mu_4
        Re_e = (Re - 2000) * (self.roughness / D_hydr)
        loss_3 = 4 * self.friction_factor(Re, Re_e, D_hydr) * (self.c4 / self.c3) ** 2 * L_ave / D_hydr

        return loss_1 + loss_2 + loss_3  # Total pressure loss factor

    def loss_factor_cone(self):
        # (Aungier, 2000) - doi:10.1115/1.800938
        return ((self.c4 - self.c5) / self.c3) ** 2  # Total pressure loss factor

    def friction_factor_old(self, Re, Re_e, D_hydr):
        # Calculation of the friction factor - (Aungier, 2000) -
        # 10.1115/1.800938 - pp. 74 - 75

        # Colebrook function - CFS
        def f0_cfs(x):
            x = np.asarray(x, dtype=float)  # Ensure x is an array
            x = np.clip(x, 1e-4, 10)  # Element-wise clamp
            return 2.51 * 10 ** (1 / (2 * x)) - Re * x

        x0_cfs = 1.0
        cfs_solution = fsolve(f0_cfs, x0_cfs, xtol=1e-6)[0]
        cfs = cfs_solution ** 2 / 4

        # Colebrook function - CFR
        def f0_cfr(x):
            x = np.asarray(x, dtype=float)  # Ensure x is an array
            x = np.clip(x, 1e-4, 10)  # Element-wise clamp
            return 10 ** (1 / (2 * x)) * self.roughness - 3.71 * D_hydr

        x0_cfr = 0.5
        cfr_solution = fsolve(f0_cfr, x0_cfr, xtol=1e-6)[0]
        cfr = cfr_solution ** 2 / 4

        # Transition based on Re_e
        if Re_e < 60:
            Cf = cfs
        else:
            Cf = cfs + (cfr - cfs) * (1 - 60 / Re_e)

        return Cf

    def friction_factor_haaland(self, Re, D_hydr):
        eD = self.roughness / D_hydr
        if Re <= 0:
            raise ValueError("Reynolds number must be positive.")
        return 1 / (-1.8 * np.log10((6.9 / Re) + (eD / 3.7) ** 1.11)) ** 2

    def friction_factor(self, Re, Re_e, D_hydr):
        """
        Calculate the Darcy friction factor using a Colebrook-like formulation
        and robust bounded root-finding via Brent's method (brentq).

        Reference: Aungier (2000), pp. 74–75
        """

        # Colebrook function - CFS (smooth pipe)
        def f0_cfs(x):
            return 2.51 * 10 ** (1 / (2 * x)) - Re * x

        # Colebrook function - CFR (rough pipe)
        def f0_cfr(x):
            return 10 ** (1 / (2 * x)) * self.roughness - 3.71 * D_hydr

        try:
            cfs_root = brentq(f0_cfs, 0.05, 1.5, xtol=1e-6)
            cfs = cfs_root ** 2 / 4
        except ValueError:
            print("Warning: brentq failed for cfs; using fallback")
            cfs = 0.02  # reasonable fallback

        try:
            cfr_root = brentq(f0_cfr, 0.05, 1.5, xtol=1e-6)
            cfr = cfr_root ** 2 / 4
        except ValueError:
            print("Warning: brentq failed for cfr; using fallback")
            cfr = 0.02

        # Transition between CFS and CFR
        if Re_e < 60:
            Cf = cfs
        else:
            Cf = cfs + (cfr - cfs) * (1 - 60 / Re_e)

        return Cf

    def simulate(self, verbose=True):
        """
        Begin simulation of the compressor by calculating inlet velocity and static conditions.

        Parameters:
        verbose (bool): If True, prints simulation status and key parameters.
        """
        self.verbose = verbose

        if self.verbose:
            print("Running compressor simulation...")
            print(f"Mass flow rate: {self.mdot} kg/s")
            print(f"Rotational speed: {self.omega_rpm} RPM ({self.omega_rads:.2f} rad/s)")
            print(f"Inlet total pressure: {self.p_01:.2f} Pa")
            print(f"Inlet total temperature: {self.T_01:.2f} K")
            print(f"Fluid: {self.fluid}")

        """ Impeller inlet """

        self.u1_rms = self.omega_rads * self.r1_rms
        self.u1_hub = self.omega_rads * self.r1_hub
        self.u1_shroud = self.omega_rads * self.r1_shroud

        def _residual_rho_1(rho_guess):
            # Evaluate the section
            self._section_1(rho_guess)
            # Compare guess and model result
            return rho_guess - self.p_1 / (self.R_fluid * self.T_1)

        rho_1_solution, info, ier, msg = fsolve(_residual_rho_1, self.rho_01, xtol=1e-6, full_output=True)

        # Update all the values with the final result
        if ier != 1:
            self._section_1(self.rho_01)
            if self.verbose:
                print(f"Warning: fsolve did not converge for impeller inlet. Message: {msg}")
        else:
            self._section_1(rho_1_solution[0])
            if self.verbose:
                print(f"Warning: fsolve converged for impeller inlet. Message: {msg}")

        self.c1 = self.mdot / (self.A1 * self.rho_1)
        self.c1_meridional = self.c1 * math.cos(math.radians(self.alpha1))
        self.c1_tangential = self.c1 * math.sin(math.radians(self.alpha1))

        self.w1_rms = math.sqrt(self.c1 ** 2 + self.u1_rms ** 2)
        self.beta1_rms = math.degrees(math.acos(self.c1 / self.w1_rms))
        self.w1_shroud = math.sqrt(self.c1 ** 2 + self.u1_shroud ** 2)
        self.beta1_shroud = math.degrees(math.acos(self.c1 / self.w1_shroud))
        self.w1_hub = math.sqrt(self.c1 ** 2 + self.u1_hub ** 2)
        self.beta1_hub = math.degrees(math.acos(self.c1 / self.w1_hub))

        self.T_1 = self.T_01 - self.c1 ** 2 / (2 * self.cp_fluid)
        a1 = math.sqrt(self.k_fluid * self.R_fluid * self.T_1)
        self.Ma_1 = self.c1 / a1
        self.p_1 = self.p_01 / ((1 + ((self.k_fluid - 1) / 2) * self.Ma_1 ** 2) ** (self.k_fluid / (self.k_fluid - 1)))

        self.h_1 = self.h_01 - self.c1 ** 2 / 2
        self.fluid_as.update(PT_INPUTS, self.p_1, self.T_1)
        self.s_1 = self.fluid_as.smass()

        """ Throat section """

        # Compute blade-relative velocities at RMS
        c_blade_rms = self.u1_rms / math.tan(math.radians(abs(self.beta1_blade_rms)))
        c_blade_hub = c_blade_rms
        c_blade_shroud = c_blade_rms

        self.beta1_blade_hub = -math.degrees(math.atan(self.u1_hub / c_blade_hub))
        self.beta1_blade_shroud = -math.degrees(math.atan(self.u1_shroud / c_blade_shroud))

        s_shroud = 2 * math.pi * self.r1_shroud / self.N_blades - self.blade_thickness_inlet
        s_hub = 2 * math.pi * self.r1_hub / self.N_blades - self.blade_thickness_inlet
        s_rms = 2 * math.pi * self.r1_rms / self.N_blades - self.blade_thickness_inlet

        o_shroud = s_shroud * math.cos(math.radians(self.beta1_blade_shroud))
        o_hub = s_hub * math.cos(math.radians(self.beta1_blade_hub))
        o_rms = s_rms * math.cos(math.radians(self.beta1_blade_rms))

        self.A1_throat = self.N_blades / 2 * (
                o_hub * (self.r1_rms - self.r1_hub) +
                o_rms * (self.r1_shroud - self.r1_hub) +
                o_shroud * (self.r1_shroud - self.r1_rms)
        )

        def _residual_rho_1throat(rho_guess):
            # Evaluate the section
            self._section_1throat(rho_guess)
            # Compare guess and model result
            return rho_guess - self.p_1throat / (self.R_fluid * self.T_1throat)

        rho_1throat_sol, info, ier, msg = fsolve(_residual_rho_1throat, self.rho_1, xtol=1e-6, full_output=True)

        # Update all the values with the final result
        if ier != 1:
            self._section_1throat(self.rho_1)
            if self.verbose:
                print(f"Warning: fsolve for throat did not converge. Message: {msg}")
        else:
            self._section_1throat(rho_1throat_sol[0])
            if self.verbose:
                print(f"Warning: fsolve for throat converged. Message: {msg}")

        """ Impeller outlet """

        # Impeller outlet
        self.u2 = self.omega_rads * self.r2

        def _residual_rho_2(rho_guess):
            # Evaluate the section
            self._section_2(rho_guess)
            # Compare guess and model result
            return rho_guess - self.p_2 / (self.R_fluid * self.T_2)

        # Solve for rho_2
        rho_2_sol, info, ier, msg = fsolve(_residual_rho_2, self.rho_1throat, xtol=1e-6, full_output=True)

        # Update all the values with the final result
        if ier != 1:
            self._section_2(self.rho_1throat)
            if self.verbose:
                print(f"Warning: fsolve for impeller outlet did not converge. Message: {msg}")
        else:
            self._section_2(rho_2_sol[0])
            if self.verbose:
                print(f"Warning: fsolve converged for impeller outlet. Message: {msg}")

        # Final updates
        a2 = math.sqrt(self.k_fluid * self.R_fluid * self.T_2)
        self.Ma_2 = self.c2 / a2
        self.p_02 = self.p_2 * (1 + ((self.k_fluid - 1) / 2) * self.Ma_2 ** 2) ** (self.k_fluid / (self.k_fluid - 1))
        self.fluid_as.update(PT_INPUTS, self.p_2, self.T_2)
        self.s_2 = self.fluid_as.smass()
        self.RR = (self.h_2 + self.w2 ** 2 / 2 - self.u2 ** 2 / 2) - (
                self.h_1 + self.w1_rms ** 2 / 2 - self.u1_rms ** 2 / 2)

        """ Vaneless diffuser """

        self.h_03 = self.h_02
        self.T_03 = self.T_02

        def _residual_rho_3(rho_guess):
            # Evaluate the section
            self._section_3(rho_guess)
            # Compare guess and model result
            return rho_guess - self.p_3 / (self.R_fluid * self.T_3)

        # Solve for rho_3
        rho_3_sol, info, ier, msg = fsolve(_residual_rho_3, self.rho_2, xtol=1e-6, full_output=True)

        # Update all the values with the final result
        if ier != 1:
            self._section_3(self.rho_2)
            if self.verbose:
                print(f"Warning: fsolve for vaneless diffuser outlet did not converge. Message: {msg}")
        else:
            self._section_3(rho_3_sol[0])
            if self.verbose:
                print(f"Warning: fsolve converged for vaneless diffuser outlet. Message: {msg}")

        self.fluid_as.update(PT_INPUTS, self.p_3, self.T_3)
        self.s_3 = self.fluid_as.smass()

        a3 = math.sqrt(self.k_fluid * self.R_fluid * self.T_3)
        self.Ma_3 = self.c3 / a3
        self.p_03 = self.p_3 * (1 + (self.k_fluid - 1) / 2 * self.Ma_3 ** 2) ** (self.k_fluid / (self.k_fluid - 1))
        self.h_3is = self.h_03is - self.c3 ** 2 / 2

        """ Volute """

        self.h_04 = self.h_03  # total enthalpy conservation
        self.T_04 = self.T_03

        def _residual_rho_4(rho_guess):
            # Evaluate the section
            self._section_4(rho_guess)
            # Compare guess and model result
            return rho_guess - self.p_4 / (self.R_fluid * self.T_4)

        rho_4_sol, info, ier, msg = fsolve(_residual_rho_4, self.rho_3, xtol=1e-6, full_output=True)
        if ier != 1:
            self._section_4(self.rho_3)
            if self.verbose:
                print(f"Warning: fsolve did not converge for the volute outlet. Message: {msg}")
        else:
            self._section_4(rho_4_sol[0])
            if self.verbose:
                print(f"Warning: fsolve converged for the volute outlet. Message: {msg}")

        self.h_4 = self.h_04 - self.c4 ** 2 / 2
        self.fluid_as.update(PT_INPUTS, self.p_4, self.T_4)
        self.s_4 = self.fluid_as.smass()

        """ Exit cone """

        self.h_05 = self.h_04  # total enthalpy conservation
        self.T_05 = self.T_04
        self.rho_5 = self.rho_4  # assumed equal due to low Mach number
        self.c5 = self.mdot / (self.rho_5 * self.A5)
        self.T_5 = self.T_05 - self.c5 ** 2 / (2 * self.cp_fluid)
        a5 = math.sqrt(self.k_fluid * self.R_fluid * self.T_5)
        self.Ma_5 = self.c5 / a5
        self.p_05 = self.p_04 - (self.p_03 - self.p_3) * self.loss_factor_cone()
        self.p_5 = self.p_05 / ((1 + ((self.k_fluid - 1) / 2) * self.Ma_5 ** 2) ** (self.k_fluid / (self.k_fluid - 1)))
        self.h_5 = self.h_05 - self.c5 ** 2 / 2
        self.fluid_as.update(PT_INPUTS, self.p_5, self.T_5)
        self.s_5 = self.fluid_as.smass()

        # Global performance
        rho_volute_ave = (self.rho_3 + self.rho_4) / 2
        Dh_loss_volute = (self.p_03 - self.p_04) / rho_volute_ave
        rho_cone_ave = (self.rho_4 + self.rho_5) / 2
        Dh_loss_cone = (self.p_04 - self.p_05) / rho_cone_ave

        Dh0_id = self.L_eulero - self.Dh_internal_loss - self.loss_vaneless_diffuser() - Dh_loss_volute - Dh_loss_cone
        Dh0_act = self.L_eulero + self.Dh_external_loss

        self.eta_is_tt = Dh0_id / Dh0_act
        self.eta_is_ts = (Dh0_id - self.c5 ** 2 / 2) / Dh0_act

        Dh0 = self.h_05 - self.h_01
        Ds = self.s_5 - self.s_1
        DT = self.T_5 - self.T_1
        Dh0_pol = Dh0 - Ds * DT / math.log(self.T_5 / self.T_1)
        self.eta_pol_tt = Dh0_pol / Dh0_act
        self.eta_pol_ts = (Dh0_pol - self.c5 ** 2 / 2) / Dh0_act

        self.phi = self.mdot / (self.rho_01 * self.u2 * (2 * self.r2) ** 2)
        self.psi = Dh0_act / self.u2 ** 2

        self.PR_tt = self.p_05 / self.p_01
        self.PR_ts = self.p_5 / self.p_01
        self.PR_ss = self.p_5 / self.p_1