#!/usr/bin/env python3
"""Level 4.6: mechanism screening beyond a closed Zakharov caviton.

This module does not claim a kinetic proof.  It makes the power, phase-speed,
species and pulse-time requirements explicit for four alternatives: controlled
modulational onset, an open current-free double layer, counter-streaming ion
shocks and a driven lower-hybrid branch.  The results are boundary conditions
for a later open-boundary electromagnetic PIC/Vlasov calculation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import atomic_mass, e, epsilon_0, m_e, pi
from scipy.optimize import brentq, differential_evolution

from plasma_extrema import pump_charge_time, quartic_growth


M_D = 2.014 * atomic_mass


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def selected_regime(root: Path, cfg: dict) -> pd.Series:
    table = pd.read_csv(root / cfg["input"]["extrema_summary_csv"])
    rows = table[table.search == cfg["input"]["selected_regime"]]
    if len(rows) != 1:
        raise ValueError("The selected extrema regime is missing or ambiguous")
    return rows.iloc[0]


def active_volume(regime: pd.Series, length_m: float | None = None) -> float:
    length = float(regime.interaction_length_m if length_m is None else length_m)
    return pi * float(regime.radius_m) ** 2 * length


def mode_energy(field_Vm: float, volume_m3: float) -> float:
    """Cycle-averaged electrostatic energy for a peak field amplitude."""
    return 0.25 * epsilon_0 * field_Vm**2 * volume_m3


def neutral_threshold(field_data: dict, K_m1: float) -> float:
    numerator = field_data["sound_speed_ms"] ** 2 * (
        field_data["gamma_langmuir_s-1"] ** 2
        + field_data["dispersion_P_m2s"] ** 2 * K_m1**4
    )
    denominator = (
        2.0
        * field_data["density_coupling_A_m3s"]
        * field_data["ponderomotive_B"]
        * field_data["dispersion_P_m2s"]
        * K_m1**2
    )
    return math.sqrt(numerator / denominator)


def wave_dictionary(regime: pd.Series, q_multiplier: float = 1.0) -> dict:
    return {
        "acoustic_kinetic_damping_at_carrier_s-1": float(
            regime["exact_acoustic_kinetic_damping_at_carrier_s-1"]
        ),
        "carrier_k_m-1": float(regime["exact_carrier_k_m-1"]),
        "lambda_De_m": float(regime.lambda_De_m),
        "nu_in_effective_s-1": float(regime["exact_nu_in_effective_s-1"]),
        "nu_en_s-1": float(regime["nu_en_s-1"]),
        "sound_speed_ms": float(regime.exact_sound_speed_ms),
        # Q multiplier changes the envelope damping only; increasing Q lowers gamma.
        "gamma_langmuir_s-1": float(regime["exact_gamma_langmuir_s-1"]) / q_multiplier,
        "dispersion_P_m2s": float(regime.exact_dispersion_P_m2s),
        "density_coupling_A_m3s": float(regime.exact_density_coupling_A_m3s),
        "ponderomotive_B": float(regime.exact_ponderomotive_B),
    }


def mi_control_scan(regime: pd.Series, cfg: dict) -> pd.DataFrame:
    control = cfg["modulational_control"]
    experiment = cfg["experiment"]
    ratios = np.linspace(
        float(control["modulation_to_carrier_k_minimum"]),
        float(control["modulation_to_carrier_k_maximum"]),
        int(control["points"]),
    )
    k = float(regime["exact_carrier_k_m-1"])
    K_min_geometry = 2.0 * pi / float(regime.interaction_length_m)
    volume = active_volume(regime)
    pulse = float(experiment["pulse_duration_s"])
    seed = float(control["density_seed_fraction"])
    target = float(control["onset_depletion_fraction"])
    e_folds = math.log(target / seed)
    rows = []
    for qmult in control["Q_multipliers"]:
        wave = wave_dictionary(regime, float(qmult))
        for ratio in ratios:
            K = float(ratio) * k
            threshold = neutral_threshold(wave, K)
            for factor in control["operating_field_factors"]:
                field = float(factor) * threshold
                growth = quartic_growth(field, K, wave)
                formation = e_folds / growth if growth > 0.0 else math.inf
                for coherent_fraction in experiment["coherent_power_fractions"]:
                    power = float(coherent_fraction) * float(regime.absorbed_power_W)
                    energy = mode_energy(field, volume)
                    charge = pump_charge_time(energy, power, wave["gamma_langmuir_s-1"])
                    steady_energy = power / (2.0 * wave["gamma_langmuir_s-1"])
                    steady_field = math.sqrt(4.0 * steady_energy / (epsilon_0 * volume))
                    pulse_energy = steady_energy * (1.0 - math.exp(-2.0 * wave["gamma_langmuir_s-1"] * pulse))
                    pulse_field = math.sqrt(4.0 * pulse_energy / (epsilon_0 * volume))
                    rows.append({
                        "Q_multiplier": float(qmult),
                        "K_over_k": ratio,
                        "K_m-1": K,
                        "geometrically_admissible": K >= K_min_geometry,
                        "threshold_field_Vm": threshold,
                        "operating_factor": float(factor),
                        "operating_field_Vm": field,
                        "growth_rate_s-1": growth,
                        "e_folding_time_s": 1.0 / growth if growth > 0.0 else math.inf,
                        "formation_time_s": formation,
                        "coherent_fraction": float(coherent_fraction),
                        "coherent_power_W": power,
                        "charge_time_s": charge,
                        "total_onset_time_s": charge + formation,
                        "steady_field_Vm": steady_field,
                        "pulse_end_field_Vm": pulse_field,
                        "within_pulse": charge + formation <= pulse,
                    })
    return pd.DataFrame(rows)


def effective_ion_mass(Dplus_fraction: float, molecular_mass_u: float = 6.042) -> float:
    return (Dplus_fraction * 2.014 + (1.0 - Dplus_fraction) * molecular_mass_u) * atomic_mass


def double_layer_solution(
    alpha_hot: float,
    hot_temperature_eV: float,
    cold_temperature_eV: float,
    ion_mass_kg: float,
    ion_temperature_eV: float = 0.1,
) -> dict:
    """Two-Maxwellian, current-free, collisionless double-layer screening model.

    Electron fluxes over the potential barrier are balanced against the Bohm
    ion flux.  Temperatures and potential are both expressed in eV/volts.
    """
    alpha = float(alpha_hot)
    Tc = float(cold_temperature_eV)
    Th = float(hot_temperature_eV)
    Teff = 1.0 / ((1.0 - alpha) / Tc + alpha / Th)
    sound = math.sqrt(e * (Teff + 3.0 * ion_temperature_eV) / ion_mass_kg)
    cold_flux0 = (1.0 - alpha) * math.sqrt(e * Tc / (2.0 * pi * m_e))
    hot_flux0 = alpha * math.sqrt(e * Th / (2.0 * pi * m_e))

    def current_residual(phi_V: float) -> float:
        return cold_flux0 * math.exp(-phi_V / Tc) + hot_flux0 * math.exp(-phi_V / Th) - sound

    if current_residual(0.0) <= 0.0:
        return {"valid": False, "potential_V": math.nan}
    upper = max(100.0, 100.0 * Th)
    potential = brentq(current_residual, 0.0, upper)
    cold_flux = cold_flux0 * math.exp(-potential / Tc)
    hot_flux = hot_flux0 * math.exp(-potential / Th)
    escape_temperature = (cold_flux * Tc + hot_flux * Th) / (cold_flux + hot_flux)
    return {
        "valid": True,
        "potential_V": potential,
        "effective_temperature_eV": Teff,
        "sound_speed_ms": sound,
        "escaping_hot_flux_fraction": hot_flux / (cold_flux + hot_flux),
        "escaping_electron_temperature_eV": escape_temperature,
    }


def evaluate_double_layer(vector: np.ndarray, regime: pd.Series, cfg: dict) -> dict:
    dl = cfg["open_double_layer"]
    pulse = float(cfg["experiment"]["pulse_duration_s"])
    alpha = 10.0 ** float(vector[0])
    Th = 10.0 ** float(vector[1])
    area_ratio = 10.0 ** float(vector[2])
    Dplus = float(dl["Dplus_fraction"])
    mass = effective_ion_mass(Dplus, float(dl["molecular_remainder_mass_u"]))
    solution = double_layer_solution(
        alpha, Th, float(dl["cold_electron_temperature_eV"]), mass,
        float(dl["ion_temperature_eV"]),
    )
    if not solution["valid"]:
        return {"valid": False, "score": 0.0}
    area = pi * float(regime.radius_m) ** 2 * area_ratio
    hot_volume = area * float(dl["hot_zone_length_m"])
    ne = float(regime.electron_density_m3)
    Tc = float(dl["cold_electron_temperature_eV"])
    reservoir_power = 1.5 * alpha * ne * hot_volume * e * max(0.0, Th - Tc) / pulse
    ion_rate = ne * solution["sound_speed_ms"] * area
    # Conservative lower-bound power: replenish escaping energy and ion acceleration.
    exhaust_power = ion_rate * e * (
        solution["potential_V"] + 2.0 * solution["escaping_electron_temperature_eV"]
    )
    total_power = reservoir_power + exhaust_power
    Dplus_rate = Dplus * ion_rate
    energy = solution["potential_V"]
    return {
        "valid": True,
        "hot_fraction": alpha,
        "hot_temperature_eV": Th,
        "throat_area_ratio": area_ratio,
        "throat_radius_m": float(regime.radius_m) * math.sqrt(area_ratio),
        "potential_V": energy,
        "Dplus_energy_eV": energy,
        "D3plus_energy_per_deuteron_eV": energy / 3.0,
        "effective_temperature_eV": solution["effective_temperature_eV"],
        "sound_speed_ms": solution["sound_speed_ms"],
        "escaping_hot_flux_fraction": solution["escaping_hot_flux_fraction"],
        "hot_reservoir_power_W": reservoir_power,
        "exhaust_power_W": exhaust_power,
        "total_power_W": total_power,
        "Dplus_rate_s-1": Dplus_rate,
        "score": Dplus_rate,
    }


def optimize_double_layer(regime: pd.Series, cfg: dict) -> tuple[pd.DataFrame, dict]:
    dl = cfg["open_double_layer"]
    limit = float(cfg["experiment"]["generator_power_limit_W"])
    target = float(cfg["experiment"]["target_deuteron_energies_eV"][0])
    maximum_area_ratio = float(dl["throat_area_ratio_bounds"][1])
    bounds = [
        tuple(np.log10(dl["hot_fraction_bounds"])),
        tuple(np.log10(dl["hot_temperature_bounds_eV"])),
    ]

    def at_power_limited_area(x: np.ndarray) -> dict:
        trial = evaluate_double_layer(
            np.array([x[0], x[1], math.log10(maximum_area_ratio)]), regime, cfg
        )
        if not trial.get("valid", False):
            return trial
        scale = min(1.0, limit / max(trial["total_power_W"], 1.0e-30))
        area_ratio = maximum_area_ratio * scale
        return evaluate_double_layer(
            np.array([x[0], x[1], math.log10(area_ratio)]), regime, cfg
        )

    def objective(x: np.ndarray) -> float:
        result = at_power_limited_area(x)
        if not result.get("valid", False):
            return 1.0e12
        energy_penalty = max(0.0, (target + 0.1) / max(result["Dplus_energy_eV"], 1.0) - 1.0)
        power_penalty = max(0.0, result["total_power_W"] / limit - 1.0)
        # A screening optimum is only useful if it is on the feasible side of
        # both hard requirements.  The large penalty prevents a superficially
        # higher flux from buying a small violation of 1 keV or the power cap.
        return -math.log10(max(result["score"], 1.0)) + 1.0e6 * (energy_penalty**2 + power_penalty**2)

    optimum = differential_evolution(
        objective, bounds, seed=int(dl["optimizer_seed"]),
        maxiter=int(dl["optimizer_iterations"]), popsize=12, polish=True, tol=1.0e-8,
    )
    best = at_power_limited_area(optimum.x)
    best.update({"optimizer_success": bool(optimum.success), "optimizer_message": str(optimum.message)})
    reference_vectors = []
    for alpha in [1.0e-3, 0.01, 0.03, 0.05, 0.10]:
        for Th in [100.0, 300.0, 1000.0, 3000.0]:
            reference_vectors.append(np.log10([alpha, Th, 0.01]))
    rows = [evaluate_double_layer(x, regime, cfg) for x in reference_vectors]
    return pd.DataFrame(rows), best


def reflected_deuteron_energy_eV(shock_speed_ms: float) -> float:
    """Specular reflection from a moving electrostatic shock, upstream at rest."""
    return 2.0 * M_D * shock_speed_ms**2 / e


def evaluate_shock(vector: np.ndarray, regime: pd.Series, cfg: dict) -> dict:
    sh = cfg["counterstream_shock"]
    pulse = float(cfg["experiment"]["pulse_duration_s"])
    Te, Mach, log_area, Dplus = map(float, vector)
    area_ratio = 10.0**log_area
    mass = effective_ion_mass(Dplus)
    Ti = 0.1
    sound = math.sqrt(e * (Te + 3.0 * Ti) / mass)
    shock_speed = Mach * sound
    energy = reflected_deuteron_energy_eV(shock_speed)
    area = pi * float(regime.radius_m) ** 2 * area_ratio
    ne = float(regime.electron_density_m3)
    # Two equal counterflows, total density ne: sum of both kinetic-energy fluxes.
    beam_power = 0.5 * ne * area * mass * shock_speed**3
    volume = area * float(sh["interaction_length_m"])
    heating_power = 1.5 * ne * volume * e * max(0.0, Te - float(regime.electron_temperature_eV)) / pulse
    total_power = beam_power + heating_power
    reflected_rate = (
        float(sh["reflected_fraction_upper_bound"]) * Dplus * ne * area * shock_speed
    )
    return {
        "electron_temperature_eV": Te,
        "Mach": Mach,
        "Dplus_fraction": Dplus,
        "throat_area_ratio": area_ratio,
        "sound_speed_ms": sound,
        "shock_speed_ms": shock_speed,
        "Dplus_reflected_energy_eV": energy,
        "beam_power_W": beam_power,
        "electron_heating_power_W": heating_power,
        "total_power_W": total_power,
        "reflected_Dplus_rate_s-1": reflected_rate,
        "score": reflected_rate,
    }


def optimize_shock(regime: pd.Series, cfg: dict) -> dict:
    sh = cfg["counterstream_shock"]
    limit = float(cfg["experiment"]["generator_power_limit_W"])
    target = float(cfg["experiment"]["target_deuteron_energies_eV"][0])
    maximum_area_ratio = float(sh["throat_area_ratio_bounds"][1])
    bounds = [
        tuple(sh["electron_temperature_bounds_eV"]),
        tuple(sh["Mach_bounds"]),
        tuple(sh["Dplus_fraction_bounds"]),
    ]

    def at_power_limited_area(x: np.ndarray) -> dict:
        trial_vector = np.array([x[0], x[1], math.log10(maximum_area_ratio), x[2]])
        trial = evaluate_shock(trial_vector, regime, cfg)
        scale = min(1.0, limit / max(trial["total_power_W"], 1.0e-30))
        return evaluate_shock(
            np.array([x[0], x[1], math.log10(maximum_area_ratio * scale), x[2]]),
            regime, cfg,
        )

    def objective(x: np.ndarray) -> float:
        result = at_power_limited_area(x)
        ep = max(0.0, (target + 0.1) / max(result["Dplus_reflected_energy_eV"], 1.0) - 1.0)
        pp = max(0.0, result["total_power_W"] / limit - 1.0)
        return -math.log10(max(result["score"], 1.0)) + 1.0e6 * (ep**2 + pp**2)

    optimum = differential_evolution(
        objective, bounds, seed=int(sh["optimizer_seed"]),
        maxiter=int(sh["optimizer_iterations"]), popsize=12, polish=True, tol=1.0e-5,
    )
    best = at_power_limited_area(optimum.x)
    best.update({"optimizer_success": bool(optimum.success), "optimizer_message": str(optimum.message)})
    return best


def lower_hybrid_omega(B_T: float, electron_density_m3: float) -> float:
    """Cold-plasma lower-hybrid frequency with finite omega_pe correction."""
    omega_pe = math.sqrt(electron_density_m3 * e**2 / (epsilon_0 * m_e))
    omega_ce = e * B_T / m_e
    omega_ci = e * B_T / M_D
    return math.sqrt(omega_ce * omega_ci / (1.0 + omega_ce**2 / omega_pe**2))


def magnetic_field_for_lower_hybrid(omega_target: float, ne_m3: float, bounds_T: tuple[float, float]) -> float:
    residual = lambda field: lower_hybrid_omega(field, ne_m3) - omega_target
    if residual(bounds_T[0]) * residual(bounds_T[1]) > 0.0:
        return math.nan
    return brentq(residual, bounds_T[0], bounds_T[1])


def lower_hybrid_scan(regime: pd.Series, cfg: dict) -> pd.DataFrame:
    lh = cfg["lower_hybrid"]
    experiment = cfg["experiment"]
    pulse = float(experiment["pulse_duration_s"])
    length = float(lh["active_length_m"])
    volume = active_volume(regime, length)
    ne = float(regime.electron_density_m3)
    gamma = 0.5 * float(lh["damping_scale"]) * (
        float(regime["nu_en_s-1"]) + float(regime["exact_nu_in_effective_s-1"])
    )
    bounds = tuple(map(float, lh["magnetic_field_bounds_T"]))
    rows = []
    for target in experiment["target_deuteron_energies_eV"]:
        target = float(target)
        target_velocity = math.sqrt(2.0 * e * target / M_D)
        for preenergy in lh["preacceleration_energies_eV"]:
            preenergy = min(float(preenergy), target)
            prevelocity = math.sqrt(2.0 * e * preenergy / M_D)
            # Conservative trapping width: e*Phi >= m_D*(v_phi-v_pre)^2/2.
            trap_potential = (math.sqrt(target) - math.sqrt(preenergy)) ** 2
            for harmonic in lh["axial_harmonics"]:
                k = 2.0 * pi * int(harmonic) / length
                omega = k * target_velocity
                B = magnetic_field_for_lower_hybrid(omega, ne, bounds)
                if math.isnan(B):
                    continue
                field = k * trap_potential
                energy = mode_energy(field, volume)
                required_power = 2.0 * gamma * energy / max(1.0e-30, 1.0 - math.exp(-2.0 * gamma * pulse))
                coupling = required_power / float(regime.absorbed_power_W)
                charge_time_30 = pump_charge_time(
                    energy,
                    float(lh["maximum_coherent_fraction"]) * float(regime.absorbed_power_W),
                    gamma,
                )
                bounce = math.sqrt(e * k * field / M_D)
                bounce_periods = bounce * pulse / (2.0 * pi)
                ion_wavebreaking = M_D * omega * target_velocity / e
                electron_magnetization = (e * B / m_e) / float(regime["nu_en_s-1"])
                ion_magnetization = (e * B / M_D) / float(regime["exact_nu_in_effective_s-1"])
                feasible = (
                    coupling <= float(lh["maximum_coherent_fraction"])
                    and bounce_periods >= float(lh["minimum_bounce_periods"])
                    and field < ion_wavebreaking
                    and charge_time_30 <= pulse
                )
                rows.append({
                    "target_energy_eV": target,
                    "preacceleration_energy_eV": preenergy,
                    "harmonic": int(harmonic),
                    "wavenumber_m-1": k,
                    "wavelength_m": 2.0 * pi / k,
                    "phase_velocity_ms": target_velocity,
                    "frequency_Hz": omega / (2.0 * pi),
                    "magnetic_field_T": B,
                    "trapping_potential_V": trap_potential,
                    "wave_field_Vm": field,
                    "ion_wavebreaking_field_Vm": ion_wavebreaking,
                    "mode_energy_J": energy,
                    "damping_s-1": gamma,
                    "Q_lower_hybrid": omega / (2.0 * gamma),
                    "required_mode_power_W": required_power,
                    "required_fraction_of_absorbed_power": coupling,
                    "charge_time_at_30pct_s": charge_time_30,
                    "bounce_periods_in_pulse": bounce_periods,
                    "electron_magnetization": electron_magnetization,
                    "ion_magnetization": ion_magnetization,
                    "screening_feasible": feasible,
                })
    return pd.DataFrame(rows)


def nozzle_scan(regime: pd.Series, cfg: dict) -> pd.DataFrame:
    Te = float(regime.electron_temperature_eV)
    base = float(regime.ode_transition_Dplus_energy_ceiling_eV)
    rows = []
    for ratio in cfg["nozzle"]["expansion_ratios"]:
        ratio = float(ratio)
        ambipolar_gain = Te * math.log(ratio)
        rows.append({
            "expansion_ratio": ratio,
            "isothermal_ambipolar_gain_eV": ambipolar_gain,
            "energy_with_caviton_upper_bound_eV": base + ambipolar_gain,
            "ratio_needed_for_1keV_at_current_Te": math.exp((1000.0 - base) / Te),
        })
    return pd.DataFrame(rows)


def make_summary(regime: pd.Series, mi: pd.DataFrame, dl_best: dict, shock_best: dict,
                 lh: pd.DataFrame, nozzle: pd.DataFrame) -> dict:
    admissible = mi[mi.geometrically_admissible]
    best_lh_rows = lh[(lh.target_energy_eV == 1000.0)].sort_values(
        "required_fraction_of_absorbed_power"
    )
    best_lh = best_lh_rows.iloc[0].to_dict()
    caviton_energy = float(regime.ode_transition_Dplus_energy_ceiling_eV)
    # NRL-formulary expression: density is in cm^-3.  ln(Lambda)=14 is
    # representative for this dilute regime and is kept explicit here.
    density_cm3 = float(regime.electron_density_m3) * 1.0e-6
    coulomb_logarithm = 14.0
    coulomb_nuei = 2.91e-6 * density_cm3 * coulomb_logarithm / float(regime.electron_temperature_eV) ** 1.5
    coulomb_transfer = 2.0 * m_e / M_D * coulomb_nuei
    return {
        "scope": "screening bounds; not an open-boundary kinetic proof",
        "regime": cfg_value_dict(regime),
        "modulational_instability": {
            "natural_threshold_range_geometrically_admissible_Vm": [
                float(admissible.threshold_field_Vm.min()), float(admissible.threshold_field_Vm.max())
            ],
            "minimum_formation_time_s": float(admissible.formation_time_s.min()),
            "maximum_wavebreaking_field_Vm": float(regime.exact_wavebreaking_field_Vm),
            "conclusion": "spectral gating can delay onset, but cannot store a wavebreaking-scale field in the closed Zakharov branch",
        },
        "coulomb_electron_to_deuteron": {
            "coulomb_logarithm": coulomb_logarithm,
            "electron_ion_collision_frequency_s-1": coulomb_nuei,
            "energy_exchange_rate_s-1": coulomb_transfer,
            "energy_exchange_time_s": 1.0 / coulomb_transfer,
        },
        "open_double_layer_optimum": dl_best,
        "counterstream_shock_optimum": shock_best,
        "lower_hybrid_best_1keV": best_lh,
        "nozzle_maximum_screened_energy_eV": float(nozzle.energy_with_caviton_upper_bound_eV.max()),
        "recommended_cascade": {
            "stages": ["localized hot-electron source", "open double layer/nozzle preacceleration", "driven lower-hybrid phase trapping"],
            "preacceleration_eV": 200.0,
            "final_target_eV": 1000.0,
            "status": "energetically screen-feasible; requires open-boundary electromagnetic PIC and experiment",
        },
        "caviton_alone_Dplus_ceiling_eV": caviton_energy,
    }


def cfg_value_dict(regime: pd.Series) -> dict:
    return {
        "pressure_Pa": float(regime.pressure_Pa),
        "electron_temperature_eV": float(regime.electron_temperature_eV),
        "electron_density_m3": float(regime.electron_density_m3),
        "radius_m": float(regime.radius_m),
        "chamber_length_m": float(regime.chamber_length_m),
    }


def synergy_table(regime: pd.Series, dl_best: dict, shock_best: dict, lh: pd.DataFrame) -> pd.DataFrame:
    def lh_best(pre: float) -> pd.Series:
        rows = lh[(lh.target_energy_eV == 1000.0) & np.isclose(lh.preacceleration_energy_eV, pre)]
        return rows.sort_values("required_fraction_of_absorbed_power").iloc[0]

    baseline = float(regime.ode_transition_Dplus_energy_ceiling_eV)
    lh0, lh60, lh200 = lh_best(0.0), lh_best(60.26035), lh_best(200.0)
    return pd.DataFrame([
        {"mechanism": "closed caviton", "screened_energy_eV": baseline, "additional_power_W": 0.0, "status": "calculated upper bound"},
        {"mechanism": "nozzle + closed caviton", "screened_energy_eV": baseline + float(regime.electron_temperature_eV) * math.log(100.0), "additional_power_W": 0.0, "status": "ideal isothermal upper bound"},
        {"mechanism": "open double layer", "screened_energy_eV": dl_best["Dplus_energy_eV"], "additional_power_W": dl_best["total_power_W"], "status": "current-balance screen"},
        {"mechanism": "counterstream shock", "screened_energy_eV": shock_best["Dplus_reflected_energy_eV"], "additional_power_W": shock_best["total_power_W"], "status": "reflection upper bound"},
        {"mechanism": "lower hybrid from thermal", "screened_energy_eV": 1000.0, "additional_power_W": float(lh0.required_mode_power_W), "status": "phase-trapping screen"},
        {"mechanism": "caviton 60 eV + lower hybrid", "screened_energy_eV": 1000.0, "additional_power_W": float(lh60.required_mode_power_W), "status": "phase-trapping screen"},
        {"mechanism": "open DL 200 eV + lower hybrid", "screened_energy_eV": 1000.0, "additional_power_W": float(lh200.required_mode_power_W), "status": "preferred cascade screen"},
    ])


def make_figure(mi: pd.DataFrame, synergy: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    subset = mi[(mi.operating_factor == 1.5) & (mi.coherent_fraction == 0.03)]
    for q, group in subset.groupby("Q_multiplier"):
        axes[0].plot(group.K_over_k, group.threshold_field_Vm / 1e3, label=f"Q×{q:g}")
    axes[0].axvline(float(subset[subset.geometrically_admissible].K_over_k.min()), color="k", ls="--", lw=1, label="предел камеры")
    axes[0].set(xlabel=r"$K/k$", ylabel=r"$E_{th}$, кВ/м", title="Управляемость порога МН")
    axes[0].legend(fontsize=8)
    plot = synergy.copy()
    axes[1].barh(plot.mechanism, plot.screened_energy_eV / 1e3, color=["#767676", "#8aa1b1", "#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"])
    axes[1].axvline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="расчётная энергия D⁺, кэВ", title="Скрининг механизмов (не PIC-доказательство)")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_level46.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.config.resolve().parent
    cfg = load_yaml(args.config)
    regime = selected_regime(root, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    mi = mi_control_scan(regime, cfg)
    dl_scan, dl_best = optimize_double_layer(regime, cfg)
    shock_best = optimize_shock(regime, cfg)
    lh = lower_hybrid_scan(regime, cfg)
    nozzle = nozzle_scan(regime, cfg)
    synergy = synergy_table(regime, dl_best, shock_best, lh)
    summary = make_summary(regime, mi, dl_best, shock_best, lh, nozzle)

    mi.to_csv(args.output / "level46_MI_control.csv", index=False)
    dl_scan.to_csv(args.output / "level46_double_layer_scan.csv", index=False)
    pd.DataFrame([dl_best]).to_csv(args.output / "level46_double_layer_optimum.csv", index=False)
    pd.DataFrame([shock_best]).to_csv(args.output / "level46_counterstream_optimum.csv", index=False)
    lh.to_csv(args.output / "level46_lower_hybrid.csv", index=False)
    nozzle.to_csv(args.output / "level46_nozzle.csv", index=False)
    synergy.to_csv(args.output / "level46_synergy.csv", index=False)
    with (args.output / "level46_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    make_figure(mi, synergy, root / "figures" / "21_level46_mechanism_screen.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
