#!/usr/bin/env python3
"""Audit and constrained extrema search for the level-1--5 caviton chain.

The optimizer does not claim a new kinetic acceleration mechanism. It finds
the largest wave-breaking/ponderomotive ceiling that remains compatible with
the screening global balance, wave-survival criteria, chamber geometry and a
calibrated reduced pump-collapse ODE. Its extended-window solutions are design
hypotheses that must be revalidated with traceable cross sections and kinetic
electrons.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import atomic_mass, e, epsilon_0, k as k_B, m_e, pi
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, differential_evolution

from plasma_level1 import (
    CrossSection,
    Geometry,
    Gas,
    electron_ion_collision_frequency,
    ion_neutral_rate,
    load_cross_sections,
    load_config as load_level1_config,
    neutral_density,
    plasma_parameter,
)
from plasma_level2 import load_yaml as load_level2_yaml, solve_mode_point
from plasma_level3 import build_coefficients, composition_by_name, threshold_result


@dataclass(frozen=True)
class RateCache:
    temperature_eV: np.ndarray
    values: dict[str, np.ndarray]

    def rate(self, name: str, temperature_eV: float) -> float:
        return float(np.interp(temperature_eV, self.temperature_eV, self.values[name]))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def make_rate_cache(cross_sections: dict[str, CrossSection]) -> RateCache:
    temperature = np.linspace(1.5, 20.0, 480)
    names = ["e_momentum", "ionization", "dissociation", "excitation"]
    values = {
        name: np.array([cross_sections[name].maxwell_rate(item) for item in temperature])
        for name in names
    }
    return RateCache(temperature, values)


def fast_background(
    pressure_Pa: float,
    absorbed_power_W: float,
    geometry: Geometry,
    gas: Gas,
    cross_sections: dict[str, CrossSection],
    rates: RateCache,
    level1_cfg: dict,
    ion_temperature_eV: float,
) -> dict | None:
    nn = neutral_density(pressure_Pa, gas.neutral_temperature_K)

    def residual(temperature_eV: float) -> float:
        production = nn * rates.rate("ionization", temperature_eV)
        nu_in, _, _ = ion_neutral_rate(
            nn, temperature_eV, ion_temperature_eV, gas, cross_sections["ion_momentum"]
        )
        sound = math.sqrt(
            e * (temperature_eV + 3.0 * ion_temperature_eV) / gas.ion_mass_kg
        )
        nu_bohm = geometry.wall_transmission_factor * sound * geometry.area_m2 / geometry.volume_m3
        diffusion = sound**2 / max(nu_in, sound / min(geometry.radius_m, geometry.length_m))
        nu_diff = diffusion * geometry.inverse_diffusion_length2_m2
        loss = min(nu_bohm, nu_diff)
        return math.log(max(production, 1.0e-300) / max(loss, 1.0e-300))

    temperature_grid = rates.temperature_eV
    values = np.array([residual(item) for item in temperature_grid])
    changes = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
    if not len(changes):
        return None
    index = int(changes[0])
    Te = brentq(residual, temperature_grid[index], temperature_grid[index + 1], xtol=1.0e-8)
    electron_rates = {
        name: rates.rate(name, Te)
        for name in ["e_momentum", "ionization", "dissociation", "excitation"]
    }
    nu_in, relative_energy, K_in = ion_neutral_rate(
        nn, Te, ion_temperature_eV, gas, cross_sections["ion_momentum"]
    )
    sound = math.sqrt(e * (Te + 3.0 * ion_temperature_eV) / gas.ion_mass_kg)
    nu_bohm = geometry.wall_transmission_factor * sound * geometry.area_m2 / geometry.volume_m3
    diffusion = sound**2 / max(nu_in, sound / min(geometry.radius_m, geometry.length_m))
    nu_diff = diffusion * geometry.inverse_diffusion_length2_m2
    nu_loss = min(nu_bohm, nu_diff)
    global_cfg = level1_cfg["global_model"]
    energy_per_pair = (
        float(global_cfg["ionization_energy_eV"])
        + electron_rates["dissociation"] / electron_rates["ionization"]
        * float(global_cfg["dissociation_energy_eV"])
        + electron_rates["excitation"] / electron_rates["ionization"]
        * float(global_cfg["effective_excitation_energy_eV"])
        + float(global_cfg["wall_energy_loss_Te_factor"]) * Te
    )
    ne = absorbed_power_W / (
        geometry.volume_m3 * nn * electron_rates["ionization"] * energy_per_pair * e
    )
    lambda_De = math.sqrt(epsilon_0 * Te / (ne * e))
    return {
        "electron_temperature_eV": Te,
        "electron_density_m3": ne,
        "neutral_density_m3": nn,
        "ionization_fraction": ne / nn,
        "lambda_De_m": lambda_De,
        "plasma_parameter": plasma_parameter(ne, lambda_De),
        "nu_en_s-1": nn * electron_rates["e_momentum"],
        "nu_in_background_s-1": nu_in,
        "nu_loss_s-1": nu_loss,
        "ion_relative_energy_eV": relative_energy,
        "K_in_m3s": K_in,
        "energy_per_pair_eV": energy_per_pair,
    }


def wavebreaking_ponderomotive_bound_eV(temperature_eV: float, kappa: float) -> float:
    """Cold wave-breaking electron ponderomotive potential U_p/e."""
    return temperature_eV * (1.0 + 3.0 * kappa**2) / (4.0 * kappa**2)


def quartic_growth(
    field_Vm: float,
    modulation_K_m1: float,
    wave: dict,
) -> float:
    nu_s = (
        wave["acoustic_kinetic_damping_at_carrier_s-1"]
        * modulation_K_m1 / wave["carrier_k_m-1"]
        + 0.5 * (
            wave["nu_in_effective_s-1"]
            + (modulation_K_m1 * wave["lambda_De_m"]) ** 2
            / (1.0 + (modulation_K_m1 * wave["lambda_De_m"]) ** 2)
            * wave["nu_en_s-1"]
        )
    )
    acoustic = np.array([1.0, 2.0 * nu_s, (wave["sound_speed_ms"] * modulation_K_m1) ** 2])
    envelope = np.array([
        1.0,
        2.0 * wave["gamma_langmuir_s-1"],
        wave["gamma_langmuir_s-1"] ** 2
        + (wave["dispersion_P_m2s"] * modulation_K_m1**2) ** 2,
    ])
    polynomial = np.polymul(acoustic, envelope)
    coupling = (
        2.0 * wave["density_coupling_A_m3s"] * wave["ponderomotive_B"]
        * wave["dispersion_P_m2s"] * field_Vm**2 * modulation_K_m1**4
    )
    polynomial[-1] -= coupling
    return float(np.max(np.roots(polynomial).real))


def analytic_wave_state(
    background: dict,
    kappa: float,
    interaction_length_m: float,
    ion_temperature_eV: float,
    effective_ion_mass_kg: float,
    ion_neutral_rate_scale: float,
    wave_cfg: dict,
) -> dict | None:
    ne = background["electron_density_m3"]
    Te = background["electron_temperature_eV"]
    lambda_De = background["lambda_De_m"]
    k_m1 = kappa / lambda_De
    vte = math.sqrt(e * Te / m_e)
    omega_pe = math.sqrt(ne * e**2 / (epsilon_0 * m_e))
    omega = omega_pe * math.sqrt(1.0 + 3.0 * kappa**2)
    phase = omega / k_m1
    group = 3.0 * kappa * vte / math.sqrt(1.0 + 3.0 * kappa**2)
    landau = math.sqrt(pi / 8.0) * omega_pe / kappa**3 * math.exp(
        -1.0 / (2.0 * kappa**2) - 1.5
    )
    nu_ei = electron_ion_collision_frequency(ne, Te)
    gamma_langmuir = landau + 0.5 * (background["nu_en_s-1"] + nu_ei)
    L_langmuir = group / gamma_langmuir
    Q_langmuir = omega / (2.0 * gamma_langmuir)

    sound = math.sqrt(e * (Te + 3.0 * ion_temperature_eV) / effective_ion_mass_kg)
    omega_s = k_m1 * sound / math.sqrt(1.0 + kappa**2)
    ratio = ion_temperature_eV / Te
    kinetic_ratio = math.sqrt(pi / 8.0) * (
        math.sqrt(m_e / effective_ion_mass_kg)
        + ratio**1.5 * math.exp(-1.0 / max(2.0 * ratio, 1.0e-12))
    )
    gamma_s_kinetic = kinetic_ratio * omega_s
    nu_in = ion_neutral_rate_scale * background["nu_in_background_s-1"]
    polarization = kappa**2 / (1.0 + kappa**2)
    gamma_s = gamma_s_kinetic + 0.5 * (nu_in + polarization * background["nu_en_s-1"])
    group_s = sound / (1.0 + kappa**2) ** 1.5
    L_acoustic = group_s / gamma_s
    Q_acoustic = omega_s / (2.0 * gamma_s)
    wavelength = 2.0 * pi / k_m1
    caviton_scale = float(wave_cfg["caviton_carrier_wavelengths"]) * wavelength

    # Bohm-Gross dispersion. The exact kinetic GVD is used in levels 2-4;
    # this inexpensive form is deliberately retained only for the broad search.
    dispersion_P = (
        1.5 * omega_pe * lambda_De**2 / (1.0 + 3.0 * kappa**2) ** 1.5
    )
    A = omega / (2.0 * ne)
    B = epsilon_0 / (4.0 * effective_ion_mass_kg)
    K_min = 2.0 * pi / interaction_length_m
    K_max = float(wave_cfg["modulation_to_carrier_k_maximum"]) * k_m1
    count = int(math.floor(K_max / K_min))
    if count < 1:
        return None
    modes = K_min * np.arange(1, count + 1, dtype=float)
    thresholds = np.sqrt(
        sound**2 * (gamma_langmuir**2 + dispersion_P**2 * modes**4)
        / (2.0 * A * B * dispersion_P * modes**2)
    )
    threshold_index = int(np.argmin(thresholds))
    threshold = float(thresholds[threshold_index])
    operating = float(wave_cfg["operating_field_over_threshold"]) * threshold
    growths = np.array([
        quartic_growth(operating, mode, {
            "acoustic_kinetic_damping_at_carrier_s-1": gamma_s_kinetic,
            "carrier_k_m-1": k_m1,
            "lambda_De_m": lambda_De,
            "nu_in_effective_s-1": nu_in,
            "nu_en_s-1": background["nu_en_s-1"],
            "sound_speed_ms": sound,
            "gamma_langmuir_s-1": gamma_langmuir,
            "dispersion_P_m2s": dispersion_P,
            "density_coupling_A_m3s": A,
            "ponderomotive_B": B,
        })
        for mode in modes
    ])
    fastest_index = int(np.argmax(growths))
    growth = float(growths[fastest_index])
    if growth <= 0.0:
        return None
    e_folds = math.log(
        float(wave_cfg["onset_depletion_fraction"])
        / float(wave_cfg["density_seed_fraction"])
    )
    selected_K = float(modes[fastest_index])
    return {
        "carrier_k_m-1": k_m1,
        "carrier_frequency_Hz": omega / (2.0 * pi),
        "phase_velocity_ms": phase,
        "group_velocity_ms": group,
        "gamma_langmuir_s-1": gamma_langmuir,
        "L_damp_langmuir_m": L_langmuir,
        "Q_langmuir": Q_langmuir,
        "sound_speed_ms": sound,
        "acoustic_kinetic_damping_at_carrier_s-1": gamma_s_kinetic,
        "nu_in_effective_s-1": nu_in,
        "L_damp_acoustic_m": L_acoustic,
        "Q_acoustic": Q_acoustic,
        "wavelength_m": wavelength,
        "caviton_scale_m": caviton_scale,
        "dispersion_P_m2s": dispersion_P,
        "density_coupling_A_m3s": A,
        "ponderomotive_B": B,
        "threshold_field_Vm": threshold,
        "operating_field_Vm": operating,
        "growth_rate_s-1": growth,
        "formation_time_s": e_folds / growth,
        "selected_modulation_K_m-1": selected_K,
        "initial_caviton_half_width_m": pi / selected_K,
        "wavebreaking_field_Vm": m_e * omega * phase / e,
        "wavebreaking_ponderomotive_eV": wavebreaking_ponderomotive_bound_eV(Te, kappa),
        "available_modulation_modes": count,
    }


def pump_charge_time(
    required_energy_J: float,
    mode_power_W: float,
    gamma_s1: float,
) -> float:
    steady = mode_power_W / (2.0 * gamma_s1)
    if required_energy_J >= steady or mode_power_W <= 0.0:
        return math.inf
    return -math.log1p(-required_energy_J / steady) / (2.0 * gamma_s1)


def constraint_ratio(value: float, limit: float, lower: bool) -> float:
    value = float(value)
    limit = float(limit)
    if lower:
        return max(0.0, limit / max(value, 1.0e-300) - 1.0)
    return max(0.0, value / max(limit, 1.0e-300) - 1.0)


def evaluate_candidate(
    vector: np.ndarray,
    search_cfg: dict,
    cfg: dict,
    level1_cfg: dict,
    cross_sections: dict[str, CrossSection],
    rates: RateCache,
) -> dict:
    pressure, power, radius, chamber_length, wall, kappa, interaction = vector
    plasma_cfg = cfg["plasma"]
    geometry = Geometry(radius, chamber_length, interaction, wall)
    gas = Gas(
        float(plasma_cfg["neutral_temperature_K"]),
        float(plasma_cfg["background_ion_mass_u"]) * atomic_mass,
        float(plasma_cfg["neutral_mass_u"]) * atomic_mass,
    )
    background = fast_background(
        pressure, power, geometry, gas, cross_sections, rates, level1_cfg,
        float(plasma_cfg["ion_temperature_eV"]),
    )
    if background is None:
        return {"feasible": False, "penalty": 1.0e6, "objective_eV": 0.0}
    wave = analytic_wave_state(
        background,
        kappa,
        interaction,
        float(plasma_cfg["ion_temperature_eV"]),
        float(plasma_cfg["effective_molecular_ion_mass_u"]) * atomic_mass,
        float(plasma_cfg["molecular_ion_neutral_rate_scale"]),
        cfg["wave"],
    )
    if wave is None:
        return {"feasible": False, "penalty": 1.0e6, "objective_eV": 0.0}

    engineering = cfg["engineering"]
    wave_cfg = cfg["wave"]
    ode_cfg = cfg["pump_collapse_ode"]
    violations: dict[str, float] = {}
    Te_bounds = engineering["electron_temperature_bounds_eV"]
    ne_bounds = engineering["electron_density_bounds_m3"]
    violations["Te_low"] = constraint_ratio(background["electron_temperature_eV"], Te_bounds[0], True)
    violations["Te_high"] = constraint_ratio(background["electron_temperature_eV"], Te_bounds[1], False)
    violations["ne_low"] = constraint_ratio(background["electron_density_m3"], ne_bounds[0], True)
    violations["ne_high"] = constraint_ratio(background["electron_density_m3"], ne_bounds[1], False)
    violations["ionization"] = constraint_ratio(
        background["ionization_fraction"], float(wave_cfg["maximum_ionization_fraction"]), False
    )
    violations["plasma_parameter"] = constraint_ratio(
        background["plasma_parameter"], float(wave_cfg["minimum_plasma_parameter"]), True
    )
    generator = engineering["generator_power_bounds_W"]
    absorbed = engineering["absorbed_fraction_bounds"]
    generator_low = power / float(absorbed[1])
    generator_high = power / float(absorbed[0])
    generator_overlap = max(generator_low, float(generator[0])) <= min(generator_high, float(generator[1]))
    violations["generator"] = 0.0 if generator_overlap else 1.0
    violations["caviton_fits_radius"] = constraint_ratio(wave["caviton_scale_m"], radius, False)
    violations["caviton_fits_interaction"] = constraint_ratio(
        wave["caviton_scale_m"], interaction, False
    )
    violations["interaction_fits_chamber"] = constraint_ratio(interaction, chamber_length, False)
    violations["langmuir_survival"] = constraint_ratio(
        wave["L_damp_langmuir_m"],
        float(wave_cfg["langmuir_damping_lengths_per_interaction"]) * interaction,
        True,
    )
    violations["acoustic_survival"] = constraint_ratio(
        wave["L_damp_acoustic_m"],
        float(wave_cfg["acoustic_damping_lengths_per_caviton"]) * wave["caviton_scale_m"],
        True,
    )
    violations["langmuir_Q"] = constraint_ratio(
        wave["Q_langmuir"], float(wave_cfg["minimum_langmuir_Q"]), True
    )
    violations["acoustic_Q"] = constraint_ratio(
        wave["Q_acoustic"], float(wave_cfg["minimum_acoustic_Q"]), True
    )
    mode_power = float(ode_cfg["coherent_mode_fraction"]) * power
    initial_width = wave["initial_caviton_half_width_m"]
    peak_field = float(wave_cfg["nonlinear_onset_peak_gain"]) * wave["operating_field_Vm"]
    required_energy = (
        epsilon_0 * float(ode_cfg["initial_shape_factor"]) * initial_width**2
        * interaction * peak_field**2 / 4.0
    )
    charge_time = pump_charge_time(required_energy, mode_power, wave["gamma_langmuir_s-1"])
    total_onset = charge_time + wave["formation_time_s"]
    violations["pump_energy"] = 0.0 if math.isfinite(charge_time) else 10.0
    violations["pulse_time"] = constraint_ratio(
        total_onset, float(ode_cfg["pulse_duration_s"]), False
    ) if math.isfinite(total_onset) else 10.0

    pressure_potential = background["electron_temperature_eV"] * math.log(
        1.0 / float(ode_cfg["density_floor_fraction"])
    )
    sheath_potential = (
        float(ode_cfg["sheath_drop_Te_multiplier"]) * background["electron_temperature_eV"]
    )
    objective = wave["wavebreaking_ponderomotive_eV"] + pressure_potential + sheath_potential
    penalty = float(sum(value**2 for value in violations.values()))
    feasible = penalty <= 1.0e-12
    return {
        "search": search_cfg["name"],
        "feasible": feasible,
        "penalty": penalty,
        "objective_eV": objective,
        "pressure_Pa": pressure,
        "absorbed_power_W": power,
        "radius_m": radius,
        "chamber_length_m": chamber_length,
        "wall_transmission_factor": wall,
        "k_lambda_De": kappa,
        "interaction_length_m": interaction,
        **background,
        **wave,
        "coherent_mode_power_W": mode_power,
        "generator_power_interval_W_low": generator_low,
        "generator_power_interval_W_high": generator_high,
        "required_onset_energy_J": required_energy,
        "pump_charge_time_s": charge_time,
        "total_onset_time_s": total_onset,
        "electron_pressure_potential_bound_eV": pressure_potential,
        "sheath_potential_bound_eV": sheath_potential,
        "Dplus_energy_ceiling_eV": objective,
        "D2plus_energy_per_deuteron_ceiling_eV": objective / 2.0,
        "D3plus_energy_per_deuteron_ceiling_eV": objective / 3.0,
        **{f"violation_{name}": value for name, value in violations.items()},
    }


def bounds_for_search(search: dict) -> list[tuple[float, float]]:
    names = [
        "pressure_Pa", "absorbed_power_W", "radius_m", "chamber_length_m",
        "wall_transmission_factor", "k_lambda_De", "interaction_length_m",
    ]
    bounds = []
    for name in names:
        low, high = map(float, search[name])
        if math.isclose(low, high):
            span = max(abs(low) * 1.0e-9, 1.0e-12)
            bounds.append((low, low + span))
        else:
            bounds.append((low, high))
    return bounds


def optimize_search(
    search: dict,
    cfg: dict,
    level1_cfg: dict,
    cross_sections: dict[str, CrossSection],
    rates: RateCache,
) -> tuple[dict, pd.DataFrame]:
    settings = cfg["optimizer"]
    evaluations: list[dict] = []

    def objective(vector: np.ndarray) -> float:
        result = evaluate_candidate(vector, search, cfg, level1_cfg, cross_sections, rates)
        evaluations.append(result)
        pulse = float(cfg["pump_collapse_ode"]["pulse_duration_s"])
        onset = float(result.get("total_onset_time_s", math.inf))
        time_margin = max(1.0e-6, 1.0 - onset / pulse) if math.isfinite(onset) else 1.0e-6
        score = max(result["objective_eV"] * math.sqrt(time_margin), 1.0e-12)
        return -math.log10(score) + 1.0e3 * result["penalty"]

    result = differential_evolution(
        objective,
        bounds_for_search(search),
        seed=int(settings["seed"]),
        maxiter=int(settings["maximum_iterations"]),
        popsize=int(settings["population_size"]),
        tol=float(settings["tolerance"]),
        polish=bool(settings["polish"]),
        updating="immediate",
        workers=1,
    )
    terminal = evaluate_candidate(result.x, search, cfg, level1_cfg, cross_sections, rates)
    evaluations.append(terminal)
    frame = pd.DataFrame(evaluations)
    feasible = frame[frame.feasible == True]
    if feasible.empty:
        best = terminal
    else:
        pulse = float(cfg["pump_collapse_ode"]["pulse_duration_s"])
        feasible = feasible.copy()
        feasible["time_margin"] = np.maximum(
            0.0, 1.0 - feasible.total_onset_time_s / pulse
        )
        feasible["collapse_screening_score"] = (
            feasible.objective_eV * np.sqrt(feasible.time_margin)
        )
        pool_indices = set(feasible.nlargest(20, "objective_eV").index)
        pool_indices.update(feasible.nlargest(30, "collapse_screening_score").index)
        pool_indices.update(feasible.nsmallest(10, "total_onset_time_s").index)
        best_row = None
        best_transition_score = -math.inf
        for index in pool_indices:
            candidate = feasible.loc[index].to_dict()
            history = integrate_best_ode(candidate, cfg)
            transition_up = float(history.iloc[-1].electron_ponderomotive_potential_eV)
            transition_total = (
                transition_up
                + float(candidate["electron_pressure_potential_bound_eV"])
                + float(candidate["sheath_potential_bound_eV"])
            )
            transition_score = transition_total / 3.0
            if transition_score > best_transition_score:
                best_transition_score = transition_score
                best_row = feasible.loc[index]
        assert best_row is not None
        names = [
            "pressure_Pa", "absorbed_power_W", "radius_m", "chamber_length_m",
            "wall_transmission_factor", "k_lambda_De", "interaction_length_m",
        ]
        best = evaluate_candidate(
            best_row[names].to_numpy(float), search, cfg, level1_cfg,
            cross_sections, rates,
        )
        best["screening_ode_D3plus_energy_per_deuteron_ceiling_eV"] = best_transition_score
    best["optimizer_success"] = bool(result.success)
    best["optimizer_message"] = str(result.message)
    best["optimizer_function_evaluations"] = int(result.nfev)
    return best, frame


def exact_validate_candidate(root: Path, best: dict, cfg: dict) -> dict:
    """Re-evaluate the optimum with the full level-2 kinetic roots and GVD."""
    if not best.get("feasible", False):
        return {**best, "exact_validation_passed": False, "exact_validation_error": "screening optimum infeasible"}
    try:
        level2_cfg = load_level2_yaml(root / "config_level2.yaml")
        composition = composition_by_name(level2_cfg, "molecular_nominal")
        background = {
            "pressure_Pa": best["pressure_Pa"],
            "absorbed_power_W": best["absorbed_power_W"],
            "electron_density_m3": best["electron_density_m3"],
            "electron_temperature_eV": best["electron_temperature_eV"],
            "nu_en_scaled_s-1": best["nu_en_s-1"],
            "nu_in_scaled_s-1": best["nu_in_background_s-1"],
        }
        Ti = float(cfg["plasma"]["ion_temperature_eV"])
        kappa = float(best["k_lambda_De"])
        langmuir = solve_mode_point(
            "langmuir", background, Ti, kappa, composition, level2_cfg["solver"]
        )
        acoustic = solve_mode_point(
            "ion_acoustic_fast", background, Ti, kappa, composition, level2_cfg["solver"]
        )
        coefficients = build_coefficients(
            background, Ti, kappa, composition, langmuir, acoustic, level2_cfg
        )
        wave_cfg = cfg["wave"]
        threshold_cfg = {
            "interaction_length_m": best["interaction_length_m"],
            "maximum_modulation_to_carrier_k": wave_cfg["modulation_to_carrier_k_maximum"],
            "discrete_modulation_modes": True,
            "operating_field_factor": wave_cfg["operating_field_over_threshold"],
            "density_seed_fraction": wave_cfg["density_seed_fraction"],
            "caviton_onset_depletion_fraction": wave_cfg["onset_depletion_fraction"],
        }
        threshold = threshold_result(coefficients, threshold_cfg)
        omega = float(langmuir["omega_real_rad_s"])
        phase = float(langmuir["phase_velocity_ms"])
        wavebreaking = m_e * omega * phase / e
        up = e * wavebreaking**2 / (4.0 * m_e * omega**2)
        carrier_caviton = (
            float(wave_cfg["caviton_carrier_wavelengths"]) * float(langmuir["wavelength_m"])
        )
        ode_cfg = cfg["pump_collapse_ode"]
        width = float(threshold["caviton_half_width_m"])
        peak = float(wave_cfg["nonlinear_onset_peak_gain"]) * float(threshold["operating_field_Vm"])
        required = (
            epsilon_0 * float(ode_cfg["initial_shape_factor"]) * width**2
            * float(best["interaction_length_m"]) * peak**2 / 4.0
        )
        charge = pump_charge_time(
            required, float(best["coherent_mode_power_W"]),
            abs(float(langmuir["gamma_total_s-1"])),
        )
        onset = charge + float(threshold["linear_formation_time_s"])
        exact_checks = {
            "langmuir_survival": float(langmuir["L_damp_amplitude_m"])
            >= float(wave_cfg["langmuir_damping_lengths_per_interaction"])
            * float(best["interaction_length_m"]),
            "acoustic_survival": float(acoustic["L_damp_amplitude_m"])
            >= float(wave_cfg["acoustic_damping_lengths_per_caviton"]) * carrier_caviton,
            "langmuir_Q": float(langmuir["quality_factor"]) >= float(wave_cfg["minimum_langmuir_Q"]),
            "acoustic_Q": float(acoustic["quality_factor"]) >= float(wave_cfg["minimum_acoustic_Q"]),
            "caviton_fits_radius": carrier_caviton <= float(best["radius_m"]),
            "caviton_fits_interaction": carrier_caviton <= float(best["interaction_length_m"]),
            "pump_reaches_onset": math.isfinite(charge)
            and onset <= float(ode_cfg["pulse_duration_s"]),
        }
        pressure = float(best["electron_pressure_potential_bound_eV"])
        sheath = float(best["sheath_potential_bound_eV"])
        return {
            **best,
            "exact_validation_passed": bool(all(exact_checks.values())),
            "exact_validation_error": "",
            "exact_carrier_frequency_Hz": omega / (2.0 * pi),
            "exact_carrier_k_m-1": float(langmuir["k_m-1"]),
            "exact_phase_velocity_ms": phase,
            "exact_group_velocity_ms": float(langmuir["group_velocity_ms"]),
            "exact_gamma_langmuir_s-1": abs(float(langmuir["gamma_total_s-1"])),
            "exact_L_damp_langmuir_m": float(langmuir["L_damp_amplitude_m"]),
            "exact_Q_langmuir": float(langmuir["quality_factor"]),
            "exact_L_damp_acoustic_m": float(acoustic["L_damp_amplitude_m"]),
            "exact_Q_acoustic": float(acoustic["quality_factor"]),
            "exact_carrier_caviton_scale_m": carrier_caviton,
            "exact_dispersion_P_m2s": coefficients.dispersion_P_m2s,
            "exact_density_coupling_A_m3s": coefficients.density_coupling_A_m3s,
            "exact_ponderomotive_B": coefficients.ponderomotive_B,
            "exact_sound_speed_ms": coefficients.acoustic_speed_ms,
            "exact_acoustic_kinetic_damping_at_carrier_s-1": coefficients.acoustic_kinetic_damping_at_carrier_s1,
            "exact_nu_in_effective_s-1": coefficients.ion_neutral_damping_frequency_s1,
            "exact_threshold_field_Vm": float(threshold["threshold_field_Vm"]),
            "exact_operating_field_Vm": float(threshold["operating_field_Vm"]),
            "exact_growth_rate_s-1": float(threshold["maximum_growth_rate_s-1"]),
            "exact_formation_time_s": float(threshold["linear_formation_time_s"]),
            "exact_selected_modulation_K_m-1": float(threshold["operating_modulation_K_m-1"]),
            "exact_initial_caviton_half_width_m": width,
            "exact_wavebreaking_field_Vm": wavebreaking,
            "exact_wavebreaking_ponderomotive_eV": up,
            "exact_required_onset_energy_J": required,
            "exact_pump_charge_time_s": charge,
            "exact_total_onset_time_s": onset,
            "exact_convective_formation_length_m": float(langmuir["group_velocity_ms"])
            * float(threshold["linear_formation_time_s"]),
            "exact_recirculation_passes": float(langmuir["group_velocity_ms"])
            * float(threshold["linear_formation_time_s"])
            / float(best["interaction_length_m"]),
            "exact_Dplus_energy_ceiling_eV": up + pressure + sheath,
            "exact_D2plus_energy_per_deuteron_ceiling_eV": (up + pressure + sheath) / 2.0,
            "exact_D3plus_energy_per_deuteron_ceiling_eV": (up + pressure + sheath) / 3.0,
            **{f"exact_check_{name}": value for name, value in exact_checks.items()},
        }
    except Exception as exc:
        return {
            **best,
            "exact_validation_passed": False,
            "exact_validation_error": repr(exc),
        }


def integrate_best_ode(best: dict, cfg: dict) -> pd.DataFrame:
    ode_cfg = cfg["pump_collapse_ode"]
    wave = dict(best)
    exact_mapping = {
        "carrier_k_m-1": "exact_carrier_k_m-1",
        "carrier_frequency_Hz": "exact_carrier_frequency_Hz",
        "gamma_langmuir_s-1": "exact_gamma_langmuir_s-1",
        "dispersion_P_m2s": "exact_dispersion_P_m2s",
        "density_coupling_A_m3s": "exact_density_coupling_A_m3s",
        "ponderomotive_B": "exact_ponderomotive_B",
        "sound_speed_ms": "exact_sound_speed_ms",
        "acoustic_kinetic_damping_at_carrier_s-1": "exact_acoustic_kinetic_damping_at_carrier_s-1",
        "nu_in_effective_s-1": "exact_nu_in_effective_s-1",
        "selected_modulation_K_m-1": "exact_selected_modulation_K_m-1",
        "wavebreaking_field_Vm": "exact_wavebreaking_field_Vm",
    }
    if best.get("exact_validation_passed", False):
        for target, source in exact_mapping.items():
            wave[target] = best[source]
    pulse = float(ode_cfg["pulse_duration_s"])
    follow = float(ode_cfg["collapse_follow_time_s"])
    width0 = (
        best["exact_initial_caviton_half_width_m"]
        if best.get("exact_validation_passed", False)
        else best["initial_caviton_half_width_m"]
    )
    lambda_De = best["lambda_De_m"]
    kinetic_width_floor = float(ode_cfg["kinetic_cutoff_Debye_lengths"]) * lambda_De
    envelope_width_floor = pi / (
        float(ode_cfg["envelope_validity_K_over_k_maximum"])
        * float(wave["carrier_k_m-1"])
    )
    width_floor = max(kinetic_width_floor, envelope_width_floor)
    shape0 = float(ode_cfg["initial_shape_factor"])
    shape_power = float(ode_cfg["shape_factor_power"])
    rate_scale = float(ode_cfg["calibrated_collapse_rate_scale"])
    interaction = best["interaction_length_m"]
    mode_power = best["coherent_mode_power_W"]
    gamma = wave["gamma_langmuir_s-1"]
    start_energy = (
        best["exact_required_onset_energy_J"]
        if best.get("exact_validation_passed", False)
        else best["required_onset_energy_J"]
    )
    start_time = min(
        best["exact_total_onset_time_s"]
        if best.get("exact_validation_passed", False)
        else best["total_onset_time_s"],
        pulse,
    )

    def right_hand_side(_: float, state: np.ndarray) -> np.ndarray:
        time = _
        width, energy = state
        width_safe = max(width, width_floor)
        shape = shape0 * (width_safe / width0) ** shape_power
        field = math.sqrt(
            4.0 * max(energy, 0.0)
            / (epsilon_0 * shape * width_safe**2 * interaction)
        )
        modulation = pi / width_safe
        growth = max(0.0, quartic_growth(field, modulation, wave))
        arrest = max(0.0, 1.0 - (width_floor / width_safe) ** 2)
        dwidth = -rate_scale * growth * width_safe * arrest
        source = mode_power if time <= pulse else 0.0
        denergy = source - 2.0 * gamma * energy
        return np.array([dwidth, denergy])

    def validity_event(_: float, state: np.ndarray) -> float:
        return state[0] - width_floor

    def wavebreaking_event(_: float, state: np.ndarray) -> float:
        width, energy = state
        width_safe = max(width, width_floor)
        shape = shape0 * (width_safe / width0) ** shape_power
        field = math.sqrt(
            4.0 * max(energy, 0.0)
            / (epsilon_0 * shape * width_safe**2 * interaction)
        )
        return field - wave["wavebreaking_field_Vm"]

    validity_event.terminal = True
    validity_event.direction = -1
    wavebreaking_event.terminal = True
    wavebreaking_event.direction = 1

    solution = solve_ivp(
        right_hand_side,
        (start_time, start_time + follow),
        (width0, start_energy),
        max_step=min(2.0e-9, follow / 500.0),
        rtol=2.0e-7,
        atol=(1.0e-12, 1.0e-16),
        dense_output=False,
        events=(validity_event, wavebreaking_event),
    )
    width = solution.y[0]
    energy = solution.y[1]
    shape = shape0 * (width / width0) ** shape_power
    field = np.sqrt(4.0 * np.maximum(energy, 0.0) / (
        epsilon_0 * shape * width**2 * interaction
    ))
    field_capped = np.minimum(field, wave["wavebreaking_field_Vm"])
    up = e * field_capped**2 / (
        4.0 * m_e * (2.0 * pi * wave["carrier_frequency_Hz"]) ** 2
    )
    return pd.DataFrame({
        "time_s": solution.t,
        "caviton_width_m": width,
        "wave_energy_J": energy,
        "uncapped_peak_field_Vm": field,
        "wavebreaking_capped_field_Vm": field_capped,
        "electron_ponderomotive_potential_eV": up,
        "envelope_cutoff_reached": (
            (envelope_width_floor >= kinetic_width_floor)
            & (width <= 1.001 * envelope_width_floor)
        ),
        "kinetic_cutoff_reached": (
            (kinetic_width_floor >= envelope_width_floor)
            & (width <= 1.001 * kinetic_width_floor)
        ),
        "envelope_K_over_k": pi / width / float(wave["carrier_k_m-1"]),
        "wavebreaking_reached": field >= wave["wavebreaking_field_Vm"],
    })


def audit_table(root: Path, cfg: dict) -> pd.DataFrame:
    level4 = pd.read_csv(root / cfg["input"]["level4_scenarios_csv"])
    level5 = pd.read_csv(root / "outputs/level5_scenarios.csv")
    nominal4 = level4[level4.scenario == "nominal_1p5"].iloc[0]
    nominal5 = level5[level5.scenario == "from121_collisional_decay"].iloc[0]
    return pd.DataFrame([
        ("L1 cross sections", "screening proxy tables", "quantitative design is not traceable", "replace by state-resolved measured/MCCC data"),
        ("L1 coherent power", "not separated from absorbed plasma power", "mode coupling efficiency is undefined", "measure/calibrate RF mode power and circuit Q"),
        ("L2 collision model", "collisionless root plus additive drag", "real-frequency shifts and nonlinear damping are omitted", "solve collisional kinetic dielectric"),
        ("L3 pump", "mean Fourier mode is clamped", "injected energy is not accounted", "couple envelope to a circuit/energy ODE"),
        ("L3-L4 transfer", "manual 1-D to radial Gaussian mapping", "energy and geometry are not conserved", "use one 2-D/3-D conservative state transfer"),
        ("L4 cutoff", f"cold wave breaking at {nominal4.cold_wavebreaking_estimate_Vm/1e3:.1f} kV/m", "fluid equations stop at the kinetic transition", "handoff full phase-space state to kinetic electrons"),
        ("L4 energy", f"field energy changes by {100*nominal4.relative_field_energy_change:.1f}%", "collapse localizes but does not energy-load the caviton", "explicit pump reservoir and nonlinear absorption"),
        ("L5 electrons", "quasineutral isothermal fluid", "double layers, trapping and Debye-scale fields cannot exist", "Poisson + kinetic electrons"),
        ("L5 slow-field cap", "50 kV/m", "would clip an optimized kilovolt-scale structure", "replace cap by Poisson-resolved stability criterion"),
        ("L5 density floor", "n_e/n_0 >= 0.05", "ambipolar potential remains O(Te)", "kinetic charge-separation model"),
        ("L5 statistics", f"60k particles; max {nominal5.maximum_deuteron_energy_eV:.1f} eV", "rare tails below about 1e-4 are unresolved", "importance sampling / more particles"),
    ], columns=["location", "current_implementation", "consequence", "required_change"])


def make_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    positions = np.arange(len(summary))
    dplus = summary.ode_transition_Dplus_energy_ceiling_eV.fillna(
        summary.exact_Dplus_energy_ceiling_eV.fillna(summary.Dplus_energy_ceiling_eV)
    )
    d3plus = summary.ode_transition_D3plus_energy_per_deuteron_ceiling_eV.fillna(
        summary.exact_D3plus_energy_per_deuteron_ceiling_eV.fillna(
            summary.D3plus_energy_per_deuteron_ceiling_eV
        )
    )
    ax.bar(positions - 0.2, dplus, width=0.4, label="Потолок D+")
    ax.bar(positions + 0.2, d3plus,
           width=0.4, label="Потолок D3+ на дейтрон")
    ax.axhspan(1.0e3, 1.0e5, color="#d62728", alpha=0.08, label="Цель 1–100 кэВ")
    ax.set_yscale("log")
    labels = {
        "current_validated_window": "Проверенное окно",
        "practical_extended_window": "Расширенное практическое",
        "physics_upper_window": "Физическая верхняя оценка",
    }
    ax.set_xticks(positions, [labels.get(item, item) for item in summary.search], rotation=15, ha="right")
    ax.set(ylabel="Потолок энергии, эВ на дейтрон",
           title="Ограниченные экстремумы энергонасыщения кавитона")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "19_extrema_energy_ceiling.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_extrema.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    level1_cfg = load_level1_config(root / "config.yaml")
    cross_sections = load_cross_sections(root, level1_cfg)
    rates = make_rate_cache(cross_sections)
    best_rows = []
    evaluation_frames = []
    histories = {}
    for search in cfg["searches"]:
        best, evaluations = optimize_search(
            search, cfg, level1_cfg, cross_sections, rates
        )
        best = exact_validate_candidate(root, best, cfg)
        evaluation_frames.append(evaluations)
        if best["feasible"]:
            history = integrate_best_ode(best, cfg)
            histories[search["name"]] = history
            transition = history.iloc[-1]
            pressure = float(best["electron_pressure_potential_bound_eV"])
            sheath = float(best["sheath_potential_bound_eV"])
            up = float(transition.electron_ponderomotive_potential_eV)
            total = up + pressure + sheath
            wavebreaking_ratio = (
                float(transition.uncapped_peak_field_Vm)
                / float(best.get("exact_wavebreaking_field_Vm", best["wavebreaking_field_Vm"]))
            )
            if bool(transition.wavebreaking_reached) or wavebreaking_ratio >= 0.999:
                reason = "wavebreaking"
            elif bool(transition.envelope_cutoff_reached):
                reason = "envelope_validity_cutoff"
            elif bool(transition.kinetic_cutoff_reached):
                reason = "kinetic_scale_cutoff"
            else:
                reason = "follow_time_or_pulse_end"
            best.update({
                "ode_transition_reason": reason,
                "ode_transition_time_s": float(transition.time_s),
                "ode_transition_width_m": float(transition.caviton_width_m),
                "ode_transition_peak_field_Vm": float(transition.wavebreaking_capped_field_Vm),
                "ode_transition_ponderomotive_eV": up,
                "ode_transition_Dplus_energy_ceiling_eV": total,
                "ode_transition_D2plus_energy_per_deuteron_ceiling_eV": total / 2.0,
                "ode_transition_D3plus_energy_per_deuteron_ceiling_eV": total / 3.0,
            })
        best_rows.append(best)
    summary = pd.DataFrame(best_rows)
    evaluations = pd.concat(evaluation_frames, ignore_index=True)
    audit = audit_table(root, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "extrema_summary.csv", index=False)
    evaluations.to_csv(args.output / "extrema_evaluations.csv", index=False)
    audit.to_csv(args.output / "code_audit_levels_1_5.csv", index=False)
    for name, history in histories.items():
        history.to_csv(args.output / f"extrema_history_{name}.csv", index=False)
    payload = {
        "interpretation": "constrained upper bounds; not a prediction of ion acceleration",
        "searches": summary.to_dict("records"),
        "audit": audit.to_dict("records"),
    }
    (args.output / "extrema_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(summary, root / "figures")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
