#!/usr/bin/env python3
"""Level-3 damped generalized Zakharov model for caviton onset.

The model combines a complex-root Langmuir dispersion coefficient from level 2
with a damped ion-acoustic density response. It calculates the finite-domain
modulational-instability threshold, the selected modulation scale, linear
formation rate/time, and a maintained-pump 1-D nonlinear onset check.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import atomic_mass as AMU
from scipy.constants import e, epsilon_0, m_e, pi

from plasma_level2 import (
    build_species,
    load_backgrounds,
    load_yaml,
    select_background,
    solve_kinetic_mode,
)


@dataclass(frozen=True)
class ZakharovCoefficients:
    electron_density_m3: float
    electron_temperature_eV: float
    lambda_De_m: float
    carrier_k_m1: float
    carrier_omega_rad_s: float
    carrier_group_velocity_ms: float
    langmuir_damping_s1: float
    acoustic_kinetic_damping_at_carrier_s1: float
    ion_neutral_damping_frequency_s1: float
    electron_neutral_frequency_s1: float
    effective_ion_mass_kg: float
    acoustic_speed_ms: float
    dispersion_P_m2s: float
    density_coupling_A_m3s: float
    ponderomotive_B: float


def load_level3_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def composition_by_name(level2_cfg: dict, name: str) -> dict:
    matches = [item for item in level2_cfg["composition_scenarios"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one composition named {name}")
    return matches[0]


def select_mode_rows(core: pd.DataFrame, keys: dict, composition: str) -> tuple[dict, dict]:
    mask = (
        np.isclose(core.pressure_Pa, keys["pressure_Pa"])
        & np.isclose(core.absorbed_power_W, keys["absorbed_power_W"])
        & np.isclose(core.ion_temperature_eV, keys["ion_temperature_eV"])
        & np.isclose(core.k_lambda_De, keys["k_lambda_De"])
        & (core.composition == composition)
    )
    rows = core[mask]
    if len(rows) != 2:
        raise ValueError(f"Expected two level-2 mode rows; found {len(rows)}")
    langmuir = rows[rows["mode"] == "langmuir"]
    acoustic = rows[rows["mode"] == "ion_acoustic_fast"]
    if len(langmuir) != 1 or len(acoustic) != 1:
        raise ValueError("Missing Langmuir or fast ion-acoustic row")
    return langmuir.iloc[0].to_dict(), acoustic.iloc[0].to_dict()


def langmuir_gvd(
    background: dict,
    Ti_eV: float,
    kappa: float,
    composition: dict,
    level2_cfg: dict,
) -> tuple[float, float]:
    electron, ions = build_species(background, Ti_eV, composition)
    k0 = kappa / electron.debye_length_m
    relative_step = 2.0e-3
    roots = [
        solve_kinetic_mode("langmuir", k0 * factor, electron, ions, level2_cfg["solver"])[0].real
        for factor in (1.0 - relative_step, 1.0, 1.0 + relative_step)
    ]
    second_derivative = (roots[2] - 2.0 * roots[1] + roots[0]) / (relative_step * k0) ** 2
    return 0.5 * second_derivative, roots[1]


def build_coefficients(
    background: dict,
    Ti_eV: float,
    kappa: float,
    composition: dict,
    langmuir_row: dict,
    acoustic_row: dict,
    level2_cfg: dict,
) -> ZakharovCoefficients:
    electron, ions = build_species(background, Ti_eV, composition)
    inverse_effective_mass = sum(
        item.density_m3 / electron.density_m3 / item.mass_kg for item in ions
    )
    effective_mass = 1.0 / inverse_effective_mass
    acoustic_speed = math.sqrt(e * (electron.temperature_eV + 3.0 * Ti_eV) / effective_mass)
    dispersion_P, omega = langmuir_gvd(background, Ti_eV, kappa, composition, level2_cfg)
    if dispersion_P <= 0.0:
        raise ValueError("The selected Langmuir branch is not focusing: P <= 0")
    return ZakharovCoefficients(
        electron_density_m3=electron.density_m3,
        electron_temperature_eV=electron.temperature_eV,
        lambda_De_m=electron.debye_length_m,
        carrier_k_m1=kappa / electron.debye_length_m,
        carrier_omega_rad_s=omega,
        carrier_group_velocity_ms=float(langmuir_row["group_velocity_ms"]),
        langmuir_damping_s1=abs(float(langmuir_row["gamma_total_s-1"])),
        acoustic_kinetic_damping_at_carrier_s1=abs(float(acoustic_row["gamma_kinetic_s-1"])),
        ion_neutral_damping_frequency_s1=float(acoustic_row["nu_in_effective_s-1"]),
        electron_neutral_frequency_s1=float(acoustic_row["nu_en_s-1"]),
        effective_ion_mass_kg=effective_mass,
        acoustic_speed_ms=acoustic_speed,
        dispersion_P_m2s=dispersion_P,
        density_coupling_A_m3s=omega / (2.0 * electron.density_m3),
        ponderomotive_B=epsilon_0 / (4.0 * effective_mass),
    )


def acoustic_damping(K_m1: float | np.ndarray, coeff: ZakharovCoefficients) -> float | np.ndarray:
    absolute_K = np.abs(K_m1)
    kinetic = coeff.acoustic_kinetic_damping_at_carrier_s1 * absolute_K / coeff.carrier_k_m1
    polarization = (absolute_K * coeff.lambda_De_m) ** 2
    polarization /= 1.0 + polarization
    collision = 0.5 * (
        coeff.ion_neutral_damping_frequency_s1
        + polarization * coeff.electron_neutral_frequency_s1
    )
    return kinetic + collision


def modulation_modes(coeff: ZakharovCoefficients, cfg: dict) -> np.ndarray:
    length = float(cfg["interaction_length_m"])
    K_min = 2.0 * pi / length
    K_max = float(cfg["maximum_modulation_to_carrier_k"]) * coeff.carrier_k_m1
    if cfg["discrete_modulation_modes"]:
        count = int(math.floor(K_max / K_min))
        if count < 1:
            raise ValueError("Interaction region is too short for an envelope modulation mode")
        return K_min * np.arange(1, count + 1, dtype=float)
    return np.linspace(K_min, K_max, 160)


def quartic_growth_rate(field_Vm: float, K_m1: float, coeff: ZakharovCoefficients) -> float:
    nu_s = float(acoustic_damping(K_m1, coeff))
    acoustic_polynomial = np.array([1.0, 2.0 * nu_s, (coeff.acoustic_speed_ms * K_m1) ** 2])
    envelope_polynomial = np.array([
        1.0,
        2.0 * coeff.langmuir_damping_s1,
        coeff.langmuir_damping_s1**2 + (coeff.dispersion_P_m2s * K_m1**2) ** 2,
    ])
    polynomial = np.polymul(acoustic_polynomial, envelope_polynomial)
    coupling = (
        2.0
        * coeff.density_coupling_A_m3s
        * coeff.ponderomotive_B
        * coeff.dispersion_P_m2s
        * field_Vm**2
        * K_m1**4
    )
    polynomial[-1] -= coupling
    return float(np.max(np.roots(polynomial).real))


def neutral_stability_field(K_m1: float, coeff: ZakharovCoefficients) -> float:
    numerator = coeff.acoustic_speed_ms**2 * (
        coeff.langmuir_damping_s1**2
        + coeff.dispersion_P_m2s**2 * K_m1**4
    )
    denominator = (
        2.0
        * coeff.density_coupling_A_m3s
        * coeff.ponderomotive_B
        * coeff.dispersion_P_m2s
        * K_m1**2
    )
    return math.sqrt(numerator / denominator)


def fastest_mode(field_Vm: float, modes: np.ndarray, coeff: ZakharovCoefficients) -> tuple[float, float]:
    growth = np.array([quartic_growth_rate(field_Vm, K, coeff) for K in modes])
    index = int(np.argmax(growth))
    return float(growth[index]), float(modes[index])


def threshold_result(coeff: ZakharovCoefficients, zakharov_cfg: dict) -> dict:
    modes = modulation_modes(coeff, zakharov_cfg)
    thresholds = np.array([neutral_stability_field(K, coeff) for K in modes])
    index = int(np.argmin(thresholds))
    threshold = float(thresholds[index])
    threshold_K = float(modes[index])
    continuum_optimal_K = math.sqrt(coeff.langmuir_damping_s1 / coeff.dispersion_P_m2s)
    continuum_threshold = neutral_stability_field(continuum_optimal_K, coeff)
    factor = float(zakharov_cfg["operating_field_factor"])
    operating_field = factor * threshold
    growth, operating_K = fastest_mode(operating_field, modes, coeff)
    seed = float(zakharov_cfg["density_seed_fraction"])
    target = float(zakharov_cfg["caviton_onset_depletion_fraction"])
    e_folds = math.log(target / seed)
    formation_time = e_folds / growth
    spacing = 2.0 * pi / operating_K
    half_width = pi / operating_K
    formation_speed = half_width / formation_time
    convective_length = coeff.carrier_group_velocity_ms * formation_time
    wave_energy_density = epsilon_0 * operating_field**2 / 4.0
    electron_thermal_energy_density = (
        coeff.electron_density_m3 * coeff.electron_temperature_eV * e
    )
    static_threshold_energy_ratio = (threshold_K * coeff.lambda_De_m) ** 2
    classical_static_field = math.sqrt(
        4.0 * electron_thermal_energy_density * static_threshold_energy_ratio / epsilon_0
    )
    ponderomotive_energy_eV = e * operating_field**2 / (4.0 * m_e * coeff.carrier_omega_rad_s**2)
    quasistatic_depletion = (
        coeff.ponderomotive_B * operating_field**2
        / (coeff.acoustic_speed_ms**2 * coeff.electron_density_m3)
    )
    return {
        "threshold_field_Vm": threshold,
        "threshold_modulation_K_m-1": threshold_K,
        "threshold_modulation_spacing_m": 2.0 * pi / threshold_K,
        "threshold_caviton_half_width_m": pi / threshold_K,
        "continuum_optimal_modulation_K_m-1": continuum_optimal_K,
        "continuum_optimal_modulation_spacing_m": 2.0 * pi / continuum_optimal_K,
        "continuum_optimal_caviton_half_width_m": pi / continuum_optimal_K,
        "continuum_minimum_threshold_field_Vm": continuum_threshold,
        "finite_domain_threshold_penalty": threshold / continuum_threshold,
        "classical_undamped_static_field_Vm": classical_static_field,
        "operating_field_factor": factor,
        "operating_field_Vm": operating_field,
        "maximum_growth_rate_s-1": growth,
        "operating_modulation_K_m-1": operating_K,
        "caviton_spacing_m": spacing,
        "caviton_half_width_m": half_width,
        "linear_formation_time_s": formation_time,
        "formation_speed_ms": formation_speed,
        "convective_formation_length_m": convective_length,
        "ideal_recirculation_passes": convective_length / float(zakharov_cfg["interaction_length_m"]),
        "wave_energy_density_Jm3": wave_energy_density,
        "wave_to_electron_thermal_energy_ratio": wave_energy_density / electron_thermal_energy_density,
        "ponderomotive_energy_eV": ponderomotive_energy_eV,
        "quasistatic_depletion_fraction_at_operating_field": quasistatic_depletion,
        "available_modulation_modes": int(len(modes)),
    }


def calculate_threshold_map(root_dir: Path, cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    level2_cfg = load_yaml(root_dir / cfg["input"]["level2_config"])
    backgrounds = load_backgrounds(root_dir, level2_cfg)
    core = pd.read_csv(root_dir / cfg["input"]["level2_core_csv"])
    final = pd.read_csv(root_dir / cfg["input"]["final_regimes_csv"])
    rows = []
    for parameters in final.to_dict("records"):
        keys = {name: parameters[name] for name in ["pressure_Pa", "absorbed_power_W", "ion_temperature_eV", "k_lambda_De"]}
        background = select_background(backgrounds, keys["pressure_Pa"], keys["absorbed_power_W"])
        for composition in level2_cfg["composition_scenarios"]:
            langmuir, acoustic = select_mode_rows(core, keys, composition["name"])
            coefficients = build_coefficients(
                background,
                keys["ion_temperature_eV"],
                keys["k_lambda_De"],
                composition,
                langmuir,
                acoustic,
                level2_cfg,
            )
            result = threshold_result(coefficients, cfg["zakharov"])
            rows.append({
                **keys,
                "composition": composition["name"],
                "electron_density_m3": coefficients.electron_density_m3,
                "electron_temperature_eV": coefficients.electron_temperature_eV,
                "carrier_k_m-1": coefficients.carrier_k_m1,
                "carrier_frequency_Hz": coefficients.carrier_omega_rad_s / (2.0 * pi),
                "carrier_group_velocity_ms": coefficients.carrier_group_velocity_ms,
                "langmuir_damping_s-1": coefficients.langmuir_damping_s1,
                "acoustic_speed_ms": coefficients.acoustic_speed_ms,
                "effective_ion_mass_u": coefficients.effective_ion_mass_kg / AMU,
                "dispersion_P_m2s": coefficients.dispersion_P_m2s,
                "density_coupling_A_m3s": coefficients.density_coupling_A_m3s,
                "ponderomotive_B": coefficients.ponderomotive_B,
                **result,
            })
    return pd.DataFrame(rows), level2_cfg, {"backgrounds": backgrounds, "core": core}


def simulate_caviton_onset(
    field_factor: float,
    threshold_row: dict,
    coeff: ZakharovCoefficients,
    cfg: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    length = float(cfg["interaction_length_m"])
    points = int(cfg["grid_points"])
    dt = float(cfg["time_step_s"])
    maximum_time = float(cfg["maximum_time_s"])
    record_every = int(cfg["record_every_steps"])
    x = np.arange(points) * length / points
    q = 2.0 * pi * np.fft.fftfreq(points, d=length / points)
    field0 = field_factor * float(threshold_row["threshold_field_Vm"])
    modes = modulation_modes(coeff, {
        "interaction_length_m": length,
        "maximum_modulation_to_carrier_k": 0.35,
        "discrete_modulation_modes": True,
    })
    _, seed_K = fastest_mode(field0, modes, coeff)
    envelope_seed = float(cfg["envelope_seed_fraction"])
    density_seed = float(cfg["density_seed_fraction"])
    electric = field0 * (1.0 + envelope_seed * np.cos(seed_K * x)).astype(complex)
    density = -density_seed * coeff.electron_density_m3 * np.cos(seed_K * x)
    density_velocity = np.zeros(points)
    damping_q = acoustic_damping(q, coeff)
    damping_q[0] = 0.0
    linear_half_step = np.exp(
        (-coeff.langmuir_damping_s1 - 1j * coeff.dispersion_P_m2s * q**2) * dt / 2.0
    )
    linear_half_step[0] = 1.0
    keep = np.ones(points, dtype=bool)
    if cfg["two_thirds_dealiasing"]:
        keep = np.abs(np.fft.fftfreq(points) * points) <= points / 3.0

    def filter_field(values: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft(values)
        spectrum[~keep] = 0.0
        return np.fft.ifft(spectrum)

    def density_rhs(n: np.ndarray, velocity: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_hat = np.fft.fft(n)
        velocity_hat = np.fft.fft(velocity)
        intensity_hat = np.fft.fft(intensity)
        n_hat[~keep] = 0.0
        velocity_hat[~keep] = 0.0
        intensity_hat[~keep] = 0.0
        acceleration_hat = (
            -(coeff.acoustic_speed_ms * q) ** 2 * n_hat
            - 2.0 * damping_q * velocity_hat
            - coeff.ponderomotive_B * q**2 * intensity_hat
        )
        return velocity, np.fft.ifft(acceleration_hat).real

    records = []
    target = float(cfg["caviton_onset_depletion_fraction"])
    onset_time = math.nan
    maximum_steps = int(math.ceil(maximum_time / dt))
    for step in range(maximum_steps + 1):
        time = step * dt
        depletion = -float(np.min(density)) / coeff.electron_density_m3
        if step % record_every == 0 or depletion >= target:
            records.append({
                "time_s": time,
                "minimum_density_perturbation_fraction": float(np.min(density)) / coeff.electron_density_m3,
                "density_rms_fraction": float(np.std(density)) / coeff.electron_density_m3,
                "maximum_field_over_pump": float(np.max(np.abs(electric))) / field0,
                "field_rms_over_pump": float(np.sqrt(np.mean(np.abs(electric) ** 2))) / field0,
            })
        if depletion >= target:
            onset_time = time
            break

        electric_hat = np.fft.fft(electric) * linear_half_step
        electric_hat[~keep] = 0.0
        if cfg["maintained_mean_pump"]:
            electric_hat[0] = field0 * points
        electric = np.fft.ifft(electric_hat)
        electric = filter_field(
            electric * np.exp(-1j * coeff.density_coupling_A_m3s * density * dt)
        )

        intensity = np.abs(electric) ** 2
        k1n, k1v = density_rhs(density, density_velocity, intensity)
        k2n, k2v = density_rhs(density + 0.5 * dt * k1n, density_velocity + 0.5 * dt * k1v, intensity)
        k3n, k3v = density_rhs(density + 0.5 * dt * k2n, density_velocity + 0.5 * dt * k2v, intensity)
        k4n, k4v = density_rhs(density + dt * k3n, density_velocity + dt * k3v, intensity)
        density += dt * (k1n + 2.0 * k2n + 2.0 * k3n + k4n) / 6.0
        density_velocity += dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        density = filter_field(density).real
        density_velocity = filter_field(density_velocity).real

        electric_hat = np.fft.fft(electric) * linear_half_step
        electric_hat[~keep] = 0.0
        if cfg["maintained_mean_pump"]:
            electric_hat[0] = field0 * points
        electric = np.fft.ifft(electric_hat)

    profile = pd.DataFrame({
        "x_m": x,
        "density_perturbation_fraction": density / coeff.electron_density_m3,
        "field_envelope_Vm": np.abs(electric),
        "field_over_pump": np.abs(electric) / field0,
    })
    summary = {
        "field_factor": field_factor,
        "field_Vm": field0,
        "seed_modulation_K_m-1": seed_K,
        "seed_caviton_half_width_m": pi / seed_K,
        "onset_time_s": onset_time,
        "onset_reached": bool(np.isfinite(onset_time)),
        "final_minimum_density_perturbation_fraction": float(np.min(density)) / coeff.electron_density_m3,
        "final_maximum_field_over_pump": float(np.max(np.abs(electric))) / field0,
        "convective_length_at_onset_m": (
            coeff.carrier_group_velocity_ms * onset_time if np.isfinite(onset_time) else math.nan
        ),
        "apparent_formation_speed_ms": (
            (pi / seed_K) / onset_time if np.isfinite(onset_time) else math.nan
        ),
        "ideal_recirculation_passes": (
            coeff.carrier_group_velocity_ms * onset_time / length if np.isfinite(onset_time) else math.nan
        ),
    }
    return summary, pd.DataFrame(records), profile


def nonlinear_validation(
    root_dir: Path,
    cfg: dict,
    threshold_map: pd.DataFrame,
    level2_cfg: dict,
    context: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    simulation_cfg = cfg["nonlinear_validation"]
    if not simulation_cfg["enabled"]:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    keys = {name: simulation_cfg[name] for name in ["pressure_Pa", "absorbed_power_W", "ion_temperature_eV", "k_lambda_De"]}
    composition = composition_by_name(level2_cfg, simulation_cfg["composition"])
    background = select_background(context["backgrounds"], keys["pressure_Pa"], keys["absorbed_power_W"])
    langmuir, acoustic = select_mode_rows(context["core"], keys, composition["name"])
    coeff = build_coefficients(
        background,
        keys["ion_temperature_eV"],
        keys["k_lambda_De"],
        composition,
        langmuir,
        acoustic,
        level2_cfg,
    )
    threshold_match = threshold_map[
        np.isclose(threshold_map.pressure_Pa, keys["pressure_Pa"])
        & np.isclose(threshold_map.absorbed_power_W, keys["absorbed_power_W"])
        & np.isclose(threshold_map.ion_temperature_eV, keys["ion_temperature_eV"])
        & np.isclose(threshold_map.k_lambda_De, keys["k_lambda_De"])
        & (threshold_map.composition == composition["name"])
    ]
    if len(threshold_match) != 1:
        raise ValueError("Nonlinear validation threshold row is ambiguous")
    threshold_row = threshold_match.iloc[0].to_dict()
    run_summaries = []
    reference_timeseries = pd.DataFrame()
    reference_profile = pd.DataFrame()
    for factor in simulation_cfg["field_factors"]:
        resolved_simulation_cfg = {
            **cfg["zakharov"],
            **simulation_cfg,
        }
        summary, timeseries, profile = simulate_caviton_onset(
            float(factor), threshold_row, coeff, resolved_simulation_cfg
        )
        run_summaries.append({**keys, "composition": composition["name"], **summary})
        if math.isclose(float(factor), float(simulation_cfg["reference_field_factor"])):
            reference_timeseries = timeseries
            reference_profile = profile
    return pd.DataFrame(run_summaries), reference_timeseries, reference_profile


def convergence_study(
    root_dir: Path,
    cfg: dict,
    threshold_map: pd.DataFrame,
    level2_cfg: dict,
    context: dict,
) -> pd.DataFrame:
    convergence_cfg = cfg.get("convergence_study", {})
    if not convergence_cfg.get("enabled", False):
        return pd.DataFrame()
    rows = []
    for case in convergence_cfg["cases"]:
        case_cfg = {
            **cfg,
            "nonlinear_validation": {
                **cfg["nonlinear_validation"],
                "field_factors": [float(convergence_cfg["field_factor"])],
                "reference_field_factor": float(convergence_cfg["field_factor"]),
                "grid_points": int(case["grid_points"]),
                "time_step_s": float(case["time_step_s"]),
            },
        }
        run, _, _ = nonlinear_validation(root_dir, case_cfg, threshold_map, level2_cfg, context)
        rows.append({
            "grid_points": int(case["grid_points"]),
            "time_step_s": float(case["time_step_s"]),
            "onset_time_s": float(run.iloc[0]["onset_time_s"]),
            "final_maximum_field_over_pump": float(run.iloc[0]["final_maximum_field_over_pump"]),
        })
    result = pd.DataFrame(rows).sort_values(["grid_points", "time_step_s"]).reset_index(drop=True)
    reference_time = float(result.iloc[-1]["onset_time_s"])
    result["relative_onset_time_error_vs_finest"] = (
        (result["onset_time_s"] - reference_time).abs() / reference_time
    )
    return result


def summarize(
    thresholds: pd.DataFrame,
    nonlinear_runs: pd.DataFrame,
    convergence: pd.DataFrame,
) -> dict:
    def interval(frame: pd.DataFrame, column: str) -> list[float]:
        return [float(frame[column].min()), float(frame[column].max())]

    by_composition = {}
    for name, group in thresholds.groupby("composition"):
        by_composition[name] = {
            "threshold_field_Vm": interval(group, "threshold_field_Vm"),
            "operating_field_Vm": interval(group, "operating_field_Vm"),
            "caviton_half_width_m": interval(group, "caviton_half_width_m"),
            "continuum_optimal_caviton_half_width_m": interval(
                group, "continuum_optimal_caviton_half_width_m"
            ),
            "finite_domain_threshold_penalty": interval(group, "finite_domain_threshold_penalty"),
            "maximum_growth_rate_s-1": interval(group, "maximum_growth_rate_s-1"),
            "linear_formation_time_s": interval(group, "linear_formation_time_s"),
            "formation_speed_ms": interval(group, "formation_speed_ms"),
            "convective_formation_length_m": interval(group, "convective_formation_length_m"),
            "wave_to_electron_thermal_energy_ratio": interval(group, "wave_to_electron_thermal_energy_ratio"),
        }
    return {
        "model_scope": "1-D damped generalized Zakharov modulational onset; maintained pump for nonlinear validation",
        "counts": {
            "threshold_rows": int(len(thresholds)),
            "nonlinear_validation_runs": int(len(nonlinear_runs)),
            "convergence_runs": int(len(convergence)),
        },
        "ranges_by_composition": by_composition,
        "nonlinear_validation": nonlinear_runs.to_dict("records"),
        "convergence_study": convergence.to_dict("records"),
    }


def make_figures(
    thresholds: pd.DataFrame,
    nonlinear_runs: pd.DataFrame,
    timeseries: pd.DataFrame,
    profile: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"pure_Dplus": "#1f77b4", "molecular_nominal": "#d62728", "D3plus_dominant": "#2ca02c"}

    center = thresholds[
        np.isclose(thresholds.pressure_Pa, 0.05)
        & np.isclose(thresholds.absorbed_power_W, 15.0)
        & np.isclose(thresholds.ion_temperature_eV, 0.1)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for composition, group in center.groupby("composition"):
        axes[0].plot(group.k_lambda_De, group.threshold_field_Vm / 1e3, marker="o", label=composition, color=colors[composition])
        axes[1].plot(group.k_lambda_De, group.linear_formation_time_s * 1e6, marker="o", label=composition, color=colors[composition])
    axes[0].set(xlabel=r"$k\lambda_{De}$", ylabel="Пороговое поле, кВ/м", title="Порог модуляционной неустойчивости")
    axes[1].set(xlabel=r"$k\lambda_{De}$", ylabel="Линейное время, мкс", title="Формирование при 1.5 порога")
    axes[1].legend(title="Ионный состав")
    fig.suptitle("Уровень 3: порог и время формирования")
    fig.tight_layout()
    fig.savefig(output_dir / "10_level3_threshold_time.png", dpi=180)
    plt.close(fig)

    if not nonlinear_runs.empty:
        fig, ax = plt.subplots(figsize=(8.3, 5.2))
        ax.plot(nonlinear_runs.field_factor, nonlinear_runs.onset_time_s * 1e6, marker="o")
        ax.set(xlabel=r"$E_0/E_{th}$", ylabel="Время до 5% разрежения, мкс",
               title="Нелинейная проверка: центральная молекулярная точка")
        fig.tight_layout()
        fig.savefig(output_dir / "11_level3_nonlinear_onset.png", dpi=180)
        plt.close(fig)

    if not timeseries.empty and not profile.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        axes[0].plot(timeseries.time_s * 1e6, -timeseries.minimum_density_perturbation_fraction * 100.0,
                     label="Глубина каверны")
        axes[0].plot(timeseries.time_s * 1e6, timeseries.maximum_field_over_pump,
                     label=r"$|E|_{max}/E_0$")
        axes[0].set(xlabel="Время, мкс", ylabel="Относительная величина", title="Развитие неустойчивости")
        axes[0].legend()
        axes[1].plot(profile.x_m * 1e3, profile.density_perturbation_fraction * 100.0, label=r"$\delta n/n_0$, %")
        axes[1].plot(profile.x_m * 1e3, profile.field_over_pump, label=r"$|E|/E_0$")
        axes[1].set(xlabel="Координата, мм", ylabel="Относительная величина", title="Профиль при достижении 5%")
        axes[1].legend()
        fig.suptitle("Формирование кавитона при 1.5 порога")
        fig.tight_layout()
        fig.savefig(output_dir / "12_level3_caviton_onset.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_level3.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent
    cfg = load_level3_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    thresholds, level2_cfg, context = calculate_threshold_map(root_dir, cfg)
    nonlinear_runs, timeseries, profile = nonlinear_validation(root_dir, cfg, thresholds, level2_cfg, context)
    convergence = convergence_study(root_dir, cfg, thresholds, level2_cfg, context)
    summary = summarize(thresholds, nonlinear_runs, convergence)
    thresholds.to_csv(args.output / "level3_threshold_map.csv", index=False)
    nonlinear_runs.to_csv(args.output / "level3_nonlinear_runs.csv", index=False)
    timeseries.to_csv(args.output / "level3_reference_timeseries.csv", index=False)
    profile.to_csv(args.output / "level3_reference_profile.csv", index=False)
    convergence.to_csv(args.output / "level3_convergence.csv", index=False)
    (args.output / "level3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(thresholds, nonlinear_runs, timeseries, profile, root_dir / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
