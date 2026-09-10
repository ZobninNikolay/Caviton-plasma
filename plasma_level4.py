#!/usr/bin/env python3
"""Level-4 axisymmetric extended Zakharov-fluid model of caviton collapse.

The high-frequency Langmuir envelope is coupled to nonlinear radial ion
continuity and momentum equations. Neutral drag/charge exchange, chemistry,
an axial magnetic field, background outflow, boundary losses and an optional
ion-acoustic drift drive are explicit, separately switchable closures.
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
from scipy.constants import e, epsilon_0, m_e, pi
from scipy.linalg import solve_banded

from plasma_level2 import load_backgrounds, load_yaml, select_background
from plasma_level3 import (
    ZakharovCoefficients,
    build_coefficients,
    composition_by_name,
    select_mode_rows,
)


@dataclass(frozen=True)
class CollapseScenario:
    name: str
    field_factor: float
    maintained_pump: bool
    langmuir_damping_scale: float
    ion_neutral_rate_scale: float
    chemistry_enabled: bool
    boundary_sponge_enabled: bool
    magnetic_field_T: float
    outflow_edge_ms: float
    drift_growth_s1: float


@dataclass(frozen=True)
class CollapseContext:
    coeff: ZakharovCoefficients
    background: dict
    threshold_row: dict
    nonlinear_rows: pd.DataFrame
    threshold_field_Vm: float
    wavebreaking_field_Vm: float


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def match_rows(frame: pd.DataFrame, point: dict) -> pd.Series:
    return (
        np.isclose(frame.pressure_Pa, point["pressure_Pa"])
        & np.isclose(frame.absorbed_power_W, point["absorbed_power_W"])
        & np.isclose(frame.ion_temperature_eV, point["ion_temperature_eV"])
        & np.isclose(frame.k_lambda_De, point["k_lambda_De"])
        & (frame.composition == point["composition"])
    )


def build_context(root: Path, cfg: dict) -> CollapseContext:
    point = cfg["operating_point"]
    level2_cfg = load_yaml(root / cfg["input"]["level2_config"])
    backgrounds = load_backgrounds(root, level2_cfg)
    core = pd.read_csv(root / cfg["input"]["level2_core_csv"])
    thresholds = pd.read_csv(root / cfg["input"]["level3_threshold_csv"])
    nonlinear = pd.read_csv(root / cfg["input"]["level3_nonlinear_csv"])
    background = select_background(backgrounds, point["pressure_Pa"], point["absorbed_power_W"])
    composition = composition_by_name(level2_cfg, point["composition"])
    langmuir, acoustic = select_mode_rows(core, point, point["composition"])
    coeff = build_coefficients(
        background,
        point["ion_temperature_eV"],
        point["k_lambda_De"],
        composition,
        langmuir,
        acoustic,
        level2_cfg,
    )
    threshold_match = thresholds[match_rows(thresholds, point)]
    nonlinear_match = nonlinear[match_rows(nonlinear, point)]
    if len(threshold_match) != 1 or nonlinear_match.empty:
        raise ValueError("The selected level-3 operating point is missing or ambiguous")
    wavebreaking = m_e * coeff.carrier_omega_rad_s * (
        coeff.carrier_omega_rad_s / coeff.carrier_k_m1
    ) / e
    return CollapseContext(
        coeff=coeff,
        background=background,
        threshold_row=threshold_match.iloc[0].to_dict(),
        nonlinear_rows=nonlinear_match.copy(),
        threshold_field_Vm=float(threshold_match.iloc[0]["threshold_field_Vm"]),
        wavebreaking_field_Vm=wavebreaking,
    )


def resolve_scenario(item: dict, cfg: dict) -> CollapseScenario:
    physics = cfg["physics"]
    pump = cfg["pump"]
    return CollapseScenario(
        name=item["name"],
        field_factor=float(item.get("field_factor", pump["field_factor"])),
        maintained_pump=bool(item.get("maintained_pump", pump["maintained"])),
        langmuir_damping_scale=float(item.get(
            "langmuir_damping_scale", physics["langmuir_sideband_damping_scale"]
        )),
        ion_neutral_rate_scale=float(item.get(
            "ion_neutral_rate_scale", physics["ion_neutral_rate_scale"]
        )),
        chemistry_enabled=bool(item.get("chemistry_enabled", True)),
        boundary_sponge_enabled=bool(item.get("boundary_sponge_enabled", True)),
        magnetic_field_T=float(item.get("magnetic_field_T", physics["axial_magnetic_field_T"])),
        outflow_edge_ms=float(item.get("outflow_edge_ms", physics["background_outflow_edge_ms"])),
        drift_growth_s1=float(item.get(
            "drift_growth_s-1", physics["ion_acoustic_drift_growth_s-1"]
        )),
    )


def radial_grid(radius: float, cells: int) -> tuple[np.ndarray, float]:
    dr = radius / cells
    return (np.arange(cells, dtype=float) + 0.5) * dr, dr


def radial_laplacian_coefficients(r: np.ndarray, dr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = len(r)
    lower = np.zeros(cells)
    diagonal = np.zeros(cells)
    upper = np.zeros(cells)
    inner_face = np.arange(cells, dtype=float) * dr
    outer_face = (np.arange(cells, dtype=float) + 1.0) * dr
    lower[1:] = inner_face[1:] / (r[1:] * dr**2)
    upper[:-1] = outer_face[:-1] / (r[:-1] * dr**2)
    diagonal = -(lower + upper)
    return lower, diagonal, upper


def apply_tridiagonal(
    values: np.ndarray,
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    result = diagonal * values
    result[1:] += lower[1:] * values[:-1]
    result[:-1] += upper[:-1] * values[1:]
    return result


def crank_nicolson_dispersion(
    electric: np.ndarray,
    duration: float,
    dispersion_P: float,
    laplacian: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    lower, diagonal, upper = laplacian
    coefficient = 0.5j * dispersion_P * duration
    rhs = electric + coefficient * apply_tridiagonal(electric, lower, diagonal, upper)
    banded = np.zeros((3, len(electric)), dtype=complex)
    banded[0, 1:] = -coefficient * upper[:-1]
    banded[1, :] = 1.0 - coefficient * diagonal
    banded[2, :-1] = -coefficient * lower[1:]
    return solve_banded((1, 1), banded, rhs, check_finite=False)


def gradient(values: np.ndarray, dr: float) -> np.ndarray:
    result = np.empty_like(values)
    result[0] = 0.0
    result[-1] = (values[-1] - values[-2]) / dr
    result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dr)
    return result


def upwind_derivative(values: np.ndarray, velocity: np.ndarray, dr: float) -> np.ndarray:
    backward = np.empty_like(values)
    forward = np.empty_like(values)
    backward[0] = 0.0
    backward[1:] = (values[1:] - values[:-1]) / dr
    forward[:-1] = (values[1:] - values[:-1]) / dr
    forward[-1] = -values[-1] / dr
    return np.where(velocity >= 0.0, backward, forward)


def divergence_upwind(
    quantity: np.ndarray,
    velocity: np.ndarray,
    r: np.ndarray,
    dr: float,
    outer_ambient: float,
) -> np.ndarray:
    cells = len(r)
    face_velocity = np.zeros(cells + 1)
    face_velocity[1:cells] = 0.5 * (velocity[:-1] + velocity[1:])
    face_velocity[cells] = velocity[-1]
    left = np.empty(cells + 1)
    right = np.empty(cells + 1)
    left[0] = quantity[0]
    right[0] = quantity[0]
    left[1:cells] = quantity[:-1]
    right[1:cells] = quantity[1:]
    left[cells] = quantity[-1]
    right[cells] = outer_ambient
    flux = face_velocity * np.where(face_velocity >= 0.0, left, right)
    face_radius = np.arange(cells + 1, dtype=float) * dr
    return (face_radius[1:] * flux[1:] - face_radius[:-1] * flux[:-1]) / (r * dr)


def sponge_profile(r: np.ndarray, radius: float, start_fraction: float, maximum_rate: float) -> np.ndarray:
    start = start_fraction * radius
    coordinate = np.clip((r - start) / max(radius - start, np.finfo(float).eps), 0.0, 1.0)
    return maximum_rate * coordinate**4


def local_envelope_step(
    electric: np.ndarray,
    rho: np.ndarray,
    duration: float,
    pump_field: float,
    coeff: ZakharovCoefficients,
    langmuir_damping: float,
    sponge: np.ndarray,
    maintained_pump: bool,
) -> np.ndarray:
    complex_rate = langmuir_damping + sponge + 1j * (
        coeff.density_coupling_A_m3s * coeff.electron_density_m3 * (rho - 1.0)
    )
    decay = np.exp(-complex_rate * duration)
    source = langmuir_damping * pump_field if maintained_pump else 0.0
    response = np.where(
        np.abs(complex_rate) > 1.0e-30,
        source * (1.0 - decay) / complex_rate,
        source * duration,
    )
    return electric * decay + response


def ion_rhs(
    rho: np.ndarray,
    radial_velocity: np.ndarray,
    azimuthal_velocity: np.ndarray,
    intensity: np.ndarray,
    r: np.ndarray,
    dr: float,
    laplacian: tuple[np.ndarray, np.ndarray, np.ndarray],
    coeff: ZakharovCoefficients,
    background: dict,
    scenario: CollapseScenario,
    cfg: dict,
    sponge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physics = cfg["physics"]
    radius = float(cfg["geometry"]["radial_domain_m"])
    background_flow = scenario.outflow_edge_ms * r / radius
    rho_safe = np.maximum(rho, 1.0e-4)
    density_rhs = -divergence_upwind(rho, radial_velocity, r, dr, 1.0)
    density_rhs -= divergence_upwind(rho - 1.0, background_flow, r, dr, 0.0)

    ionization = scenario.chemistry_enabled * float(physics["ionization_rate_scale"]) * (
        float(background["neutral_density_m3"]) * float(background["K_ion_m3s"])
    )
    recombination = scenario.chemistry_enabled * float(
        physics["recombination_coefficient_m3s"]
    ) * coeff.electron_density_m3
    wall_loss = scenario.chemistry_enabled * float(physics["wall_loss_rate_scale"]) * float(
        background["nu_loss_s-1"]
    )
    equilibrium_source = ionization - recombination - wall_loss
    density_rhs += (
        ionization * rho - recombination * rho**2 - wall_loss * rho - equilibrium_source
    )
    density_rhs += float(physics["density_diffusivity_m2s"]) * apply_tridiagonal(
        rho - 1.0, *laplacian
    )
    density_rhs -= sponge * (rho - 1.0)

    total_ion_neutral = scenario.ion_neutral_rate_scale * coeff.ion_neutral_damping_frequency_s1
    mass_loading = ionization
    effective_drag = max(0.0, total_ion_neutral + mass_loading - scenario.drift_growth_s1)
    omega_ci = e * scenario.magnetic_field_T / coeff.effective_ion_mass_kg
    advecting_velocity = radial_velocity + background_flow
    pressure_force = -coeff.acoustic_speed_ms**2 * gradient(np.log(rho_safe), dr)
    ponderomotive_force = -epsilon_0 * gradient(intensity, dr) / (
        4.0 * coeff.effective_ion_mass_kg * coeff.electron_density_m3 * rho_safe
    )
    viscosity = float(physics["ion_viscosity_m2s"])
    radial_rhs = (
        -advecting_velocity * upwind_derivative(radial_velocity, advecting_velocity, dr)
        + pressure_force
        + ponderomotive_force
        + omega_ci * azimuthal_velocity
        - effective_drag * radial_velocity
        + viscosity * apply_tridiagonal(radial_velocity, *laplacian)
        - sponge * radial_velocity
    )
    azimuthal_rhs = (
        -advecting_velocity * upwind_derivative(azimuthal_velocity, advecting_velocity, dr)
        - omega_ci * radial_velocity
        - effective_drag * azimuthal_velocity
        + viscosity * apply_tridiagonal(azimuthal_velocity, *laplacian)
        - sponge * azimuthal_velocity
    )
    radial_rhs[0] = 0.0
    azimuthal_rhs[0] = 0.0
    return density_rhs, radial_rhs, azimuthal_rhs


def radial_integral(values: np.ndarray, r: np.ndarray, dr: float) -> float:
    return float(2.0 * pi * np.sum(values * r) * dr)


def state_metrics(
    time: float,
    electric: np.ndarray,
    rho: np.ndarray,
    radial_velocity: np.ndarray,
    azimuthal_velocity: np.ndarray,
    r: np.ndarray,
    dr: float,
    pump_field: float,
    coeff: ZakharovCoefficients,
) -> dict:
    intensity = np.abs(electric) ** 2
    excess = np.maximum(intensity - pump_field**2, 0.0)
    excess_integral = radial_integral(excess, r, dr)
    if excess_integral > 0.0:
        field_scale = math.sqrt(radial_integral(r**2 * excess, r, dr) / excess_integral)
        localized_fraction = radial_integral(excess[r <= field_scale], r[r <= field_scale], dr) / excess_integral
    else:
        field_scale = math.nan
        localized_fraction = 0.0
    deficit = np.maximum(1.0 - rho, 0.0)
    deficit_integral = radial_integral(deficit, r, dr)
    density_scale = (
        math.sqrt(radial_integral(r**2 * deficit, r, dr) / deficit_integral)
        if deficit_integral > 0.0 else math.nan
    )
    field_energy = epsilon_0 * radial_integral(intensity, r, dr) / 4.0
    excess_field_energy = epsilon_0 * excess_integral / 4.0
    ion_kinetic = radial_integral(
        0.5 * coeff.effective_ion_mass_kg * coeff.electron_density_m3 * rho
        * (radial_velocity**2 + azimuthal_velocity**2),
        r,
        dr,
    )
    return {
        "time_s": time,
        "field_scale_m": field_scale,
        "density_cavity_scale_m": density_scale,
        "maximum_field_Vm": float(np.max(np.abs(electric))),
        "maximum_field_over_pump": float(np.max(np.abs(electric))) / pump_field,
        "minimum_density_fraction": float(np.min(rho)),
        "maximum_density_depletion_fraction": 1.0 - float(np.min(rho)),
        "localized_excess_energy_fraction": localized_fraction,
        "field_energy_per_axial_length_Jm": field_energy,
        "excess_field_energy_per_axial_length_Jm": excess_field_energy,
        "ion_kinetic_energy_per_axial_length_Jm": ion_kinetic,
        "maximum_radial_ion_speed_ms": float(np.max(np.abs(radial_velocity))),
    }


def initial_peak_ratio(context: CollapseContext, field_factor: float) -> float:
    rows = context.nonlinear_rows
    index = int(np.argmin(np.abs(rows.field_factor.to_numpy() - field_factor)))
    return float(rows.iloc[index]["final_maximum_field_over_pump"])


def simulate(
    context: CollapseContext,
    cfg: dict,
    scenario: CollapseScenario,
    radial_cells: int | None = None,
    time_step_s: float | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    coeff = context.coeff
    radius = float(cfg["geometry"]["radial_domain_m"])
    scale = float(cfg["geometry"]["initial_caviton_scale_m"])
    depletion = float(cfg["geometry"]["initial_density_depletion_fraction"])
    numerics = cfg["numerics"]
    cells = int(radial_cells or numerics["radial_cells"])
    dt = float(time_step_s or numerics["time_step_s"])
    maximum_time = float(numerics["maximum_time_s"])
    record_every = max(1, int(round(float(numerics["record_interval_s"]) / dt)))
    r, dr = radial_grid(radius, cells)
    laplacian = radial_laplacian_coefficients(r, dr)
    sponge = sponge_profile(
        r,
        radius,
        float(cfg["boundary"]["sponge_start_fraction"]),
        float(cfg["boundary"]["maximum_sponge_rate_s-1"]),
    ) if scenario.boundary_sponge_enabled else np.zeros(cells)

    pump_field = scenario.field_factor * context.threshold_field_Vm
    peak_ratio = initial_peak_ratio(context, scenario.field_factor)
    shape = np.exp(-(r / scale) ** 2)
    electric = pump_field * (1.0 + (peak_ratio - 1.0) * shape).astype(complex)
    rho = 1.0 - depletion * shape
    radial_velocity = np.zeros(cells)
    azimuthal_velocity = np.zeros(cells)
    langmuir_damping = scenario.langmuir_damping_scale * coeff.langmuir_damping_s1
    kinetic_cutoff = float(numerics["kinetic_cutoff_Debye_lengths"]) * coeff.lambda_De_m
    density_floor = float(numerics["density_floor_fraction"])
    initial = state_metrics(
        0.0, electric, rho, radial_velocity, azimuthal_velocity, r, dr, pump_field, coeff
    )
    records = [initial]
    best_field_scale = float(initial["field_scale_m"])
    best_snapshot = (
        electric.copy(), rho.copy(), radial_velocity.copy(), azimuthal_velocity.copy()
    )
    stop_reason = "maximum_time"
    physical_cutoff_time = math.nan
    maximum_steps = int(math.ceil(maximum_time / dt))

    for step in range(1, maximum_steps + 1):
        electric = crank_nicolson_dispersion(electric, 0.5 * dt, coeff.dispersion_P_m2s, laplacian)
        electric = local_envelope_step(
            electric, rho, 0.5 * dt, pump_field, coeff, langmuir_damping, sponge,
            scenario.maintained_pump,
        )
        intensity = np.abs(electric) ** 2
        k1 = ion_rhs(
            rho, radial_velocity, azimuthal_velocity, intensity, r, dr, laplacian,
            coeff, context.background, scenario, cfg, sponge,
        )
        midpoint_rho = rho + 0.5 * dt * k1[0]
        midpoint_radial = radial_velocity + 0.5 * dt * k1[1]
        midpoint_azimuthal = azimuthal_velocity + 0.5 * dt * k1[2]
        k2 = ion_rhs(
            midpoint_rho, midpoint_radial, midpoint_azimuthal, intensity, r, dr,
            laplacian, coeff, context.background, scenario, cfg, sponge,
        )
        rho += dt * k2[0]
        radial_velocity += dt * k2[1]
        azimuthal_velocity += dt * k2[2]
        radial_velocity[0] = 0.0
        azimuthal_velocity[0] = 0.0

        electric = local_envelope_step(
            electric, rho, 0.5 * dt, pump_field, coeff, langmuir_damping, sponge,
            scenario.maintained_pump,
        )
        electric = crank_nicolson_dispersion(electric, 0.5 * dt, coeff.dispersion_P_m2s, laplacian)

        time = step * dt
        immediate_stop = None
        if float(np.min(rho)) <= density_floor:
            immediate_stop = "density_floor"
        if float(np.max(np.abs(electric))) >= context.wavebreaking_field_Vm:
            immediate_stop = "wavebreaking_field"
        if step % record_every == 0 or step == maximum_steps or immediate_stop is not None:
            metrics = state_metrics(
                time, electric, rho, radial_velocity, azimuthal_velocity, r, dr,
                pump_field, coeff,
            )
            records.append(metrics)
            if metrics["field_scale_m"] < best_field_scale:
                best_field_scale = float(metrics["field_scale_m"])
                best_snapshot = (
                    electric.copy(), rho.copy(), radial_velocity.copy(), azimuthal_velocity.copy()
                )
            if not np.isfinite(list(metrics.values())).all():
                stop_reason = "non_finite_state"
                break
            if immediate_stop is not None:
                stop_reason = immediate_stop
                physical_cutoff_time = time
                break
            if metrics["field_scale_m"] <= max(kinetic_cutoff, 4.0 * dr):
                stop_reason = "kinetic_scale_cutoff"
                physical_cutoff_time = time
                break
            if metrics["maximum_field_Vm"] >= context.wavebreaking_field_Vm:
                stop_reason = "wavebreaking_field"
                physical_cutoff_time = time
                break

    history = pd.DataFrame(records)
    minimum_index = int(history.field_scale_m.idxmin())
    minimum_state = history.loc[minimum_index]
    maximum_field = float(history.maximum_field_Vm.max())
    compression = float(initial["field_scale_m"] / minimum_state.field_scale_m)
    peak_gain = maximum_field / float(initial["maximum_field_Vm"])
    collapse_detected = (
        compression >= float(numerics["minimum_collapse_compression"])
        and peak_gain >= float(numerics["minimum_peak_field_gain"])
    )
    collapse_time = float(minimum_state.time_s) if collapse_detected else math.nan
    if collapse_detected and stop_reason == "maximum_time":
        stop_reason = "arrested_after_localization"
    best_electric, best_rho, best_radial, best_azimuthal = best_snapshot
    profile = pd.DataFrame({
        "radius_m": r,
        "density_fraction_at_minimum_scale": best_rho,
        "field_at_minimum_scale_Vm": np.abs(best_electric),
        "radial_velocity_at_minimum_scale_ms": best_radial,
        "azimuthal_velocity_at_minimum_scale_ms": best_azimuthal,
        "density_fraction": rho,
        "field_envelope_Vm": np.abs(electric),
        "field_over_pump": np.abs(electric) / pump_field,
        "radial_ion_velocity_ms": radial_velocity,
        "azimuthal_ion_velocity_ms": azimuthal_velocity,
    })
    cx_fraction = float(cfg["physics"]["charge_exchange_fraction_of_nu_in"])
    total_in = scenario.ion_neutral_rate_scale * coeff.ion_neutral_damping_frequency_s1
    nominal_ionization = float(context.background["neutral_density_m3"]) * float(
        context.background["K_ion_m3s"]
    )
    nominal_recombination = float(cfg["physics"]["recombination_coefficient_m3s"]) * (
        coeff.electron_density_m3
    )
    nominal_wall_loss = float(context.background["nu_loss_s-1"])
    field_energy = history.field_energy_per_axial_length_Jm
    summary = {
        "scenario": scenario.name,
        "field_factor": scenario.field_factor,
        "threshold_field_Vm": context.threshold_field_Vm,
        "pump_field_Vm": pump_field,
        "configured_seed_scale_m": scale,
        "initial_scale_m": float(initial["field_scale_m"]),
        "minimum_scale_m": float(minimum_state.field_scale_m),
        "scale_compression_ratio": compression,
        "time_of_minimum_scale_s": float(minimum_state.time_s),
        "collapse_time_s": collapse_time,
        "physical_cutoff_time_s": physical_cutoff_time,
        "collapse_detected": bool(collapse_detected),
        "stop_reason": stop_reason,
        "initial_maximum_field_Vm": float(initial["maximum_field_Vm"]),
        "maximum_field_Vm": maximum_field,
        "field_at_minimum_scale_Vm": float(minimum_state.maximum_field_Vm),
        "maximum_field_gain": peak_gain,
        "cold_wavebreaking_estimate_Vm": context.wavebreaking_field_Vm,
        "maximum_density_depletion_fraction": float(history.maximum_density_depletion_fraction.max()),
        "final_density_depletion_fraction": float(history.iloc[-1].maximum_density_depletion_fraction),
        "maximum_localized_excess_energy_fraction": float(history.localized_excess_energy_fraction.max()),
        "maximum_radial_ion_speed_ms": float(history.maximum_radial_ion_speed_ms.max()),
        "ion_neutral_momentum_rate_s-1": (1.0 - cx_fraction) * total_in,
        "charge_exchange_rate_s-1": cx_fraction * total_in,
        "chemistry_enabled": scenario.chemistry_enabled,
        "ionization_rate_s-1": nominal_ionization if scenario.chemistry_enabled else 0.0,
        "recombination_rate_at_n0_s-1": nominal_recombination if scenario.chemistry_enabled else 0.0,
        "wall_loss_rate_s-1": nominal_wall_loss if scenario.chemistry_enabled else 0.0,
        "magnetic_field_T": scenario.magnetic_field_T,
        "ion_cyclotron_frequency_rad_s": e * scenario.magnetic_field_T / coeff.effective_ion_mass_kg,
        "outflow_edge_ms": scenario.outflow_edge_ms,
        "drift_growth_s-1": scenario.drift_growth_s1,
        "maintained_pump": scenario.maintained_pump,
        "radial_cells": cells,
        "time_step_s": dt,
        "kinetic_cutoff_m": kinetic_cutoff,
        "relative_field_energy_change": (
            float(field_energy.iloc[-1] / field_energy.iloc[0] - 1.0)
        ),
        "relative_field_energy_range": (
            float((field_energy.max() - field_energy.min()) / field_energy.iloc[0])
        ),
    }
    return summary, history, profile


def run_scenarios(
    context: CollapseContext,
    cfg: dict,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summaries = []
    histories = {}
    profiles = {}
    for item in cfg["scenarios"]:
        scenario = resolve_scenario(item, cfg)
        summary, history, profile = simulate(context, cfg, scenario)
        summaries.append(summary)
        histories[scenario.name] = history
        profiles[scenario.name] = profile
    return pd.DataFrame(summaries), histories, profiles


def convergence_study(context: CollapseContext, cfg: dict) -> pd.DataFrame:
    conv_cfg = cfg["convergence_study"]
    matches = [item for item in cfg["scenarios"] if item["name"] == conv_cfg["scenario"]]
    if len(matches) != 1:
        raise ValueError("Convergence scenario is missing or ambiguous")
    scenario = resolve_scenario(matches[0], cfg)
    rows = []
    for case in conv_cfg["cases"]:
        summary, _, _ = simulate(
            context,
            cfg,
            scenario,
            radial_cells=int(case["radial_cells"]),
            time_step_s=float(case["time_step_s"]),
        )
        rows.append(summary)
    result = pd.DataFrame(rows).sort_values("radial_cells").reset_index(drop=True)
    finest = result.iloc[-1]
    for column in ["minimum_scale_m", "maximum_field_Vm", "time_of_minimum_scale_s"]:
        result[f"relative_error_{column}_vs_finest"] = (
            (result[column] - float(finest[column])).abs() / max(abs(float(finest[column])), 1.0e-30)
        )
    return result


def timescale_table(context: CollapseContext, cfg: dict, summaries: pd.DataFrame) -> pd.DataFrame:
    nominal = summaries[summaries.scenario == "nominal_1p5"].iloc[0]
    full = summaries[summaries.scenario == "full_B10mT_flow200"].iloc[0]
    collapse_time = float(nominal.collapse_time_s)
    entries = [
        ("collapse_localization", 1.0 / collapse_time),
        ("langmuir_sideband_damping", context.coeff.langmuir_damping_s1),
        ("ion_neutral_total", float(nominal["ion_neutral_momentum_rate_s-1"] + nominal["charge_exchange_rate_s-1"])),
        ("ionization", float(nominal["ionization_rate_s-1"])),
        ("recombination_at_n0", float(nominal["recombination_rate_at_n0_s-1"])),
        ("wall_loss", float(nominal["wall_loss_rate_s-1"])),
        ("ion_gyro_B10mT", float(full["ion_cyclotron_frequency_rad_s"])),
        ("flow_transit_200ms", float(full["outflow_edge_ms"]) / float(cfg["geometry"]["radial_domain_m"])),
    ]
    return pd.DataFrame([
        {
            "process": name,
            "rate_s-1": rate,
            "characteristic_time_s": 1.0 / rate,
            "characteristic_time_over_collapse_time": (1.0 / rate) / collapse_time,
        }
        for name, rate in entries if rate > 0.0
    ])


def make_figures(
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    profiles: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    selected = [name for name in ["nominal_1p25", "nominal_1p5", "nominal_2p0"] if name in histories]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for name in selected:
        history = histories[name]
        axes[0].plot(history.time_s * 1e6, history.field_scale_m * 1e3, label=name)
        axes[1].plot(history.time_s * 1e6, history.maximum_field_Vm / 1e3, label=name)
    axes[0].set(xlabel="Время, мкс", ylabel="L(t), мм", title="Радиальный масштаб ВЧ-поля")
    axes[1].set(xlabel="Время, мкс", ylabel=r"$E_{max}$, кВ/м", title="Максимум огибающей")
    axes[1].legend()
    fig.suptitle("Уровень 4: нелинейная локализация")
    fig.tight_layout()
    fig.savefig(output_dir / "13_level4_localization.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for name in ["nominal_1p5", "full_B10mT_flow200", "drift_assisted_1p5", "free_decay_1p5"]:
        if name not in histories:
            continue
        history = histories[name]
        axes[0].plot(history.time_s * 1e6, history.maximum_density_depletion_fraction * 100.0, label=name)
        axes[1].plot(history.time_s * 1e6, history.maximum_radial_ion_speed_ms, label=name)
    axes[0].set(xlabel="Время, мкс", ylabel="Разрежение, %", title="Ответ ионной плотности")
    axes[1].set(xlabel="Время, мкс", ylabel="Макс. радиальная скорость, м/с", title="Ионное движение")
    axes[1].legend()
    fig.suptitle("Столкновения, магнитное поле, поток и потери")
    fig.tight_layout()
    fig.savefig(output_dir / "14_level4_density_response.png", dpi=180)
    plt.close(fig)

    if "nominal_1p5" in profiles:
        profile = profiles["nominal_1p5"]
        fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
        ax2 = ax1.twinx()
        ax1.plot(profile.radius_m * 1e3, profile.field_envelope_Vm / 1e3, color="#d62728", label="Поле")
        ax2.plot(profile.radius_m * 1e3, profile.density_fraction, color="#1f77b4", label="Плотность")
        ax1.set(xlabel="Радиус, мм", ylabel="Поле, кВ/м", title="Финальный профиль nominal_1p5")
        ax2.set_ylabel(r"$n_i/n_0$")
        fig.tight_layout()
        fig.savefig(output_dir / "15_level4_final_profile.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_level4.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(args.config)
    context = build_context(root, cfg)
    summaries, histories, profiles = run_scenarios(context, cfg)
    convergence = convergence_study(context, cfg)
    timescales = timescale_table(context, cfg, summaries)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output / "level4_scenarios.csv", index=False)
    observable_columns = [
        "scenario", "field_factor", "threshold_field_Vm", "pump_field_Vm",
        "configured_seed_scale_m", "initial_scale_m", "collapse_time_s",
        "minimum_scale_m", "field_at_minimum_scale_Vm", "maximum_field_Vm",
        "maximum_density_depletion_fraction", "maximum_localized_excess_energy_fraction",
        "physical_cutoff_time_s", "stop_reason",
    ]
    summaries[observable_columns].to_csv(args.output / "level4_observables.csv", index=False)
    convergence.to_csv(args.output / "level4_convergence.csv", index=False)
    timescales.to_csv(args.output / "level4_timescales.csv", index=False)
    for name, history in histories.items():
        history.to_csv(args.output / f"level4_history_{name}.csv", index=False)
    for name, profile in profiles.items():
        profile.to_csv(args.output / f"level4_profile_{name}.csv", index=False)
    summary = {
        "model_scope": "axisymmetric 2-D radial extended Zakharov envelope plus nonlinear ion fluid",
        "operating_point": cfg["operating_point"],
        "threshold_field_Vm": context.threshold_field_Vm,
        "wavebreaking_field_Vm": context.wavebreaking_field_Vm,
        "scenario_count": int(len(summaries)),
        "scenarios": summaries.to_dict("records"),
        "convergence": convergence.to_dict("records"),
        "timescales": timescales.to_dict("records"),
    }
    (args.output / "level4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figures(summaries, histories, profiles, root / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
