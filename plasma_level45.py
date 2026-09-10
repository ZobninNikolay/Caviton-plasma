#!/usr/bin/env python3
"""Level 4.5: kinetic-electron 1-D electrostatic PIC/Vlasov-Poisson audit.

The code is deliberately a mechanism discriminator, not a reactor-yield model.
It transfers a phase-resolved Langmuir packet from the corrected level-4 ODE,
advances kinetic electrons and molecular deuterium ions with Poisson's equation,
and reports whether double layers, wave breaking, shocks or ion trapping are
compatible with the validated low-temperature-plasma window.
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
from scipy.constants import atomic_mass, e, epsilon_0, m_e, pi
from scipy.special import ndtri


@dataclass
class Population:
    x: np.ndarray
    v: np.ndarray
    weight: np.ndarray
    charge: np.ndarray
    mass_ratio: np.ndarray
    deuterons: np.ndarray
    name_code: np.ndarray


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def periodic_distance(x: np.ndarray, center: float, length: float) -> np.ndarray:
    return (x - center + 0.5 * length) % length - 0.5 * length


def cic_deposit(x: np.ndarray, weights: np.ndarray, cells: int, length: float) -> np.ndarray:
    coordinate = x * cells / length
    left = np.floor(coordinate).astype(np.int64) % cells
    fraction = coordinate - np.floor(coordinate)
    result = np.bincount(left, weights=weights * (1.0 - fraction), minlength=cells)
    result += np.bincount((left + 1) % cells, weights=weights * fraction, minlength=cells)
    return result / (length / cells)


def cic_interpolate(x: np.ndarray, field: np.ndarray, length: float) -> np.ndarray:
    cells = len(field)
    coordinate = x * cells / length
    left_floor = np.floor(coordinate)
    left = left_floor.astype(np.int64) % cells
    fraction = coordinate - left_floor
    return field[left] * (1.0 - fraction) + field[(left + 1) % cells] * fraction


def poisson_field(ion_density: np.ndarray, electron_density: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    cells = len(ion_density)
    charge_density = ion_density - electron_density
    charge_density -= charge_density.mean()
    wavenumber = 2.0 * pi * np.fft.fftfreq(cells, d=length / cells)
    charge_hat = np.fft.fft(charge_density)
    field_hat = np.zeros(cells, dtype=complex)
    potential_hat = np.zeros(cells, dtype=complex)
    nonzero = wavenumber != 0.0
    field_hat[nonzero] = charge_hat[nonzero] / (1j * wavenumber[nonzero])
    potential_hat[nonzero] = charge_hat[nonzero] / wavenumber[nonzero] ** 2
    return np.fft.ifft(field_hat).real, np.fft.ifft(potential_hat).real


def quiet_maxwell(count: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    probabilities = (np.arange(count, dtype=float) + 0.5) / count
    values = sigma * ndtri(probabilities)
    return values[rng.permutation(count)]


def density_profiles(
    grid_x: np.ndarray,
    length: float,
    width: float,
    depletion: float,
    kappa: float,
    phase: float,
    requested_field: float,
    electron_floor: float,
    double_layer_potential: float = 0.0,
    double_layer_width: float = 20.0,
) -> dict:
    distance = periodic_distance(grid_x, 0.5 * length, length)
    envelope = np.exp(-(distance / width) ** 2)
    ion_density = 1.0 - depletion * (envelope - envelope.mean())
    carrier = envelope * np.cos(kappa * distance + phase)
    carrier -= carrier.mean()
    target_unscaled = requested_field * carrier
    if double_layer_potential > 0.0:
        z = distance / double_layer_width
        slow_field_unit = 0.5 / double_layer_width / np.cosh(np.clip(z, -30.0, 30.0)) ** 2
        slow_field_unit -= slow_field_unit.mean()
        target_unscaled = target_unscaled + double_layer_potential * slow_field_unit
    k_grid = 2.0 * pi * np.fft.fftfreq(len(grid_x), d=length / len(grid_x))
    derivative_target = np.fft.ifft(1j * k_grid * np.fft.fft(target_unscaled)).real
    positive = derivative_target > 0.0
    if np.any(positive):
        maximum_scale = float(np.min((ion_density[positive] - electron_floor) / derivative_target[positive]))
    else:
        maximum_scale = math.inf
    scale = min(1.0, max(0.0, 0.999 * maximum_scale))
    actual_field = requested_field * scale
    target_field = scale * target_unscaled
    electron_density = ion_density - scale * derivative_target
    return {
        "ion_density": ion_density,
        "electron_density": electron_density,
        "target_field": target_field,
        "requested_field": requested_field,
        "actual_field": actual_field,
        "charge_feasible_fraction": scale,
        "minimum_initial_electron_density": float(electron_density.min()),
    }


def make_population(
    length: float,
    cells: int,
    ppc: int,
    profiles: dict,
    packet_width: float,
    kappa: float,
    phase: float,
    omega_norm: float,
    Te_eV: float,
    Ti_eV: float,
    ion_species: list[dict],
    flow_mach: float,
    sound_speed_norm: float,
    rng: np.random.Generator,
) -> tuple[Population, Population]:
    count_e = cells * ppc
    x_e = (np.arange(count_e, dtype=float) + 0.5) * length / count_e
    w_e = np.interp(x_e, np.linspace(0.0, length, cells, endpoint=False), profiles["electron_density"], period=length)
    w_e *= length / count_e
    distance_e = periodic_distance(x_e, 0.5 * length, length)
    envelope_e = np.exp(-(distance_e / packet_width) ** 2)
    coherent_velocity = profiles["actual_field"] / omega_norm * envelope_e * np.sin(kappa * distance_e + phase)
    # A paired quiet start in every cell suppresses the finite-particle current
    # noise that otherwise produces secular grid heating over ion time scales.
    cell_velocities = ndtri((np.arange(ppc, dtype=float) + 0.5) / ppc)
    v_e = np.tile(cell_velocities, cells) + coherent_velocity
    electrons = Population(
        x_e, v_e, w_e, -np.ones(count_e), np.ones(count_e),
        np.ones(count_e), np.zeros(count_e, dtype=np.int16),
    )

    total_ions = cells * ppc
    ion_parts = []
    code = 0
    for item in ion_species:
        fraction = float(item["fraction"])
        count = max(8, int(round(total_ions * fraction)))
        x = (np.arange(count, dtype=float) + 0.5 + 0.173 * code) * length / count
        x %= length
        density = np.interp(x, np.linspace(0.0, length, cells, endpoint=False), profiles["ion_density"], period=length)
        weight = density * length * fraction / count
        mass_ratio = float(item["mass_u"]) * atomic_mass / m_e
        thermal_sigma = math.sqrt((Ti_eV / Te_eV) / mass_ratio)
        v = quiet_maxwell(count, thermal_sigma, rng)
        if flow_mach > 0.0:
            distance = periodic_distance(x, 0.5 * length, length)
            v += -flow_mach * sound_speed_norm * np.tanh(distance / max(10.0, 0.05 * length))
        ion_parts.append((x, v, weight, np.full(count, mass_ratio), np.full(count, int(item["deuterons_per_ion"])), np.full(count, code)))
        code += 1
    ions = Population(
        np.concatenate([part[0] for part in ion_parts]),
        np.concatenate([part[1] for part in ion_parts]),
        np.concatenate([part[2] for part in ion_parts]),
        np.ones(sum(len(part[0]) for part in ion_parts)),
        np.concatenate([part[3] for part in ion_parts]),
        np.concatenate([part[4] for part in ion_parts]),
        np.concatenate([part[5] for part in ion_parts]).astype(np.int16),
    )
    return electrons, ions


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(probability, cumulative, values[order]))


def binned_ion_moments(ions: Population, cells: int, length: float) -> tuple[np.ndarray, np.ndarray]:
    density = cic_deposit(ions.x, ions.weight, cells, length)
    momentum = cic_deposit(ions.x, ions.weight * ions.v, cells, length)
    return density, momentum / np.maximum(density, 1.0e-10)


def run_pic_case(case: dict, regime: pd.Series, cfg: dict) -> tuple[dict, pd.DataFrame]:
    pic = cfg["pic"]
    transfer = cfg["transfer"]
    cells = int(case.get("grid_cells", pic["grid_cells"]))
    ppc = int(case.get("particles_per_cell", pic["particles_per_cell_per_charge_population"]))
    dt = float(case.get("normalized_time_step", pic["normalized_time_step"]))
    end_time = float(case.get("normalized_end_time", pic["normalized_end_time"]))
    record_every = int(pic["record_every_steps"])
    phase = float(case.get("phase", 0.0))
    Te = float(regime.electron_temperature_eV)
    lambda_De = float(regime.lambda_De_m)
    omega_pe = math.sqrt(float(regime.electron_density_m3) * e**2 / (epsilon_0 * m_e))
    kappa = float(regime.k_lambda_De)
    omega_norm = 2.0 * pi * float(regime.exact_carrier_frequency_Hz) / omega_pe
    interaction_norm = float(regime.interaction_length_m) / lambda_De
    length = 2.0 * interaction_norm
    width = float(regime.ode_transition_width_m) / lambda_De
    field_unit = Te / lambda_De
    requested = (
        float(case.get("field_fraction", transfer["requested_field_fraction_of_transition"]))
        * float(regime.ode_transition_peak_field_Vm) / field_unit
    )
    grid_x = (np.arange(cells, dtype=float) + 0.5) * length / cells
    profiles = density_profiles(
        grid_x, length, width,
        float(case.get("depletion", transfer["density_depletion_fraction"])),
        kappa, phase, requested,
        float(transfer["minimum_electron_density_fraction"]),
        float(case.get("double_layer_potential_Te", 0.0)),
        float(cfg["controls"]["imposed_double_layer_width_Debye"]),
    )
    sound_norm = float(regime.exact_sound_speed_ms) / (lambda_De * omega_pe)
    rng = np.random.default_rng(int(pic["random_seed"]) + int(case.get("seed_offset", 0)))
    electrons, ions = make_population(
        length, cells, ppc, profiles, width, kappa, phase, omega_norm, Te,
        float(pic["ion_temperature_eV"]), pic["ion_species"],
        float(case.get("flow_mach", 0.0)), sound_norm, rng,
    )
    nu_e = float(regime["nu_en_s-1"]) / omega_pe if pic["collisions_enabled"] else 0.0
    nu_i = float(regime["exact_nu_in_effective_s-1"]) / omega_pe if pic["collisions_enabled"] else 0.0

    def field_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ne = cic_deposit(electrons.x, electrons.weight, cells, length)
        ni = cic_deposit(ions.x, ions.weight, cells, length)
        field, potential = poisson_field(ni, ne, length)
        return ne, ni, field, potential

    ne, ni, field, potential = field_state()
    electrons.v += -0.5 * dt * cic_interpolate(electrons.x, field, length)
    ions.v += 0.5 * dt * cic_interpolate(ions.x, field, length) / ions.mass_ratio
    initial_total_energy = None
    records = []
    steps = int(math.ceil(end_time / dt))
    wavebreaking_norm = float(regime.exact_wavebreaking_field_Vm) / field_unit
    vphase_norm = float(regime.exact_phase_velocity_ms) / (lambda_De * omega_pe)

    for step in range(steps + 1):
        if step % record_every == 0 or step == steps:
            efield_e = cic_interpolate(electrons.x, field, length)
            efield_i = cic_interpolate(ions.x, field, length)
            ve_full = electrons.v + 0.5 * dt * efield_e
            vi_full = ions.v - 0.5 * dt * efield_i / ions.mass_ratio
            ion_energy = 0.5 * Te * ions.mass_ratio * vi_full**2 / ions.deuterons
            electron_energy = 0.5 * Te * ve_full**2
            field_energy = 0.5 * float(np.mean(field**2))
            kinetic_e = float(np.sum(electrons.weight * 0.5 * ve_full**2) / length)
            kinetic_i = float(np.sum(ions.weight * 0.5 * ions.mass_ratio * vi_full**2) / length)
            total_energy = field_energy + kinetic_e + kinetic_i
            if initial_total_energy is None:
                initial_total_energy = total_energy
            density_i, bulk_i = binned_ion_moments(ions, cells, length)
            trapping_width = math.sqrt(max(0.0, 2.0 * float(np.ptp(potential))))
            trapped = np.abs(ve_full - vphase_norm) <= trapping_width
            records.append({
                "time_omega_pe": step * dt,
                "time_s": step * dt / omega_pe,
                "maximum_field_normalized": float(np.max(np.abs(field))),
                "maximum_field_Vm": float(np.max(np.abs(field))) * field_unit,
                "potential_span_Te": float(np.ptp(potential)),
                "potential_span_eV": float(np.ptp(potential)) * Te,
                "minimum_electron_density_fraction": float(ne.min()),
                "minimum_ion_density_fraction": float(ni.min()),
                "maximum_ion_compression": float(ni.max()),
                "maximum_ion_bulk_Mach": float(np.max(np.abs(bulk_i))) / max(sound_norm, 1.0e-30),
                "electron_trapped_weight_fraction": float(np.sum(electrons.weight[trapped]) / np.sum(electrons.weight)),
                "electron_p999_energy_eV": weighted_quantile(electron_energy, electrons.weight, 0.999),
                "deuteron_p999_energy_eV": weighted_quantile(ion_energy, ions.weight, 0.999),
                "maximum_deuteron_energy_eV": float(np.max(ion_energy)),
                "fraction_above_100eV": float(np.sum(ions.weight[ion_energy >= 100.0]) / np.sum(ions.weight)),
                "fraction_above_1keV": float(np.sum(ions.weight[ion_energy >= 1000.0]) / np.sum(ions.weight)),
                "field_energy_normalized": field_energy,
                "electron_kinetic_energy_normalized": kinetic_e,
                "ion_kinetic_energy_normalized": kinetic_i,
                "relative_total_energy_error": (total_energy - initial_total_energy) / initial_total_energy,
                "wavebreaking_field_ratio": float(np.max(np.abs(field))) / wavebreaking_norm,
            })
        if step == steps:
            break
        electrons.x = (electrons.x + dt * electrons.v) % length
        ions.x = (ions.x + dt * ions.v) % length
        ne, ni, field, potential = field_state()
        electrons.v += -dt * cic_interpolate(electrons.x, field, length)
        ions.v += dt * cic_interpolate(ions.x, field, length) / ions.mass_ratio
        if nu_e > 0.0:
            flips = rng.random(len(electrons.v)) < nu_e * dt
            electrons.v[flips] *= -1.0
        if nu_i > 0.0:
            exchange = rng.random(len(ions.v)) < nu_i * dt
            if np.any(exchange):
                sigma = np.sqrt((float(pic["ion_temperature_eV"]) / Te) / ions.mass_ratio[exchange])
                ions.v[exchange] = rng.normal(size=int(np.sum(exchange))) * sigma

    history = pd.DataFrame(records)
    final = history.iloc[-1]
    summary = {
        "case": case["name"],
        "phase_rad": phase,
        "depletion_fraction": float(case.get("depletion", transfer["density_depletion_fraction"])),
        "flow_mach_seed": float(case.get("flow_mach", 0.0)),
        "double_layer_seed_Te": float(case.get("double_layer_potential_Te", 0.0)),
        "grid_cells": cells,
        "particles_electrons": len(electrons.x),
        "particles_ions": len(ions.x),
        "physical_mass_ratio_Dplus": 2.014 * atomic_mass / m_e,
        "normalized_end_time": end_time,
        "normalized_time_step": dt,
        "physical_end_time_s": end_time / omega_pe,
        "ion_plasma_periods_Dplus": end_time / (2.0 * pi * math.sqrt(2.014 * atomic_mass / m_e)),
        "requested_initial_field_Vm": requested * field_unit,
        "charge_feasible_initial_field_Vm": profiles["actual_field"] * field_unit,
        "charge_feasible_fraction": profiles["charge_feasible_fraction"],
        "minimum_constructed_electron_density": profiles["minimum_initial_electron_density"],
        "initial_K_over_k": pi / width / kappa,
        "maximum_wavebreaking_field_ratio": float(history.wavebreaking_field_ratio.max()),
        "maximum_potential_span_eV": float(history.potential_span_eV.max()),
        "maximum_potential_span_Te": float(history.potential_span_Te.max()),
        "maximum_electron_trapped_fraction": float(history.electron_trapped_weight_fraction.max()),
        "maximum_ion_compression": float(history.maximum_ion_compression.max()),
        "maximum_ion_bulk_Mach": float(history.maximum_ion_bulk_Mach.max()),
        "maximum_electron_p999_energy_eV": float(history.electron_p999_energy_eV.max()),
        "maximum_deuteron_p999_energy_eV": float(history.deuteron_p999_energy_eV.max()),
        "maximum_deuteron_energy_eV": float(history.maximum_deuteron_energy_eV.max()),
        "maximum_fraction_above_100eV": float(history.fraction_above_100eV.max()),
        "maximum_fraction_above_1keV": float(history.fraction_above_1keV.max()),
        "maximum_absolute_energy_error": float(history.relative_total_energy_error.abs().max()),
        "double_layer_candidate": bool(
            history.potential_span_Te.max() >= 5.0
            and history.iloc[-1].potential_span_Te >= 0.5 * history.potential_span_Te.max()
        ),
        "wavebreaking_observed": bool(history.wavebreaking_field_ratio.max() >= 1.0),
        "kinetic_shock_candidate": bool(
            history.maximum_ion_compression.max() >= 1.5
            and history.maximum_ion_bulk_Mach.max() >= 1.6
        ),
        "resolved_zero_count_95pct_upper_fraction": 3.0 / len(ions.x),
    }
    return summary, history


def geometry_table(regime: pd.Series, cfg: dict) -> pd.DataFrame:
    radius = float(regime.radius_m)
    length = float(regime.chamber_length_m)
    volume = pi * radius**2 * length
    rows = []
    for item in cfg["geometries"]:
        name = item["name"]
        if name == "cylinder":
            dimensions = f"R={radius:.4f} m, L={length:.4f} m"
            area = 2.0 * pi * radius * length + 2.0 * pi * radius**2
            path = 2.0 * length
        elif name == "equal_volume_sphere":
            sphere_radius = (3.0 * volume / (4.0 * pi)) ** (1.0 / 3.0)
            dimensions = f"R={sphere_radius:.4f} m"
            area = 4.0 * pi * sphere_radius**2
            path = 4.0 * sphere_radius
        elif name == "equal_volume_torus_AR2":
            aspect = float(item["aspect_ratio"])
            minor = (volume / (2.0 * pi**2 * aspect)) ** (1.0 / 3.0)
            major = aspect * minor
            dimensions = f"Rmajor={major:.4f} m, a={minor:.4f} m"
            area = 4.0 * pi**2 * major * minor
            path = 2.0 * pi * major
        else:
            throat = float(item["throat_area_ratio"])
            dimensions = f"R={radius:.4f} m, L={length:.4f} m, Athroat/A={throat:.2f}"
            area = 1.15 * (2.0 * pi * radius * length + 2.0 * pi * radius**2)
            path = 2.0 * length
        area_to_volume = area / volume
        overlap = float(item["mode_overlap_multiplier"])
        recirculation = float(item["recirculation_multiplier"])
        rows.append({
            "geometry": name,
            "equal_volume_m3": volume,
            "dimensions": dimensions,
            "surface_to_volume_m-1": area_to_volume,
            "relative_wall_loss": area_to_volume / (2.0 / radius + 2.0 / length),
            "assumed_mode_overlap_multiplier": overlap,
            "assumed_recirculation_multiplier": recirculation,
            "round_trip_path_m": path,
            "screening_coherent_coupling_multiplier": overlap**2 * recirculation,
        })
    return pd.DataFrame(rows)


def quality_and_onset_scan(regime: pd.Series, cfg: dict, geometries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    omega = 2.0 * pi * float(regime.exact_carrier_frequency_Hz)
    base_gamma = float(regime["exact_gamma_langmuir_s-1"])
    base_required = float(regime.exact_required_onset_energy_J)
    base_growth = float(regime["exact_growth_rate_s-1"])
    seed = 1.0e-3
    absorbed_power = float(regime.absorbed_power_W)
    for _, geometry in geometries.iterrows():
        coupling_multiplier = float(geometry.screening_coherent_coupling_multiplier)
        wall_multiplier = float(geometry.relative_wall_loss)
        for q_multiplier in cfg["quality_scan"]["Q_multipliers"]:
            gamma = base_gamma * (0.85 + 0.15 * wall_multiplier) / float(q_multiplier)
            q_loaded = omega / (2.0 * gamma)
            for coherent_fraction in cfg["quality_scan"]["coherent_mode_fractions"]:
                mode_power = min(0.95, float(coherent_fraction) * coupling_multiplier) * absorbed_power
                steady_energy = mode_power / (2.0 * gamma)
                if base_required >= steady_energy:
                    charge_time = math.inf
                else:
                    charge_time = -math.log1p(-base_required / steady_energy) / (2.0 * gamma)
                for target in cfg["quality_scan"]["onset_depletions"]:
                    formation = math.log(float(target) / seed) / base_growth
                    total = charge_time + formation
                    rows.append({
                        "geometry": geometry.geometry,
                        "Q_multiplier": float(q_multiplier),
                        "loaded_Q": q_loaded,
                        "coherent_fraction_setting": float(coherent_fraction),
                        "effective_mode_power_W": mode_power,
                        "onset_depletion_fraction": float(target),
                        "charge_time_s": charge_time,
                        "formation_time_s": formation,
                        "total_time_s": total,
                        "charge_fraction_of_total": charge_time / total if math.isfinite(total) else math.nan,
                    })
    return pd.DataFrame(rows)


def mechanism_bounds(regime: pd.Series) -> pd.DataFrame:
    Te = float(regime.electron_temperature_eV)
    cs = float(regime.exact_sound_speed_ms)
    vphase = float(regime.exact_phase_velocity_ms)
    mD = 2.014 * atomic_mass
    v1 = math.sqrt(2.0 * 1.0e3 * e / mD)
    v100 = math.sqrt(2.0 * 1.0e5 * e / mD)
    return pd.DataFrame([
        ("Langmuir resonance", vphase, 0.5 * mD * vphase**2 / e, "phase velocity is far above 1-100 keV deuterons"),
        ("Ion-acoustic resonance", cs, 0.5 * mD * cs**2 / e, "phase velocity is far below 1 keV deuterons"),
        ("Mach-3 shock reflection", 3.0 * cs, 2.0 * mD * (3.0 * cs) ** 2 / e, "optimistic reflected-ion energy"),
        ("Five-Te double layer", math.nan, 5.0 * Te, "thermal-electron potential scale"),
        ("1 keV deuteron", v1, 1.0e3, "target lower edge"),
        ("100 keV deuteron", v100, 1.0e5, "target upper edge"),
    ], columns=["mechanism", "velocity_ms", "energy_eV", "interpretation"])


def synergy_table(regime: pd.Series, quality: pd.DataFrame) -> pd.DataFrame:
    target = quality[quality.onset_depletion_fraction == 0.05]
    def pick(geometry: str, q: float, coherent: float) -> pd.Series:
        return target[
            (target.geometry == geometry)
            & np.isclose(target.Q_multiplier, q)
            & np.isclose(target.coherent_fraction_setting, coherent)
        ].iloc[0]
    selections = [
        ("baseline", pick("cylinder", 1.0, 0.03)),
        ("Q_x4_only", pick("cylinder", 4.0, 0.03)),
        ("coupling_30pct_only", pick("cylinder", 1.0, 0.30)),
        ("sphere_Q4_coupling30", pick("equal_volume_sphere", 4.0, 0.30)),
        ("torus_Q4_coupling30", pick("equal_volume_torus_AR2", 4.0, 0.30)),
        ("nozzle_Q4_coupling30", pick("converging_diverging_nozzle", 4.0, 0.30)),
    ]
    baseline_time = float(selections[0][1].total_time_s)
    energy_ceiling = float(regime.ode_transition_D3plus_energy_per_deuteron_ceiling_eV)
    rows = []
    for name, row in selections:
        rows.append({
            "approach": name,
            "geometry": row.geometry,
            "Q_multiplier": row.Q_multiplier,
            "coherent_fraction_setting": row.coherent_fraction_setting,
            "charge_time_s": row.charge_time_s,
            "formation_time_s": row.formation_time_s,
            "total_time_s": row.total_time_s,
            "total_time_reduction_fraction": 1.0 - float(row.total_time_s) / baseline_time,
            "corrected_D3_energy_ceiling_eV": energy_ceiling,
        })
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[~summary.case.str.contains("shock_control")].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(selected))
    ax.bar(x, selected.maximum_deuteron_p999_energy_eV)
    ax.axhline(1.0e3, color="#d62728", linestyle="--", label="1 кэВ")
    ax.set_yscale("log")
    ax.set_xticks(x, selected.case, rotation=18, ha="right")
    ax.set_ylabel("p99,9 энергии дейтрона, эВ")
    ax.set_title("Полноэлектронная PIC-проверка уровня 4.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_level45.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--only-case", type=str, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    extrema = pd.read_csv(root / cfg["input"]["extrema_summary_csv"])
    regime = extrema[extrema.search == cfg["input"]["selected_regime"]].iloc[0]
    phases = [float(item) for item in cfg["transfer"]["carrier_phases_rad"]]
    cases = [
        {"name": f"valid_phase_{index}", "phase": phase, "seed_offset": 100 * index}
        for index, phase in enumerate(phases)
    ]
    cases.extend([
        {"name": "valid_phase_0_refined", "phase": 0.0, "grid_cells": 2048,
         "particles_per_cell": 12, "normalized_time_step": 0.04, "seed_offset": 50},
        {"name": "valid_phase_0_long", "phase": 0.0, "normalized_end_time": 7000.0,
         "seed_offset": 75},
        {"name": "cold_wavebreaking_attempt", "phase": 0.0, "field_fraction": 1.0,
         "seed_offset": 80},
        {"name": "deep_cavity_20pct", "phase": 0.0, "depletion": float(cfg["controls"]["deep_cavity_fraction"]), "seed_offset": 500},
        {"name": "double_layer_control", "phase": 0.0, "double_layer_potential_Te": float(cfg["controls"]["imposed_double_layer_potential_Te"]), "field_fraction": 0.0, "seed_offset": 600},
        {"name": "shock_control_M2", "phase": 0.0, "field_fraction": 0.0, "flow_mach": float(cfg["controls"]["converging_flow_mach"]), "seed_offset": 700},
    ])
    if args.only_case is not None:
        cases = [case for case in cases if case["name"] == args.only_case]
        if not cases:
            raise ValueError(f"Unknown case: {args.only_case}")
    summaries = []
    histories = []
    for case in cases:
        result, history = run_pic_case(case, regime, cfg)
        summaries.append(result)
        history.insert(0, "case", case["name"])
        histories.append(history)
    summary = pd.DataFrame(summaries)
    history = pd.concat(histories, ignore_index=True)
    scenario_path = args.output / "level45_pic_scenarios.csv"
    history_path = args.output / "level45_pic_history.csv"
    if args.only_case is not None and scenario_path.exists() and history_path.exists():
        previous_summary = pd.read_csv(scenario_path)
        previous_history = pd.read_csv(history_path)
        previous_summary = previous_summary[~previous_summary.case.isin(summary.case)]
        previous_history = previous_history[~previous_history.case.isin(summary.case)]
        summary = pd.concat([previous_summary, summary], ignore_index=True)
        history = pd.concat([previous_history, history], ignore_index=True)
    geometry = geometry_table(regime, cfg)
    quality = quality_and_onset_scan(regime, cfg, geometry)
    mechanisms = mechanism_bounds(regime)
    summary["quantitatively_converged"] = summary.maximum_absolute_energy_error <= 0.01
    synergy = synergy_table(regime, quality)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(scenario_path, index=False)
    history.to_csv(history_path, index=False)
    geometry.to_csv(args.output / "level45_geometry.csv", index=False)
    quality.to_csv(args.output / "level45_Q_onset_scan.csv", index=False)
    mechanisms.to_csv(args.output / "level45_mechanism_bounds.csv", index=False)
    synergy.to_csv(args.output / "level45_synergy.csv", index=False)
    payload = {
        "scope": "1-D electrostatic kinetic electrons and ions with periodic Poisson solve",
        "selected_regime": cfg["input"]["selected_regime"],
        "pic": summary.to_dict("records"),
        "geometry": geometry.to_dict("records"),
        "mechanism_bounds": mechanisms.to_dict("records"),
        "synergy": synergy.to_dict("records"),
    }
    (args.output / "level45_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(summary, root / "figures/20_level45_pic_tail.png")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
